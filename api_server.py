"""
IndexTTS2 API 服务器
支持声音克隆、情感控制、音色缓存、推理参数调优
根路由挂载 WebUI，/v1/ 路由提供 API
"""
import os
import sys
import io
import uuid
import time
import json
import base64
import hashlib
import asyncio
import traceback
import shutil
from typing import Optional
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import soundfile as sf

from fastapi import FastAPI, Request, Response, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

import uvicorn
import argparse
from loguru import logger

from indextts.infer_v2 import IndexTTS2


# ============== 全局变量 ==============
tts: Optional[IndexTTS2] = None
args: argparse.Namespace = None
SPEAKER_CACHE_DIR = "assets/speaker_cache"
SPEAKER_META_FILE = os.path.join(SPEAKER_CACHE_DIR, "meta.json")

_gpu_semaphore: Optional[asyncio.Semaphore] = None
_queue_slots: Optional[asyncio.Semaphore] = None
_meta_cache: Optional[dict] = None
_meta_mtime: Optional[float] = None


# ============== 音色缓存管理 ==============

def _load_meta() -> dict:
    global _meta_cache, _meta_mtime
    if os.path.exists(SPEAKER_META_FILE):
        mtime = os.path.getmtime(SPEAKER_META_FILE)
        if _meta_cache is not None and _meta_mtime == mtime:
            return _meta_cache
        with open(SPEAKER_META_FILE, "r", encoding="utf-8") as f:
            _meta_cache = json.load(f)
        _meta_mtime = mtime
        return _meta_cache
    _meta_cache = {}
    _meta_mtime = None
    return _meta_cache


def _save_meta(meta: dict):
    global _meta_cache, _meta_mtime
    _meta_cache = meta
    os.makedirs(SPEAKER_CACHE_DIR, exist_ok=True)
    with open(SPEAKER_META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    _meta_mtime = os.path.getmtime(SPEAKER_META_FILE)


def _md5_file(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ============== 推理参数提取 ==============

def _extract_params(data: dict) -> dict:
    """从请求体中提取推理参数，返回 infer() 需要的 kwargs"""
    def as_bool(value) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    params = {}
    for key in ("emo_audio_prompt", "emo_alpha", "emo_vector", "use_emo_text",
                "emo_text", "use_random"):
        if key in data:
            params[key] = data[key]
    if "emo_alpha" not in params:
        params["emo_alpha"] = 1.0
    else:
        params["emo_alpha"] = float(params["emo_alpha"])
    if "emo_vector" in params:
        v = params["emo_vector"]
        if not isinstance(v, list) or len(v) != 8:
            raise ValueError("emo_vector 必须是长度为 8 的数组 [高兴,愤怒,悲伤,害怕,厌恶,忧郁,惊讶,平静]")
        params["emo_vector"] = [float(x) for x in v]
    for key in ("use_emo_text", "use_random", "do_sample"):
        if key in params:
            params[key] = as_bool(params[key])
        elif key in data:
            params[key] = as_bool(data[key])
    for key in ("interval_silence", "max_text_tokens_per_segment",
                "num_beams", "top_k", "max_mel_tokens", "diffusion_steps"):
        if key in data and data[key] is not None:
            params[key] = int(data[key])
    for key in ("top_p", "temperature", "length_penalty", "repetition_penalty",
                "inference_cfg_rate"):
        if key in data and data[key] is not None:
            params[key] = float(data[key])
    if params.get("repetition_penalty", 10.0) <= 0:
        raise ValueError("repetition_penalty 必须大于 0")
    if params.get("temperature", 0.8) <= 0:
        raise ValueError("temperature 必须大于 0")
    if "top_p" in params and not 0 <= params["top_p"] <= 1:
        raise ValueError("top_p 必须在 0 到 1 之间")
    if "diffusion_steps" in params and params["diffusion_steps"] < 1:
        raise ValueError("diffusion_steps 必须大于 0")
    return params


def _encode_audio(wav: np.ndarray, sr: int, fmt: str) -> tuple[bytes, str]:
    buf = io.BytesIO()
    if fmt == "mp3":
        sf.write(buf, wav, sr, format="MP3")
        return buf.getvalue(), "audio/mpeg"
    elif fmt == "pcm":
        return wav.tobytes(), "audio/pcm"
    else:
        sf.write(buf, wav, sr, format="WAV")
        return buf.getvalue(), "audio/wav"


def _resolve_speaker(data: dict) -> Optional[str]:
    """解析音色：支持 voice_id（缓存）或 spk_audio_prompt（文件路径）"""
    speaker_id = data.get("speaker_id") or data.get("voice")
    if speaker_id:
        meta = _load_meta()
        if speaker_id in meta:
            return meta[speaker_id]["audio_path"]
        return speaker_id
    return data.get("spk_audio_prompt")


def _unique_output_path(prefix: str = "") -> str:
    """生成唯一的输出文件路径（UUID 防并发冲突）"""
    name = f"{prefix}{uuid.uuid4().hex[:8]}" if prefix else uuid.uuid4().hex[:12]
    return f"outputs/{name}.wav"


# ============== 同步推理 ==============

def _do_infer(text: str, spk_audio_prompt: str, output_path: str, params: dict):
    result = tts.infer(
        spk_audio_prompt=spk_audio_prompt,
        text=text,
        output_path=output_path,
        **params,
    )
    if isinstance(result, tuple) and len(result) == 2:
        return result
    if isinstance(result, str) and os.path.exists(result):
        wav, sr = sf.read(result, dtype="int16")
        return sr, wav
    if result is None and output_path and os.path.exists(output_path):
        wav, sr = sf.read(output_path, dtype="int16")
        return sr, wav
    return result


async def _acquire_with_timeout(semaphore: asyncio.Semaphore, timeout: float) -> bool:
    try:
        await asyncio.wait_for(semaphore.acquire(), timeout=timeout)
        return True
    except asyncio.TimeoutError:
        return False


async def _guarded_infer(text: str, spk: str, output_path: str, params: dict):
    """带信号量保护的推理，控制 GPU 并发"""
    request_start = time.perf_counter()
    if _queue_slots is None or _gpu_semaphore is None:
        raise RuntimeError("推理队列尚未初始化")

    accepted = await _acquire_with_timeout(_queue_slots, args.queue_timeout)
    if not accepted:
        raise TimeoutError(f"推理队列已满，请稍后重试（等待超过 {args.queue_timeout:.1f}s）")

    try:
        await _gpu_semaphore.acquire()
        queue_elapsed = time.perf_counter() - request_start
        infer_start = time.perf_counter()
        try:
            result = await asyncio.to_thread(_do_infer, text, spk, output_path, params)
        finally:
            _gpu_semaphore.release()
        return {
            "result": result,
            "queue_time": queue_elapsed,
            "infer_time": time.perf_counter() - infer_start,
            "total_time": time.perf_counter() - request_start,
        }
    finally:
        _queue_slots.release()


# ============== WebSocket 管理 ==============

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active_connections.append(ws)
        logger.info(f"WebSocket 连接数: {len(self.active_connections)}")

    def disconnect(self, ws: WebSocket):
        if ws in self.active_connections:
            self.active_connections.remove(ws)
            logger.info(f"WebSocket 连接数: {len(self.active_connections)}")

    async def send_json(self, message: dict, ws: WebSocket):
        try:
            await ws.send_json(message)
        except Exception as e:
            logger.error(f"发送消息失败: {e}")


manager = ConnectionManager()


# ============== 主模型自动下载 ==============

def _ensure_main_model(model_dir: str):
    """主模型缺失时从 ModelScope 自动补全（已存在则跳过）"""
    cfg_path = os.path.join(model_dir, "config.yaml")
    if os.path.exists(cfg_path):
        return

    logger.info(f"主模型不存在，从 ModelScope 下载到 {model_dir} ...")
    # Linux 服务器上默认写到 $HOME/.cache/modelscope，Windows 上写到 C 盘用户目录；
    # 都可通过 MODELSCOPE_CACHE 环境变量覆盖。
    default_cache = os.path.join(
        os.path.expanduser("~"), ".cache", "modelscope"
    )
    ms_cache = os.environ.get("MODELSCOPE_CACHE", default_cache)
    os.makedirs(ms_cache, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    try:
        from modelscope.hub.snapshot_download import snapshot_download as ms_snapshot
        # exist_ok=True 让 modelscope 在文件已存在时直接复用，不重复下载
        ms_snapshot(model_id="IndexTeam/IndexTTS-2", cache_dir=ms_cache, exist_ok=True)

        # 从缓存目录复制到 model_dir
        src_dir = os.path.join(ms_cache, "IndexTeam", "IndexTTS-2")
        if not os.path.isdir(src_dir):
            for root, dirs, files in os.walk(ms_cache):
                if "config.yaml" in files:
                    src_dir = root
                    break
        if os.path.isdir(src_dir):
            for item in os.listdir(src_dir):
                src = os.path.join(src_dir, item)
                dst = os.path.join(model_dir, item)
                if os.path.exists(dst):
                    continue
                if os.path.isdir(src):
                    shutil.copytree(src, dst)
                else:
                    shutil.copy2(src, dst)
            logger.info(f"主模型下载完成: {model_dir}")
    except Exception as e:
        logger.error(f"主模型下载失败: {e}")


# ============== 应用生命周期 ==============

@asynccontextmanager
async def lifespan(app: FastAPI):
    global tts, _gpu_semaphore, _queue_slots
    _gpu_semaphore = asyncio.Semaphore(args.max_concurrency)
    _queue_slots = asyncio.Semaphore(args.max_concurrency + args.queue_size)

    # 自动下载主模型（缓存到 H 盘）
    _ensure_main_model(args.model_dir)

    logger.info("Initializing IndexTTS2...")

    try:
        device = None if args.device == "auto" else args.device
        tts = IndexTTS2(
            cfg_path=os.path.join(args.model_dir, "config.yaml"),
            model_dir=args.model_dir,
            use_fp16=args.fp16,
            device=device,
            use_cuda_kernel=False,
            use_deepspeed=args.deepspeed,
            use_accel=args.accel,
            use_torch_compile=False,
        )
        logger.info("IndexTTS2 initialized successfully")
    except Exception as e:
        logger.warning(f"模型初始化失败，将以有限功能运行: {e}")
        tts = None

    # 加载音色缓存到内存
    _load_meta()
    logger.info(f"已加载 {len(_meta_cache or {})} 个音色")

    yield
    logger.info("Shutting down...")


# ============== FastAPI 应用 ==============

app = FastAPI(
    lifespan=lifespan,
    title="IndexTTS2 API",
    version="2.1",
    description="IndexTTS2 零样本语音合成 API — 声音克隆 · 情感控制 · 音色缓存 · 推理参数调优",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.add("logs/api.log", rotation="10 MB", retention=10, level="DEBUG", enqueue=True)


# ============== 健康检查 ==============

@app.get("/health", tags=["基础"])
async def health_check():
    """健康检查"""
    if tts is None:
        return {"status": "partial", "model_loaded": False,
                "message": "模型未初始化"}
    return {"status": "healthy", "model_loaded": True,
            "timestamp": time.time(), "device": str(tts.device),
            "max_concurrency": args.max_concurrency,
            "queue_size": args.queue_size,
            "queue_timeout": args.queue_timeout}


@app.get("/ws/docs", tags=["WebSocket"])
async def websocket_docs():
    """WebSocket 接口说明。OpenAPI 不原生展示 ws:// 路由，因此用 HTTP 端点暴露文档。"""
    return {
        "endpoint": "/ws",
        "protocol": "websocket",
        "url_example": "ws://localhost:8002/ws",
        "message_types": {
            "tts": "普通 WebSocket 合成，完成后一次性返回 base64 WAV",
            "tts_stream": "流式合成入口，当前实现完成后返回 base64 WAV",
            "ping": "心跳检测，返回 pong",
            "get_voices": "返回当前音色元数据",
        },
        "tts_request_example": {
            "type": "tts",
            "text": "你好，这是 WebSocket 合成测试。",
            "voice": "spk_xxxxxxxx",
            "num_beams": 1,
            "do_sample": False,
            "top_k": 10,
            "top_p": 0.8,
            "temperature": 0.8,
            "max_mel_tokens": 900,
            "diffusion_steps": 12,
            "repetition_penalty": 10.0,
        },
        "tts_response_example": {
            "type": "completed",
            "audio_base64": "<wav base64>",
            "sample_rate": 22050,
            "queue_time": 0.0,
            "infer_time": 1.23,
            "total_time": 1.23,
        },
        "supported_speech_params": [
            "voice",
            "text",
            "speaker_id",
            "spk_audio_prompt",
            "emo_audio_prompt",
            "emo_alpha",
            "emo_vector",
            "use_emo_text",
            "emo_text",
            "use_random",
            "interval_silence",
            "max_text_tokens_per_segment",
            "num_beams",
            "do_sample",
            "top_k",
            "top_p",
            "temperature",
            "max_mel_tokens",
            "length_penalty",
            "repetition_penalty",
            "diffusion_steps",
            "inference_cfg_rate",
        ],
    }


# ============== 音色管理 (/v1/audio/voices) ==============

@app.post("/v1/audio/voices", tags=["音色管理"])
async def upload_speaker(
    audio: UploadFile = File(..., description="音色参考音频（3-10秒最佳）"),
    speaker_name: str = Form("", description="音色名称（可选）"),
):
    """上传音频并注册音色，返回 voice_id 用于后续合成。"""
    if tts is None:
        return JSONResponse(status_code=503, content={"error": "模型未初始化"})

    try:
        if not audio.filename:
            return JSONResponse(status_code=400, content={"error": "请提供音频文件"})

        ext = os.path.splitext(audio.filename)[-1].lower()
        if ext not in {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac"}:
            return JSONResponse(status_code=400, content={"error": f"不支持的格式: {ext}，支持: mp3, wav, m4a, flac, ogg, aac"})

        audio_bytes = await audio.read()
        if len(audio_bytes) < 1024:
            return JSONResponse(status_code=400, content={"error": "音频文件过小，请上传 3-10 秒的参考音频"})
        if len(audio_bytes) > 20 * 1024 * 1024:
            return JSONResponse(status_code=400, content={"error": "音频文件过大（超过 20MB），请压缩后上传"})

        os.makedirs(SPEAKER_CACHE_DIR, exist_ok=True)
        tmp_path = os.path.join(SPEAKER_CACHE_DIR, f"tmp_{uuid.uuid4().hex[:8]}{ext}")
        with open(tmp_path, "wb") as f:
            f.write(audio_bytes)

        md5 = _md5_file(tmp_path)
        voice_id = f"spk_{md5[:8]}"

        meta = _load_meta()
        if voice_id in meta:
            os.remove(tmp_path)
            return {"voice_id": voice_id, "status": "exists", "message": "该音频已注册"}

        final_path = os.path.join(SPEAKER_CACHE_DIR, f"{voice_id}{ext}")
        os.rename(tmp_path, final_path)

        # 预热：通过一次短推理缓存 speaker embedding
        try:
            warmup_path = _unique_output_path("warmup_")
            os.makedirs("outputs", exist_ok=True)
            await _guarded_infer("测试", final_path, warmup_path, {})
            if os.path.exists(warmup_path):
                os.remove(warmup_path)
        except Exception as warmup_err:
            logger.warning(f"音色预热失败（不影响使用）: {warmup_err}")

        meta[voice_id] = {
            "voice_name": speaker_name or voice_id,
            "audio_path": final_path,
            "md5": md5,
            "original_filename": audio.filename,
            "created_at": time.time(),
            "embedding_cached": True,
        }
        _save_meta(meta)

        logger.info(f"音色注册成功: {voice_id} ({speaker_name})")
        return {"voice_id": voice_id, "md5": md5, "status": "new",
                "message": "音色注册成功，可使用 /v1/audio/speech 的 voice 参数合成语音"}

    except Exception as e:
        logger.error(f"上传音色失败: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/v1/audio/voices", tags=["音色管理"])
async def list_speakers():
    """获取所有已注册的音色列表"""
    meta = _load_meta()
    voices = []
    for vid, info in meta.items():
        voices.append({
            "voice_id": vid,
            "name": info.get("voice_name", ""),
            "original_filename": info.get("original_filename", ""),
            "created_at": info.get("created_at"),
        })
    return {"object": "list", "data": voices}


@app.delete("/v1/audio/voices/{voice_id}", tags=["音色管理"])
async def delete_speaker(voice_id: str):
    """删除指定音色"""
    meta = _load_meta()
    if voice_id not in meta:
        return JSONResponse(status_code=404, content={"error": f"音色 {voice_id} 不存在"})

    info = meta.pop(voice_id)
    _save_meta(meta)

    audio_path = info.get("audio_path", "")
    if audio_path and os.path.exists(audio_path):
        os.remove(audio_path)

    logger.info(f"音色已删除: {voice_id}")
    return {"status": "deleted", "voice_id": voice_id}


# ============== 语音合成 (/v1/audio/speech) ==============

@app.post("/v1/audio/speech", tags=["语音合成"])
async def openai_speech(request: Request):
    """语音合成（OpenAI Speech API 规范）。"""
    if tts is None:
        return JSONResponse(status_code=503, content={"error": "模型未初始化"})

    try:
        data = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "请求体必须是有效 JSON"})

    text = data.get("input", "")
    if not text or not isinstance(text, str) or not text.strip():
        return JSONResponse(status_code=400, content={"error": "input 为必需参数，且不能为空"})

    try:
        params = _extract_params(data)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

    voice = data.get("voice", "")
    meta = _load_meta()
    spk_audio_prompt = meta.get(voice, {}).get("audio_path", voice) if voice else "examples/voice_01.wav"

    if not os.path.exists(spk_audio_prompt):
        return JSONResponse(status_code=400, content={"error": f"音色文件不存在: {voice}，请先通过 /v1/audio/voices 上传"})

    output_format = data.get("response_format", "wav")
    if output_format not in ("wav", "mp3", "pcm"):
        return JSONResponse(status_code=400, content={"error": f"不支持的格式: {output_format}，可选: wav, mp3, pcm"})

    try:
        output_path = _unique_output_path()
        os.makedirs("outputs", exist_ok=True)

        infer_record = await _guarded_infer(text.strip(), spk_audio_prompt, output_path, params)
        result = infer_record["result"]

        if result is None:
            return JSONResponse(status_code=500, content={"error": "合成失败，模型返回空结果"})

        sr, wav = result
        logger.info(
            f"TTS 完成: {len(text)}字, queue={infer_record['queue_time']:.2f}s, "
            f"infer={infer_record['infer_time']:.2f}s, 格式: {output_format}"
        )

        audio_bytes, media_type = _encode_audio(wav, sr, output_format)
        return Response(
            content=audio_bytes,
            media_type=media_type,
            headers={
                "X-IndexTTS-Voice": voice,
                "X-IndexTTS-Sample-Rate": str(sr),
                "X-IndexTTS-Queue-Time": f"{infer_record['queue_time']:.3f}",
                "X-IndexTTS-Infer-Time": f"{infer_record['infer_time']:.3f}",
                "X-IndexTTS-Total-Time": f"{infer_record['total_time']:.3f}",
                "X-IndexTTS-Output-Format": output_format,
            },
        )

    except Exception as e:
        logger.error(f"TTS 失败: {e}")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": f"合成失败: {str(e)}"})


# ============== WebSocket ==============

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """WebSocket 流式合成"""
    await manager.connect(ws)
    try:
        await manager.send_json({"type": "connected", "message": "IndexTTS2 WebSocket"}, ws)

        while True:
            data = await ws.receive_json()
            msg_type = data.get("type")

            try:
                if msg_type in ("tts", "tts_stream"):
                    if tts is None:
                        await manager.send_json({"type": "error", "message": "模型未初始化"}, ws)
                        continue

                    text = data.get("text", "")
                    if not text or not text.strip():
                        await manager.send_json({"type": "error", "message": "text 不能为空"}, ws)
                        continue

                    try:
                        params = _extract_params(data)
                    except ValueError as e:
                        await manager.send_json({"type": "error", "message": str(e)}, ws)
                        continue

                    spk = _resolve_speaker(data) or "examples/voice_01.wav"

                    if msg_type == "tts_stream":
                        await manager.send_json({"type": "stream_started"}, ws)

                    output_path = _unique_output_path("ws_")
                    infer_record = await _guarded_infer(text, spk, output_path, params)
                    result = infer_record["result"]

                    if result:
                        sr, wav = result
                        with io.BytesIO() as buf:
                            sf.write(buf, wav, sr, format="WAV")
                            wav_bytes = buf.getvalue()

                        await manager.send_json({
                            "type": "stream_completed" if msg_type == "tts_stream" else "completed",
                            "audio_base64": base64.b64encode(wav_bytes).decode(),
                            "sample_rate": sr,
                            "queue_time": round(infer_record["queue_time"], 3),
                            "infer_time": round(infer_record["infer_time"], 3),
                            "total_time": round(infer_record["total_time"], 3),
                        }, ws)
                    else:
                        await manager.send_json({"type": "error", "message": "合成失败"}, ws)

                elif msg_type == "ping":
                    await manager.send_json({"type": "pong"}, ws)

                elif msg_type == "get_voices":
                    await manager.send_json({"type": "voices_list", "voices": _load_meta()}, ws)

            except Exception as e:
                logger.error(f"WebSocket 消息处理失败: {e}")
                await manager.send_json({"type": "error", "message": str(e)}, ws)

    except WebSocketDisconnect:
        manager.disconnect(ws)


# ============== 挂载 WebUI 到根路由 ==============

import gradio as gr
from webui_enhanced import create_webui


def _get_tts():
    return tts


gradio_app = create_webui(_get_tts)
app = gr.mount_gradio_app(app, gradio_app, path="/")


# ============== 主程序 ==============

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IndexTTS2 API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=8002, help="监听端口")
    parser.add_argument("--model_dir", type=str, default="checkpoints/IndexTTS-2", help="模型目录")
    parser.add_argument("--device", type=str, default="auto", help="设备 (auto/cuda/cpu)")
    parser.add_argument("--fp16", action="store_true", help="使用 FP16")
    parser.add_argument(
        "--max_concurrency",
        type=int,
        default=int(os.environ.get("INDEXTTS_MAX_CONCURRENCY", "1")),
        help="同时进入模型推理的请求数。单 GPU 建议从 1 开始。",
    )
    parser.add_argument(
        "--queue_size",
        type=int,
        default=int(os.environ.get("INDEXTTS_QUEUE_SIZE", "16")),
        help="推理队列等待名额数，超过后返回队列超时错误。",
    )
    parser.add_argument(
        "--queue_timeout",
        type=float,
        default=float(os.environ.get("INDEXTTS_QUEUE_TIMEOUT", "120")),
        help="请求等待进入推理队列的最长秒数。",
    )
    parser.add_argument("--deepspeed", action="store_true", help="启用 DeepSpeed GPT 推理加速")
    parser.add_argument("--accel", action="store_true", help="启用项目自带 GPT accel engine")
    args = parser.parse_args()
    args.max_concurrency = max(1, args.max_concurrency)
    args.queue_size = max(0, args.queue_size)
    args.queue_timeout = max(0.1, args.queue_timeout)

    os.makedirs("outputs", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs(SPEAKER_CACHE_DIR, exist_ok=True)

    logger.info(f"IndexTTS2 API Server - http://{args.host}:{args.port}")
    logger.info(f"  WebUI: http://{args.host}:{args.port}/")
    logger.info(f"  API:   http://{args.host}:{args.port}/docs")
    logger.info(f"Device: {args.device}, FP16: {args.fp16}")

    uvicorn.run(app, host=args.host, port=args.port, workers=1)
