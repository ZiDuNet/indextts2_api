"""
IndexTTS2 API 服务器
基于官方代码 + 自定义接口
支持 x86 和 ARM64 (GB10)
"""
import os
import sys
import io
import time
import json
import base64
import asyncio
import traceback
from typing import List, Optional, Dict, Any
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import soundfile as sf

# FastAPI 相关
from fastapi import FastAPI, Request, Response, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

import uvicorn
import argparse
from loguru import logger

# 导入官方 TTS 模型
from indextts.infer_v2 import IndexTTS2


# ============== 全局变量 ==============
tts: Optional[IndexTTS2] = None
args: argparse.Namespace = None


# ============== WebSocket 管理 ==============
class ConnectionManager:
    """WebSocket 连接管理器"""

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket 连接数: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(f"WebSocket 连接数: {len(self.active_connections)}")

    async def send_json(self, message: dict, websocket: WebSocket):
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"发送消息失败: {e}")

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass


manager = ConnectionManager()


# ============== 应用生命周期 ==============
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动和关闭生命周期"""
    global tts

    # 启动时初始化
    logger.info("Initializing IndexTTS2...")

    try:
        device = None if args.device == "auto" else args.device
        tts = IndexTTS2(
            cfg_path=os.path.join(args.model_dir, "config.yaml"),
            model_dir=args.model_dir,
            use_fp16=args.fp16,
            device=device,
            use_cuda_kernel=False,
            use_torch_compile=False
        )

        # 预加载角色
        speaker_path = "assets/speaker.json"
        if os.path.exists(speaker_path):
            with open(speaker_path, 'r', encoding='utf-8') as f:
                speaker_dict = json.load(f)
            for speaker, audio_paths in speaker_dict.items():
                tts.registry_speaker(speaker, audio_paths)
            logger.info(f"Loaded {len(speaker_dict)} speakers")

        logger.info("IndexTTS2 initialized successfully")

    except Exception as e:
        logger.warning(f"模型初始化失败，将以有限功能运行: {str(e)}")
        logger.warning("请确保已安装正确的 Python 版本 (3.10-3.11) 并下载模型文件")
        tts = None

    yield

    # 关闭时清理
    logger.info("Shutting down...")


# ============== FastAPI 应用 ==============
app = FastAPI(
    lifespan=lifespan,
    title="IndexTTS2 API",
    version="2.0",
    description="IndexTTS2 零样本语音合成 API - 支持声音克隆和情感控制"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# 日志配置
logger.add("logs/api.log", rotation="10 MB", retention=10, level="DEBUG", enqueue=True)


# ============== 基础接口 ==============

@app.get("/health")
async def health_check():
    """健康检查"""
    if tts is None:
        return JSONResponse(status_code=200, content={
            "status": "partial",
            "model_loaded": False,
            "message": "API 服务运行中，但模型未初始化（需要 Python 3.10-3.11）"
        })
    return JSONResponse(status_code=200, content={
        "status": "healthy",
        "model_loaded": True,
        "timestamp": time.time(),
        "device": tts.device if tts else "unknown"
    })


# ============== 角色管理接口 ==============

@app.post("/register_speaker")
async def register_speaker(request: Request):
    """注册角色"""
    try:
        data = await request.json()
        name = data.get("name")
        sample_audios = data.get("sample_audios", [])

        if not name or not sample_audios:
            return JSONResponse(
                status_code=400,
                content={"error": "name 和 sample_audios 必须"}
            )

        # 保存到 speaker.json
        speaker_path = "assets/speaker.json"
        os.makedirs("assets", exist_ok=True)

        speaker_dict = {}
        if os.path.exists(speaker_path):
            with open(speaker_path, 'r', encoding='utf-8') as f:
                speaker_dict = json.load(f)

        speaker_dict[name] = sample_audios
        with open(speaker_path, 'w', encoding='utf-8') as f:
            json.dump(speaker_dict, f, ensure_ascii=False, indent=2)

        # 注册到模型（如果模型已初始化）
        if tts is not None:
            tts.registry_speaker(name, sample_audios)

        return JSONResponse(content={
            "status": "success",
            "message": f"角色 {name} 注册成功"
        })

    except Exception as e:
        logger.error(f"注册角色失败: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/audio/voices")
async def get_voices():
    """获取已注册角色列表"""
    speaker_path = "assets/speaker.json"
    if os.path.exists(speaker_path):
        with open(speaker_path, 'r', encoding='utf-8') as f:
            return JSONResponse(content={"registered_voices": json.load(f)})
    return JSONResponse(content={"registered_voices": {}})


@app.post("/update_speaker")
async def update_speaker(request: Request):
    """修改角色"""
    try:
        data = await request.json()
        name = data.get("name")
        sample_audios = data.get("sample_audios", [])
        speaker_path = "assets/speaker.json"

        if not os.path.exists(speaker_path):
            return JSONResponse(status_code=404, content={"error": "speaker.json 不存在"})

        with open(speaker_path, 'r', encoding='utf-8') as f:
            speaker_dict = json.load(f)

        if name not in speaker_dict:
            return JSONResponse(status_code=404, content={"error": f"角色 {name} 不存在"})

        speaker_dict[name] = sample_audios
        with open(speaker_path, 'w', encoding='utf-8') as f:
            json.dump(speaker_dict, f, ensure_ascii=False, indent=2)

        # 重新注册（如果模型已初始化）
        if tts is not None:
            if hasattr(tts, 'speaker_embeds') and name in tts.speaker_embeds:
                del tts.speaker_embeds[name]
            tts.registry_speaker(name, sample_audios)

        return JSONResponse(content={
            "status": "success",
            "message": f"角色 {name} 修改成功"
        })

    except Exception as e:
        logger.error(f"修改角色失败: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/delete_speaker")
async def delete_speaker(request: Request):
    """删除角色"""
    try:
        data = await request.json()
        name = data.get("name")
        speaker_path = "assets/speaker.json"

        if not os.path.exists(speaker_path):
            return JSONResponse(status_code=404, content={"error": "speaker.json 不存在"})

        with open(speaker_path, 'r', encoding='utf-8') as f:
            speaker_dict = json.load(f)

        if name not in speaker_dict:
            return JSONResponse(status_code=404, content={"error": f"角色 {name} 不存在"})

        del speaker_dict[name]
        with open(speaker_path, 'w', encoding='utf-8') as f:
            json.dump(speaker_dict, f, ensure_ascii=False, indent=2)

        # 清除缓存
        if hasattr(tts, 'speaker_embeds') and name in tts.speaker_embeds:
            del tts.speaker_embeds[name]

        return JSONResponse(content={
            "status": "success",
            "message": f"角色 {name} 删除成功"
        })

    except Exception as e:
        logger.error(f"删除角色失败: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


# ============== 文件上传 ==============

@app.post("/upload_audio")
async def upload_audio(file: UploadFile = File(...)):
    """音频文件上传"""
    try:
        ALLOWED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac"}
        ext = os.path.splitext(file.filename)[-1].lower()

        if ext not in ALLOWED_EXTENSIONS:
            return JSONResponse(
                status_code=400,
                content={"error": f"不支持的后缀: {ext}"}
            )

        os.makedirs("assets", exist_ok=True)
        filename = f"{int(time.time())}_{file.filename}"
        path = f"assets/{filename}"

        with open(path, "wb") as f:
            f.write(await file.read())

        return JSONResponse(content={
            "status": "success",
            "file_path": path
        })

    except Exception as e:
        logger.error(f"上传失败: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


# ============== TTS 合成接口 ==============

@app.post("/tts")
async def tts_api(request: Request):
    """语音合成 (二进制流)"""
    if tts is None:
        return JSONResponse(status_code=503, content={
            "error": "模型未初始化",
            "message": "请使用 Python 3.10-3.11 并正确配置模型"
        })

    try:
        data = await request.json()
        text = data.get("text", "")
        spk_audio_prompt = data.get("spk_audio_prompt", data.get("voice"))

        emo_audio_prompt = data.get("emo_audio_prompt")
        emo_alpha = data.get("emo_alpha", 1.0)
        emo_vector = data.get("emo_vector")
        use_emo_text = data.get("use_emo_text", False)
        emo_text = data.get("emo_text")

        if not text or not spk_audio_prompt:
            return JSONResponse(
                status_code=400,
                content={"error": "text 和 spk_audio_prompt 必须"}
            )

        output_path = f"outputs/{int(time.time())}.wav"
        os.makedirs("outputs", exist_ok=True)

        sr, wav = tts.infer(
            spk_audio_prompt=spk_audio_prompt,
            text=text,
            output_path=output_path,
            emo_audio_prompt=emo_audio_prompt,
            emo_alpha=emo_alpha,
            emo_vector=emo_vector,
            use_emo_text=use_emo_text,
            emo_text=emo_text
        )

        return Response(content=wav.tobytes(), media_type="application/octet-stream")

    except Exception as e:
        logger.error(f"TTS 失败: {e}")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/tts-wav")
async def tts_wav(request: Request):
    """语音合成 (WAV格式)"""
    if tts is None:
        return JSONResponse(status_code=503, content={
            "error": "模型未初始化",
            "message": "请使用 Python 3.10-3.11 并正确配置模型"
        })

    try:
        data = await request.json()
        text = data.get("text", "")
        spk_audio_prompt = data.get("spk_audio_prompt", data.get("voice"))

        emo_audio_prompt = data.get("emo_audio_prompt")
        emo_alpha = data.get("emo_alpha", 1.0)
        emo_vector = data.get("emo_vector")
        use_emo_text = data.get("use_emo_text", False)
        emo_text = data.get("emo_text")

        if not text or not spk_audio_prompt:
            return JSONResponse(
                status_code=400,
                content={"error": "text 和 spk_audio_prompt 必须"}
            )

        output_path = f"outputs/{int(time.time())}.wav"
        os.makedirs("outputs", exist_ok=True)

        sr, wav = tts.infer(
            spk_audio_prompt=spk_audio_prompt,
            text=text,
            output_path=output_path,
            emo_audio_prompt=emo_audio_prompt,
            emo_alpha=emo_alpha,
            emo_vector=emo_vector,
            use_emo_text=use_emo_text,
            emo_text=emo_text
        )

        # 写入 WAV
        with io.BytesIO() as buf:
            sf.write(buf, wav, sr, format='WAV')
            wav_bytes = buf.getvalue()

        return Response(content=wav_bytes, media_type="audio/wav")

    except Exception as e:
        logger.error(f"TTS 失败: {e}")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/tts_url")
async def tts_url(request: Request):
    """使用音频路径合成"""
    if tts is None:
        return JSONResponse(status_code=503, content={
            "error": "模型未初始化",
            "message": "请使用 Python 3.10-3.11 并正确配置模型"
        })

    try:
        data = await request.json()
        text = data.get("text", "")
        audio_paths = data.get("audio_paths", [])

        if not text:
            return JSONResponse(
                status_code=400,
                content={"error": "text 必须"}
            )

        output_path = f"outputs/{int(time.time())}.wav"
        os.makedirs("outputs", exist_ok=True)

        sr, wav = tts.infer(
            spk_audio_prompt=audio_paths[0] if audio_paths else "examples/voice_01.wav",
            text=text,
            output_path=output_path
        )

        with io.BytesIO() as buf:
            sf.write(buf, wav, sr, format='WAV')
            wav_bytes = buf.getvalue()

        return Response(content=wav_bytes, media_type="audio/wav")

    except Exception as e:
        logger.error(f"TTS 失败: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/audio/speech")
async def openai_compat(request: Request):
    """OpenAI 兼容接口"""
    if tts is None:
        return JSONResponse(status_code=503, content={
            "error": "模型未初始化",
            "message": "请使用 Python 3.10-3.11 并正确配置模型"
        })

    try:
        data = await request.json()
        text = data.get("input", "")
        voice = data.get("voice", "examples/voice_01.wav")

        if not text:
            return JSONResponse(
                status_code=400,
                content={"error": "input 必须"}
            )

        output_path = f"outputs/{int(time.time())}.wav"
        os.makedirs("outputs", exist_ok=True)

        sr, wav = tts.infer(
            spk_audio_prompt=voice,
            text=text,
            output_path=output_path
        )

        with io.BytesIO() as buf:
            sf.write(buf, wav, sr, format='WAV')
            wav_bytes = buf.getvalue()

        return Response(content=wav_bytes, media_type="audio/wav")

    except Exception as e:
        logger.error(f"TTS 失败: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


# ============== WebSocket 接口 ==============

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """WebSocket 流式合成"""
    await manager.connect(ws)
    try:
        await manager.send_json({"type": "connected", "message": "IndexTTS2 WebSocket"}, ws)

        while True:
            data = await ws.receive_json()
            await handle_ws_message(data, ws)

    except WebSocketDisconnect:
        manager.disconnect(ws)


async def handle_ws_message(data: dict, ws: WebSocket):
    """处理 WebSocket 消息"""
    msg_type = data.get("type")

    try:
        if msg_type == "tts":
            text = data.get("text", "")
            spk_audio_prompt = data.get("spk_audio_prompt", data.get("voice", "examples/voice_01.wav"))

            output_path = f"outputs/ws_{int(time.time())}.wav"
            sr, wav = tts.infer(
                spk_audio_prompt=spk_audio_prompt,
                text=text,
                output_path=output_path
            )

            with io.BytesIO() as buf:
                sf.write(buf, wav, sr, format='WAV')
                wav_bytes = buf.getvalue()

            await manager.send_json({
                "type": "completed",
                "audio_base64": base64.b64encode(wav_bytes).decode(),
                "sample_rate": sr
            }, ws)

        elif msg_type == "ping":
            await manager.send_json({"type": "pong"}, ws)

        elif msg_type == "get_voices":
            speaker_path = "assets/speaker.json"
            voices = {}
            if os.path.exists(speaker_path):
                with open(speaker_path, 'r', encoding='utf-8') as f:
                    voices = json.load(f)
            await manager.send_json({"type": "voices_list", "voices": voices}, ws)

        elif msg_type == "tts_stream":
            text = data.get("text", "")
            spk = data.get("character", data.get("voice", "examples/voice_01.wav"))

            await manager.send_json({"type": "stream_started"}, ws)

            output_path = f"outputs/ws_{int(time.time())}.wav"
            sr, wav = tts.infer(spk_audio_prompt=spk, text=text, output_path=output_path)

            with io.BytesIO() as buf:
                sf.write(buf, wav, sr, format='WAV')
                wav_bytes = buf.getvalue()

            await manager.send_json({
                "type": "stream_completed",
                "audio_base64": base64.b64encode(wav_bytes).decode()
            }, ws)

    except Exception as e:
        logger.error(f"WebSocket 处理失败: {e}")
        await manager.send_json({"type": "error", "message": str(e)}, ws)


# ============== 主程序 ==============

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IndexTTS2 API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=8002, help="监听端口")
    parser.add_argument("--model_dir", type=str, default="checkpoints", help="模型目录")
    parser.add_argument("--device", type=str, default="auto", help="设备 (auto/cuda/cpu)")
    parser.add_argument("--fp16", action="store_true", help="使用 FP16")
    args = parser.parse_args()

    # 确保输出目录存在
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("assets", exist_ok=True)

    logger.info(f"IndexTTS2 API Server - http://{args.host}:{args.port}")
    logger.info(f"Device: {args.device}, FP16: {args.fp16}")

    uvicorn.run(app, host=args.host, port=args.port)