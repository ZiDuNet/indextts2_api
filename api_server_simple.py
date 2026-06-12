"""
IndexTTS2 API 服务器 - 简化版（可测试）
不需要模型也能测试所有接口
"""
import os
import io
import time
import json
import base64
from typing import List, Optional

from fastapi import FastAPI, Request, Response, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

import uvicorn
import argparse
import numpy as np
import soundfile as sf
from loguru import logger


# ============== 模拟 TTS（生成测试音频）=============
def generate_test_audio(text: str, duration: float = 2.0) -> tuple:
    """生成测试音频（正弦波）"""
    sample_rate = 16000
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    # 生成正弦波作为测试音频
    freq = 440  # A4
    audio = np.sin(2 * np.pi * freq * t) * 0.3
    audio = (audio * 32767).astype(np.int16)
    return sample_rate, audio


# ============== 全局变量 ==============
args = None


# ============== 应用生命周期 ==============
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("IndexTTS2 API (Simple Mode)")
    yield
    logger.info("Shutting down...")


# ============== FastAPI 应用 ==============
app = FastAPI(
    lifespan=lifespan,
    title="IndexTTS2 API",
    version="2.0",
    description="IndexTTS2 零样本语音合成 API (Simplified - 可测试)"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

logger.add("logs/api.log", rotation="10 MB", retention=10, level="DEBUG", enqueue=True)


# ============== 接口 ==============

@app.get("/health")
async def health_check():
    return JSONResponse(status_code=200, content={
        "status": "healthy",
        "mode": "simple",
        "timestamp": time.time()
    })


@app.post("/register_speaker")
async def register_speaker(request: Request):
    try:
        data = await request.json()
        name = data.get("name")
        sample_audios = data.get("sample_audios", [])

        if not name or not sample_audios:
            return JSONResponse(status_code=400, content={"error": "name 和 sample_audios 必须"})

        speaker_path = "assets/speaker.json"
        os.makedirs("assets", exist_ok=True)

        speaker_dict = {}
        if os.path.exists(speaker_path):
            with open(speaker_path, 'r', encoding='utf-8') as f:
                speaker_dict = json.load(f)

        speaker_dict[name] = sample_audios
        with open(speaker_path, 'w', encoding='utf-8') as f:
            json.dump(speaker_dict, f, ensure_ascii=False, indent=2)

        return JSONResponse(content={"status": "success", "message": f"角色 {name} 注册成功"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/audio/voices")
async def get_voices():
    speaker_path = "assets/speaker.json"
    if os.path.exists(speaker_path):
        with open(speaker_path, 'r', encoding='utf-8') as f:
            return JSONResponse(content={"registered_voices": json.load(f)})
    return JSONResponse(content={"registered_voices": {}})


@app.post("/update_speaker")
async def update_speaker(request: Request):
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

        return JSONResponse(content={"status": "success", "message": f"角色 {name} 修改成功"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/delete_speaker")
async def delete_speaker(request: Request):
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

        return JSONResponse(content={"status": "success", "message": f"角色 {name} 删除成功"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/upload_audio")
async def upload_audio(file: UploadFile = File(...)):
    try:
        ALLOWED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac"}
        ext = os.path.splitext(file.filename)[-1].lower()

        if ext not in ALLOWED_EXTENSIONS:
            return JSONResponse(status_code=400, content={"error": f"不支持的后缀: {ext}"})

        os.makedirs("assets", exist_ok=True)
        filename = f"{int(time.time())}_{file.filename}"
        path = f"assets/{filename}"

        with open(path, "wb") as f:
            f.write(await file.read())

        return JSONResponse(content={"status": "success", "file_path": path})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/tts")
async def tts_api(request: Request):
    """语音合成 (二进制) - 测试模式"""
    try:
        data = await request.json()
        text = data.get("text", "")
        spk_audio_prompt = data.get("spk_audio_prompt", data.get("voice"))

        if not text:
            return JSONResponse(status_code=400, content={"error": "text 必须"})

        # 生成测试音频
        duration = min(len(text) * 0.1, 5.0)  # 根据文本长度生成音频
        sr, wav = generate_test_audio(text, duration)

        return Response(content=wav.tobytes(), media_type="application/octet-stream")
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/tts-wav")
async def tts_wav(request: Request):
    """语音合成 (WAV) - 测试模式"""
    try:
        data = await request.json()
        text = data.get("text", "")
        spk_audio_prompt = data.get("spk_audio_prompt", data.get("voice"))

        if not text:
            return JSONResponse(status_code=400, content={"error": "text 必须"})

        # 生成测试音频
        duration = min(len(text) * 0.1, 5.0)
        sr, wav = generate_test_audio(text, duration)

        # 写入 WAV
        with io.BytesIO() as buf:
            sf.write(buf, wav, sr, format='WAV')
            wav_bytes = buf.getvalue()

        return Response(content=wav_bytes, media_type="audio/wav")
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/tts_url")
async def tts_url(request: Request):
    """使用音频路径合成 - 测试模式"""
    try:
        data = await request.json()
        text = data.get("text", "")

        if not text:
            return JSONResponse(status_code=400, content={"error": "text 必须"})

        # 生成测试音频
        duration = min(len(text) * 0.1, 5.0)
        sr, wav = generate_test_audio(text, duration)

        with io.BytesIO() as buf:
            sf.write(buf, wav, sr, format='WAV')
            wav_bytes = buf.getvalue()

        return Response(content=wav_bytes, media_type="audio/wav")
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/audio/speech")
async def openai_compat(request: Request):
    """OpenAI 兼容接口 - 测试模式"""
    try:
        data = await request.json()
        text = data.get("input", "")

        if not text:
            return JSONResponse(status_code=400, content={"error": "input 必须"})

        # 生成测试音频
        duration = min(len(text) * 0.1, 5.0)
        sr, wav = generate_test_audio(text, duration)

        with io.BytesIO() as buf:
            sf.write(buf, wav, sr, format='WAV')
            wav_bytes = buf.getvalue()

        return Response(content=wav_bytes, media_type="audio/wav")
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# ============== WebSocket ==============

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        await ws.send_json({"type": "connected", "message": "IndexTTS2 WebSocket (Test Mode)"})
        while True:
            data = await ws.receive_json()
            msg_type = data.get("type")

            if msg_type == "tts":
                text = data.get("text", "test")
                sr, wav = generate_test_audio(text, 2.0)
                with io.BytesIO() as buf:
                    sf.write(buf, wav, sr, format='WAV')
                    wav_bytes = buf.getvalue()
                await ws.send_json({
                    "type": "completed",
                    "audio_base64": base64.b64encode(wav_bytes).decode(),
                    "sample_rate": sr
                })
            elif msg_type == "ping":
                await ws.send_json({"type": "pong"})
    except WebSocketDisconnect:
        pass


# ============== 主程序 ==============

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IndexTTS2 API Server (Test)")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8002)
    args = parser.parse_args()

    os.makedirs("outputs", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("assets", exist_ok=True)

    logger.info(f"IndexTTS2 API (Simple) - http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)