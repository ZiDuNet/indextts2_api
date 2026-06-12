"""
IndexTTS2 中文 API 控制台。

页面通过真实 HTTP/WebSocket 接口调用服务，方便同时测试音色管理、语音合成、
WebSocket 合成、请求体和响应头回显。
"""
import argparse
import asyncio
import base64
import json
import os
import threading
import time
import uuid
from pathlib import Path

import gradio as gr
import requests


DEFAULT_API_BASE = os.environ.get("INDEXTTS_API_BASE", "http://127.0.0.1:8002").rstrip("/")
WS_SESSIONS: dict[str, "WebSocketSession"] = {}
WS_LOCK = threading.Lock()


def _pretty(data) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def _api_base(value: str | None) -> str:
    return (value or DEFAULT_API_BASE).rstrip("/")


def _voice_choices(voices: list[dict]) -> list[tuple[str, str]]:
    choices = []
    for item in voices:
        voice_id = item.get("voice_id", "")
        name = item.get("name") or voice_id
        filename = item.get("original_filename") or "-"
        choices.append((f"{name}｜{voice_id}｜{filename}", voice_id))
    return choices


def _headers_dict(headers) -> dict:
    keep = {}
    for key, value in headers.items():
        lower = key.lower()
        if lower.startswith("x-indextts") or lower == "content-type":
            keep[key] = value
    return keep


def _save_response(content: bytes, response_format: str, prefix: str) -> str:
    ext = {
        "mp3": "mp3",
        "opus": "opus",
        "aac": "aac",
        "flac": "flac",
        "wav": "wav",
        "pcm": "pcm",
    }.get(response_format, "bin")
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.{ext}"
    path.write_bytes(content)
    return str(path)


def _optional_text(value: str | None):
    value = (value or "").strip()
    return value or None


def _request_echo(method: str, url: str, body: dict | None = None) -> str:
    data = {"方法": method, "地址": url}
    if body is not None:
        data["请求体"] = body
    return _pretty(data)


def _response_echo(status_code: int, elapsed: float, headers, body=None, output_file=None, byte_count=None):
    data = {
        "状态码": status_code,
        "耗时秒": round(elapsed, 3),
        "响应头": _headers_dict(headers),
    }
    if output_file:
        data["输出文件"] = output_file
    if byte_count is not None:
        data["字节数"] = byte_count
    if body is not None:
        data["返回体"] = body
    return data


def _websocket_url(api_base: str) -> str:
    ws_base = _api_base(api_base).replace("http://", "ws://").replace("https://", "wss://")
    return f"{ws_base}/ws"


class WebSocketSession:
    def __init__(self, api_base: str):
        self.api_base = _api_base(api_base)
        self.url = _websocket_url(api_base)
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()
        self.websocket = None
        self.connected_message = None
        self.message_count = 0
        future = asyncio.run_coroutine_threadsafe(self._connect(), self.loop)
        future.result(timeout=30)

    async def _connect(self):
        import websockets

        self.websocket = await websockets.connect(self.url, open_timeout=30)
        self.connected_message = json.loads(await self.websocket.recv())

    async def _send(self, payload: dict):
        await self.websocket.send(json.dumps(payload, ensure_ascii=False))
        while True:
            message = json.loads(await self.websocket.recv())
            if message.get("type") in {"stream_started"}:
                continue
            if message.get("type") in {"completed", "stream_completed", "error", "pong", "voices_list"}:
                self.message_count += 1
                return message

    def send(self, payload: dict):
        future = asyncio.run_coroutine_threadsafe(self._send(payload), self.loop)
        return future.result(timeout=900)

    async def _close(self):
        if self.websocket is not None:
            await self.websocket.close()

    def close(self):
        try:
            future = asyncio.run_coroutine_threadsafe(self._close(), self.loop)
            future.result(timeout=10)
        finally:
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.thread.join(timeout=5)


def _close_ws_session(session_id: str | None):
    if not session_id:
        return
    with WS_LOCK:
        session = WS_SESSIONS.pop(session_id, None)
    if session:
        session.close()


def build_speech_payload(
    text,
    voice,
    response_format,
    emo_audio_prompt,
    emo_alpha,
    send_emo_vector,
    happy,
    angry,
    sad,
    afraid,
    disgusted,
    melancholic,
    surprised,
    calm,
    use_emo_text,
    emo_text,
    use_random,
    interval_silence,
    max_text_tokens_per_segment,
    num_beams,
    do_sample,
    top_k,
    top_p,
    temperature,
    max_mel_tokens,
    length_penalty,
    repetition_penalty,
    diffusion_steps,
    inference_cfg_rate,
):
    if not text or not str(text).strip():
        raise gr.Error("请输入要合成的文本")
    if not voice:
        raise gr.Error("请先上传或选择音色")

    payload = {
        "input": str(text).strip(),
        "voice": voice,
        "response_format": response_format,
        "emo_alpha": float(emo_alpha),
        "use_emo_text": bool(use_emo_text),
        "use_random": bool(use_random),
        "interval_silence": int(interval_silence),
        "max_text_tokens_per_segment": int(max_text_tokens_per_segment),
        "num_beams": int(num_beams),
        "do_sample": bool(do_sample),
        "top_k": int(top_k),
        "top_p": float(top_p),
        "temperature": float(temperature),
        "max_mel_tokens": int(max_mel_tokens),
        "length_penalty": float(length_penalty),
        "repetition_penalty": float(repetition_penalty),
        "diffusion_steps": int(diffusion_steps),
        "inference_cfg_rate": float(inference_cfg_rate),
    }

    emo_audio = _optional_text(emo_audio_prompt)
    if emo_audio:
        payload["emo_audio_prompt"] = emo_audio

    emo_text = _optional_text(emo_text)
    if emo_text:
        payload["emo_text"] = emo_text

    if send_emo_vector:
        payload["emo_vector"] = [
            float(happy),
            float(angry),
            float(sad),
            float(afraid),
            float(disgusted),
            float(melancholic),
            float(surprised),
            float(calm),
        ]

    return payload


def health_check(api_base):
    url = f"{_api_base(api_base)}/health"
    started = time.perf_counter()
    try:
        resp = requests.get(url, timeout=30)
        elapsed = time.perf_counter() - started
        body = resp.json() if resp.content else {}
        return _request_echo("GET", url), _response_echo(resp.status_code, elapsed, resp.headers, body=body)
    except Exception as exc:
        return _request_echo("GET", url), {"错误": str(exc)}


def refresh_voices(api_base):
    url = f"{_api_base(api_base)}/v1/audio/voices"
    try:
        resp = requests.get(url, timeout=30)
        body = resp.json()
        choices = _voice_choices(body.get("data", []))
        value = choices[0][1] if choices else None
        update = gr.update(choices=choices, value=value)
        return update, update, update, update, _response_echo(resp.status_code, 0, resp.headers, body=body)
    except Exception as exc:
        return gr.update(), gr.update(), gr.update(), gr.update(), {"错误": str(exc)}


def upload_voice(api_base, audio_file, speaker_name):
    if not audio_file:
        return gr.update(), gr.update(), gr.update(), gr.update(), {"错误": "请先选择音色参考音频"}

    url = f"{_api_base(api_base)}/v1/audio/voices"
    started = time.perf_counter()
    with open(audio_file, "rb") as file_obj:
        files = {"audio": (Path(audio_file).name, file_obj)}
        data = {"speaker_name": speaker_name or ""}
        resp = requests.post(url, files=files, data=data, timeout=600)
    elapsed = time.perf_counter() - started
    body = resp.json() if resp.content else {}

    manage_update, http_update, ws_update, delete_update, voices_result = refresh_voices(api_base)
    voice_id = body.get("voice_id")
    if voice_id:
        manage_update["value"] = voice_id
        http_update["value"] = voice_id
        ws_update["value"] = voice_id
        delete_update["value"] = voice_id

    result = _response_echo(resp.status_code, elapsed, resp.headers, body=body)
    result["音色列表刷新"] = voices_result
    return manage_update, http_update, ws_update, delete_update, result


def delete_voice(api_base, voice_id):
    if not voice_id:
        return gr.update(), gr.update(), gr.update(), gr.update(), {"错误": "请选择要删除的音色"}

    url = f"{_api_base(api_base)}/v1/audio/voices/{voice_id}"
    started = time.perf_counter()
    resp = requests.delete(url, timeout=60)
    elapsed = time.perf_counter() - started
    body = resp.json() if resp.content else {}

    manage_update, http_update, ws_update, delete_update, voices_result = refresh_voices(api_base)
    result = _response_echo(resp.status_code, elapsed, resp.headers, body=body)
    result["音色列表刷新"] = voices_result
    return manage_update, http_update, ws_update, delete_update, result


def synthesize_via_api(api_base, api_mode, openai_model, *values):
    payload = build_speech_payload(*values)
    if "普通接口" in api_mode:
        url = f"{_api_base(api_base)}/tts"
        payload = dict(payload)
        payload["text"] = payload.pop("input")
    else:
        url = f"{_api_base(api_base)}/v1/audio/speech"
        model = _optional_text(openai_model)
        if model:
            payload["model"] = model
    started = time.perf_counter()
    request_echo = _request_echo("POST", url, payload)

    try:
        resp = requests.post(url, json=payload, timeout=900)
        elapsed = time.perf_counter() - started
        content_type = resp.headers.get("content-type", "")

        if resp.status_code >= 400 or content_type.startswith("application/json"):
            body = resp.json() if resp.content else {}
            return None, None, request_echo, _response_echo(resp.status_code, elapsed, resp.headers, body=body)

        output_path = _save_response(resp.content, payload["response_format"], "api_tts")
        playable_audio = output_path if payload["response_format"] in ("wav", "mp3") else None
        return playable_audio, output_path, request_echo, _response_echo(
            resp.status_code,
            elapsed,
            resp.headers,
            output_file=output_path,
            byte_count=len(resp.content),
        )
    except Exception as exc:
        return None, None, request_echo, {"错误": str(exc)}


def ws_connect(api_base, session_id):
    _close_ws_session(session_id)
    request_echo = _request_echo("WS CONNECT", _websocket_url(api_base))
    started = time.perf_counter()
    try:
        session = WebSocketSession(api_base)
        session_id = uuid.uuid4().hex
        with WS_LOCK:
            WS_SESSIONS[session_id] = session
        return (
            session_id,
            gr.update(value="已连接", interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=True),
            gr.update(interactive=True),
            request_echo,
            {
                "状态": "已连接",
                "地址": session.url,
                "耗时秒": round(time.perf_counter() - started, 3),
                "服务端回显": session.connected_message,
            },
        )
    except Exception as exc:
        return (
            None,
            gr.update(value="未连接", interactive=True),
            gr.update(interactive=True),
            gr.update(interactive=False),
            gr.update(interactive=False),
            request_echo,
            {"错误": str(exc)},
        )


def ws_disconnect(session_id):
    _close_ws_session(session_id)
    return (
        None,
        gr.update(value="未连接", interactive=True),
        gr.update(interactive=True),
        gr.update(interactive=False),
        gr.update(interactive=False),
        _request_echo("WS CLOSE", "/ws"),
        {"状态": "已断开"},
    )


def ws_send_tts(api_base, session_id, message_type, *values):
    if not session_id:
        raise gr.Error("请先建立 WebSocket 连接")
    with WS_LOCK:
        session = WS_SESSIONS.get(session_id)
    if session is None:
        raise gr.Error("WebSocket 会话已失效，请重新连接")

    payload = build_speech_payload(*values)
    payload["type"] = message_type
    payload["text"] = payload.pop("input")
    started = time.perf_counter()
    request_echo = _request_echo("WS SEND", session.url, payload)

    try:
        message = session.send(payload)
        elapsed = time.perf_counter() - started

        audio_path = None
        playable_audio = None
        response_format = message.get("format") or payload.get("response_format", "wav")
        if message.get("audio_base64"):
            audio_path = _save_response(base64.b64decode(message["audio_base64"]), response_format, "ws_tts")
            playable_audio = audio_path if response_format in ("wav", "mp3") else None
            message = {key: value for key, value in message.items() if key != "audio_base64"}
            message["audio_file"] = audio_path

        return playable_audio, audio_path, request_echo, {
            "耗时秒": round(elapsed, 3),
            "连接地址": session.url,
            "会话消息数": session.message_count,
            "消息回显": message,
        }
    except Exception as exc:
        return None, None, request_echo, {"错误": str(exc)}


def apply_fast_preset():
    return 1, False, 10, 0.8, 0.8, 900, 0.0, 10.0, 12, 0.7


def apply_quality_preset():
    return 3, True, 30, 0.8, 0.8, 1500, 0.0, 10.0, 25, 0.7


def create_webui(get_tts):
    css = """
    .indextts-console .gradio-container { max-width: 1480px; }
    .indextts-console textarea, .indextts-console pre {
      font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
    }
    .indextts-console .block-info { font-size: 13px; }
    """

    with gr.Blocks(
        title="IndexTTS2 控制台",
        css=css,
        fill_height=True,
        elem_classes=["indextts-console"],
    ) as demo:
        gr.Markdown("# IndexTTS2 控制台")

        with gr.Row():
            api_base = gr.Textbox(label="服务地址", value=DEFAULT_API_BASE, scale=3)
            health_btn = gr.Button("检查服务", scale=1)
            refresh_btn = gr.Button("刷新音色", scale=1)

        with gr.Row():
            health_request = gr.Code(label="健康检查请求", language="json", lines=4)
            health_response = gr.JSON(label="健康检查回显")

        with gr.Tabs():
            with gr.Tab("音色管理"):
                with gr.Row():
                    with gr.Column(scale=2):
                        upload_audio = gr.Audio(label="参考音频", type="filepath", sources=["upload", "microphone"])
                        speaker_name = gr.Textbox(label="音色名称", placeholder="例如：intro-01-tts")
                        upload_btn = gr.Button("上传并注册音色", variant="primary")
                    with gr.Column(scale=2):
                        manage_voice_dropdown = gr.Dropdown(label="已注册音色", choices=[], interactive=True)
                        delete_dropdown = gr.Dropdown(label="待删除音色", choices=[], interactive=True)
                        delete_btn = gr.Button("删除选中音色", variant="stop")
                        voices_response = gr.JSON(label="音色接口回显")

            with gr.Tab("语音合成"):
                with gr.Row():
                    with gr.Column(scale=3):
                        text = gr.Textbox(
                            label="合成文本",
                            value="这是一次 IndexTTS 二代接口测试。",
                            lines=4,
                        )
                        http_voice_dropdown = gr.Dropdown(label="合成音色", choices=[], interactive=True)
                        api_mode = gr.Radio(
                            ["OpenAI 协议（/v1/audio/speech）", "普通接口（/tts）"],
                            value="OpenAI 协议（/v1/audio/speech）",
                            label="接口类型",
                        )
                        openai_model = gr.Textbox(
                            label="OpenAI model（可选，不校验）",
                            value="gpt-4o-mini-tts",
                            placeholder="可留空；本地服务不会强制校验 model",
                        )
                        response_format = gr.Radio(
                            ["mp3", "opus", "aac", "flac", "wav", "pcm"],
                            value="wav",
                            label="输出格式",
                        )
                        emo_audio_prompt = gr.Textbox(label="情感参考音频路径", placeholder="可选，填写服务端可访问路径")

                        with gr.Accordion("情感控制", open=False):
                            emo_alpha = gr.Slider(0, 1, value=1.0, step=0.05, label="情感强度")
                            send_emo_vector = gr.Checkbox(label="发送 8 维情感向量", value=False)
                            with gr.Row():
                                happy = gr.Slider(0, 1, value=0, step=0.05, label="高兴")
                                angry = gr.Slider(0, 1, value=0, step=0.05, label="愤怒")
                                sad = gr.Slider(0, 1, value=0, step=0.05, label="悲伤")
                                afraid = gr.Slider(0, 1, value=0, step=0.05, label="害怕")
                            with gr.Row():
                                disgusted = gr.Slider(0, 1, value=0, step=0.05, label="厌恶")
                                melancholic = gr.Slider(0, 1, value=0, step=0.05, label="忧郁")
                                surprised = gr.Slider(0, 1, value=0, step=0.05, label="惊讶")
                                calm = gr.Slider(0, 1, value=0, step=0.05, label="平静")
                            use_emo_text = gr.Checkbox(label="使用文本情感识别", value=False)
                            emo_text = gr.Textbox(label="情感文本", lines=2)
                            use_random = gr.Checkbox(label="随机情感采样", value=False)

                        with gr.Accordion("速度与质量", open=True):
                            with gr.Row():
                                fast_btn = gr.Button("速度优先")
                                quality_btn = gr.Button("质量优先")
                            interval_silence = gr.Slider(0, 1000, value=200, step=10, label="分段静音毫秒")
                            max_text_tokens_per_segment = gr.Slider(20, 240, value=120, step=5, label="单段最大文本 token")
                            num_beams = gr.Slider(1, 10, value=3, step=1, label="搜索宽度")
                            do_sample = gr.Checkbox(label="启用采样", value=True)
                            top_k = gr.Slider(1, 100, value=30, step=1, label="Top-K")
                            top_p = gr.Slider(0, 1, value=0.8, step=0.01, label="Top-P")
                            temperature = gr.Slider(0.1, 2.0, value=0.8, step=0.05, label="温度")
                            max_mel_tokens = gr.Slider(100, 3000, value=1500, step=50, label="最大生成 token")
                            length_penalty = gr.Slider(0.0, 2.0, value=0.0, step=0.05, label="长度惩罚")
                            repetition_penalty = gr.Slider(0.1, 20.0, value=10.0, step=0.1, label="重复惩罚")
                            diffusion_steps = gr.Slider(1, 50, value=25, step=1, label="扩散步数")
                            inference_cfg_rate = gr.Slider(0, 2, value=0.7, step=0.05, label="CFG 强度")

                    with gr.Column(scale=2):
                        synth_btn = gr.Button("调用语音合成接口", variant="primary")
                        output_audio = gr.Audio(label="可播放音频", type="filepath")
                        output_file = gr.File(label="原始输出文件")
                        speech_request = gr.Code(label="请求回显", language="json", lines=18)
                        speech_response = gr.JSON(label="响应回显")

            with gr.Tab("WebSocket 测试"):
                ws_session = gr.State(None)
                with gr.Row():
                    ws_status = gr.Textbox(label="连接状态", value="未连接", interactive=False, scale=2)
                    ws_connect_btn = gr.Button("建立连接", variant="primary", scale=1)
                    ws_disconnect_btn = gr.Button("断开连接", variant="stop", interactive=False, scale=1)
                with gr.Row():
                    with gr.Column(scale=3):
                        ws_text = gr.Textbox(
                            label="合成文本",
                            value="这是一次 WebSocket 会话内的连续合成测试。",
                            lines=4,
                        )
                        ws_voice_dropdown = gr.Dropdown(label="合成音色", choices=[], interactive=True)
                        ws_type = gr.Radio(["tts", "tts_stream"], value="tts", label="消息类型")
                        ws_response_format = gr.Radio(
                            ["mp3", "opus", "aac", "flac", "wav", "pcm"],
                            value="wav",
                            label="输出格式",
                        )
                        ws_emo_audio_prompt = gr.Textbox(label="情感参考音频路径", placeholder="可选，填写服务端可访问路径")

                        with gr.Accordion("情感控制", open=False):
                            ws_emo_alpha = gr.Slider(0, 1, value=1.0, step=0.05, label="情感强度")
                            ws_send_emo_vector = gr.Checkbox(label="发送 8 维情感向量", value=False)
                            with gr.Row():
                                ws_happy = gr.Slider(0, 1, value=0, step=0.05, label="高兴")
                                ws_angry = gr.Slider(0, 1, value=0, step=0.05, label="愤怒")
                                ws_sad = gr.Slider(0, 1, value=0, step=0.05, label="悲伤")
                                ws_afraid = gr.Slider(0, 1, value=0, step=0.05, label="害怕")
                            with gr.Row():
                                ws_disgusted = gr.Slider(0, 1, value=0, step=0.05, label="厌恶")
                                ws_melancholic = gr.Slider(0, 1, value=0, step=0.05, label="忧郁")
                                ws_surprised = gr.Slider(0, 1, value=0, step=0.05, label="惊讶")
                                ws_calm = gr.Slider(0, 1, value=0, step=0.05, label="平静")
                            ws_use_emo_text = gr.Checkbox(label="使用文本情感识别", value=False)
                            ws_emo_text = gr.Textbox(label="情感文本", lines=2)
                            ws_use_random = gr.Checkbox(label="随机情感采样", value=False)

                        with gr.Accordion("速度与质量", open=True):
                            with gr.Row():
                                ws_fast_btn = gr.Button("速度优先")
                                ws_quality_btn = gr.Button("质量优先")
                            ws_interval_silence = gr.Slider(0, 1000, value=200, step=10, label="分段静音毫秒")
                            ws_max_text_tokens_per_segment = gr.Slider(20, 240, value=120, step=5, label="单段最大文本 token")
                            ws_num_beams = gr.Slider(1, 10, value=3, step=1, label="搜索宽度")
                            ws_do_sample = gr.Checkbox(label="启用采样", value=True)
                            ws_top_k = gr.Slider(1, 100, value=30, step=1, label="Top-K")
                            ws_top_p = gr.Slider(0, 1, value=0.8, step=0.01, label="Top-P")
                            ws_temperature = gr.Slider(0.1, 2.0, value=0.8, step=0.05, label="温度")
                            ws_max_mel_tokens = gr.Slider(100, 3000, value=1500, step=50, label="最大生成 token")
                            ws_length_penalty = gr.Slider(0.0, 2.0, value=0.0, step=0.05, label="长度惩罚")
                            ws_repetition_penalty = gr.Slider(0.1, 20.0, value=10.0, step=0.1, label="重复惩罚")
                            ws_diffusion_steps = gr.Slider(1, 50, value=25, step=1, label="扩散步数")
                            ws_inference_cfg_rate = gr.Slider(0, 2, value=0.7, step=0.05, label="CFG 强度")

                    with gr.Column(scale=2):
                        ws_send_btn = gr.Button("发送合成", variant="primary", interactive=False)
                        ws_audio = gr.Audio(label="WebSocket 音频", type="filepath")
                        ws_file = gr.File(label="WebSocket 输出文件")
                        ws_request = gr.Code(label="WebSocket 请求回显", language="json", lines=14)
                        ws_response = gr.JSON(label="WebSocket 响应回显")

        speech_inputs = [
            text,
            http_voice_dropdown,
            response_format,
            emo_audio_prompt,
            emo_alpha,
            send_emo_vector,
            happy,
            angry,
            sad,
            afraid,
            disgusted,
            melancholic,
            surprised,
            calm,
            use_emo_text,
            emo_text,
            use_random,
            interval_silence,
            max_text_tokens_per_segment,
            num_beams,
            do_sample,
            top_k,
            top_p,
            temperature,
            max_mel_tokens,
            length_penalty,
            repetition_penalty,
            diffusion_steps,
            inference_cfg_rate,
        ]

        ws_speech_inputs = [
            ws_text,
            ws_voice_dropdown,
            ws_response_format,
            ws_emo_audio_prompt,
            ws_emo_alpha,
            ws_send_emo_vector,
            ws_happy,
            ws_angry,
            ws_sad,
            ws_afraid,
            ws_disgusted,
            ws_melancholic,
            ws_surprised,
            ws_calm,
            ws_use_emo_text,
            ws_emo_text,
            ws_use_random,
            ws_interval_silence,
            ws_max_text_tokens_per_segment,
            ws_num_beams,
            ws_do_sample,
            ws_top_k,
            ws_top_p,
            ws_temperature,
            ws_max_mel_tokens,
            ws_length_penalty,
            ws_repetition_penalty,
            ws_diffusion_steps,
            ws_inference_cfg_rate,
        ]

        health_btn.click(health_check, inputs=[api_base], outputs=[health_request, health_response])
        voice_outputs = [manage_voice_dropdown, http_voice_dropdown, ws_voice_dropdown, delete_dropdown, voices_response]
        refresh_btn.click(refresh_voices, inputs=[api_base], outputs=voice_outputs)
        upload_btn.click(upload_voice, inputs=[api_base, upload_audio, speaker_name], outputs=voice_outputs)
        delete_btn.click(delete_voice, inputs=[api_base, delete_dropdown], outputs=voice_outputs)
        synth_btn.click(
            synthesize_via_api,
            inputs=[api_base, api_mode, openai_model] + speech_inputs,
            outputs=[output_audio, output_file, speech_request, speech_response],
        )
        ws_connect_btn.click(
            ws_connect,
            inputs=[api_base, ws_session],
            outputs=[ws_session, ws_status, ws_connect_btn, ws_disconnect_btn, ws_send_btn, ws_request, ws_response],
        )
        ws_disconnect_btn.click(
            ws_disconnect,
            inputs=[ws_session],
            outputs=[ws_session, ws_status, ws_connect_btn, ws_disconnect_btn, ws_send_btn, ws_request, ws_response],
        )
        ws_send_btn.click(
            ws_send_tts,
            inputs=[api_base, ws_session, ws_type] + ws_speech_inputs,
            outputs=[ws_audio, ws_file, ws_request, ws_response],
        )
        fast_btn.click(
            apply_fast_preset,
            outputs=[
                num_beams,
                do_sample,
                top_k,
                top_p,
                temperature,
                max_mel_tokens,
                length_penalty,
                repetition_penalty,
                diffusion_steps,
                inference_cfg_rate,
            ],
        )
        quality_btn.click(
            apply_quality_preset,
            outputs=[
                num_beams,
                do_sample,
                top_k,
                top_p,
                temperature,
                max_mel_tokens,
                length_penalty,
                repetition_penalty,
                diffusion_steps,
                inference_cfg_rate,
            ],
        )
        ws_fast_btn.click(
            apply_fast_preset,
            outputs=[
                ws_num_beams,
                ws_do_sample,
                ws_top_k,
                ws_top_p,
                ws_temperature,
                ws_max_mel_tokens,
                ws_length_penalty,
                ws_repetition_penalty,
                ws_diffusion_steps,
                ws_inference_cfg_rate,
            ],
        )
        ws_quality_btn.click(
            apply_quality_preset,
            outputs=[
                ws_num_beams,
                ws_do_sample,
                ws_top_k,
                ws_top_p,
                ws_temperature,
                ws_max_mel_tokens,
                ws_length_penalty,
                ws_repetition_penalty,
                ws_diffusion_steps,
                ws_inference_cfg_rate,
            ],
        )
        demo.load(refresh_voices, inputs=[api_base], outputs=voice_outputs)

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IndexTTS2 中文 API 控制台")
    parser.add_argument("--server_name", default="0.0.0.0")
    parser.add_argument("--server_port", type=int, default=7860)
    parser.add_argument("--api_base", default=DEFAULT_API_BASE)
    args = parser.parse_args()

    os.environ["INDEXTTS_API_BASE"] = args.api_base.rstrip("/")
    demo = create_webui(lambda: None)
    demo.launch(server_name=args.server_name, server_port=args.server_port)
