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
import time
from pathlib import Path

import gradio as gr
import requests


DEFAULT_API_BASE = os.environ.get("INDEXTTS_API_BASE", "http://127.0.0.1:8002").rstrip("/")


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
    ext = {"wav": "wav", "mp3": "mp3", "pcm": "pcm"}.get(response_format, "bin")
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}.{ext}"
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
        return (
            gr.update(choices=choices, value=value),
            gr.update(choices=choices, value=value),
            _response_echo(resp.status_code, 0, resp.headers, body=body),
        )
    except Exception as exc:
        return gr.update(), gr.update(), {"错误": str(exc)}


def upload_voice(api_base, audio_file, speaker_name):
    if not audio_file:
        return gr.update(), gr.update(), {"错误": "请先选择音色参考音频"}

    url = f"{_api_base(api_base)}/v1/audio/voices"
    started = time.perf_counter()
    with open(audio_file, "rb") as file_obj:
        files = {"audio": (Path(audio_file).name, file_obj)}
        data = {"speaker_name": speaker_name or ""}
        resp = requests.post(url, files=files, data=data, timeout=600)
    elapsed = time.perf_counter() - started
    body = resp.json() if resp.content else {}

    voice_update, delete_update, voices_result = refresh_voices(api_base)
    voice_id = body.get("voice_id")
    if voice_id:
        voice_update["value"] = voice_id
        delete_update["value"] = voice_id

    result = _response_echo(resp.status_code, elapsed, resp.headers, body=body)
    result["音色列表刷新"] = voices_result
    return voice_update, delete_update, result


def delete_voice(api_base, voice_id):
    if not voice_id:
        return gr.update(), gr.update(), {"错误": "请选择要删除的音色"}

    url = f"{_api_base(api_base)}/v1/audio/voices/{voice_id}"
    started = time.perf_counter()
    resp = requests.delete(url, timeout=60)
    elapsed = time.perf_counter() - started
    body = resp.json() if resp.content else {}

    voice_update, delete_update, voices_result = refresh_voices(api_base)
    result = _response_echo(resp.status_code, elapsed, resp.headers, body=body)
    result["音色列表刷新"] = voices_result
    return voice_update, delete_update, result


def synthesize_via_api(api_base, *values):
    payload = build_speech_payload(*values)
    url = f"{_api_base(api_base)}/v1/audio/speech"
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


async def _websocket_call(api_base: str, payload: dict):
    import websockets

    ws_base = _api_base(api_base).replace("http://", "ws://").replace("https://", "wss://")
    url = f"{ws_base}/ws"
    async with websockets.connect(url, open_timeout=30) as websocket:
        connected = json.loads(await websocket.recv())
        await websocket.send(json.dumps(payload, ensure_ascii=False))
        while True:
            message = json.loads(await websocket.recv())
            if message.get("type") in {"completed", "stream_completed", "error"}:
                return url, connected, message


def test_websocket(api_base, message_type, *values):
    payload = build_speech_payload(*values)
    payload["type"] = message_type
    payload["text"] = payload.pop("input")
    started = time.perf_counter()
    request_echo = _request_echo("WS", f"{_api_base(api_base)}/ws", payload)

    try:
        ws_url, connected, message = asyncio.run(_websocket_call(api_base, payload))
        elapsed = time.perf_counter() - started
        request_echo = _request_echo("WS", ws_url, payload)

        audio_path = None
        if message.get("audio_base64"):
            audio_path = _save_response(base64.b64decode(message["audio_base64"]), "wav", "ws_tts")
            message = {key: value for key, value in message.items() if key != "audio_base64"}
            message["audio_file"] = audio_path

        return audio_path, audio_path, request_echo, {
            "耗时秒": round(elapsed, 3),
            "连接回显": connected,
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
                        voice_dropdown = gr.Dropdown(label="已注册音色", choices=[], interactive=True)
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
                        response_format = gr.Radio(["wav", "mp3", "pcm"], value="wav", label="输出格式")
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
                ws_type = gr.Radio(["tts", "tts_stream"], value="tts", label="消息类型")
                ws_btn = gr.Button("发送 WebSocket 消息", variant="primary")
                ws_audio = gr.Audio(label="WebSocket 音频", type="filepath")
                ws_file = gr.File(label="WebSocket 输出文件")
                ws_request = gr.Code(label="WebSocket 请求回显", language="json", lines=14)
                ws_response = gr.JSON(label="WebSocket 响应回显")

        speech_inputs = [
            text,
            voice_dropdown,
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

        health_btn.click(health_check, inputs=[api_base], outputs=[health_request, health_response])
        refresh_btn.click(refresh_voices, inputs=[api_base], outputs=[voice_dropdown, delete_dropdown, voices_response])
        upload_btn.click(upload_voice, inputs=[api_base, upload_audio, speaker_name], outputs=[voice_dropdown, delete_dropdown, voices_response])
        delete_btn.click(delete_voice, inputs=[api_base, delete_dropdown], outputs=[voice_dropdown, delete_dropdown, voices_response])
        synth_btn.click(
            synthesize_via_api,
            inputs=[api_base] + speech_inputs,
            outputs=[output_audio, output_file, speech_request, speech_response],
        )
        ws_btn.click(
            test_websocket,
            inputs=[api_base, ws_type] + speech_inputs,
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
        demo.load(refresh_voices, inputs=[api_base], outputs=[voice_dropdown, delete_dropdown, voices_response])

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
