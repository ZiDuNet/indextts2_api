"""
IndexTTS2 API console WebUI.

The app is mounted by api_server.py and calls the real HTTP/WebSocket API so the
request payload, response headers, timings, and generated files can be inspected.
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
        choices.append((f"{name} | {voice_id} | {filename}", voice_id))
    return choices


def _headers_dict(headers) -> dict:
    return {k: v for k, v in headers.items() if k.lower().startswith("x-indextts") or k.lower() == "content-type"}


def _save_response(content: bytes, response_format: str, prefix: str) -> str:
    ext = {"wav": "wav", "mp3": "mp3", "pcm": "pcm"}.get(response_format, "bin")
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}.{ext}"
    path.write_bytes(content)
    return str(path)


def _coerce_optional_text(value: str | None):
    value = (value or "").strip()
    return value or None


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
        raise gr.Error("Input text is required")
    if not voice:
        raise gr.Error("Please upload or select a voice first")
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
    emo_audio = _coerce_optional_text(emo_audio_prompt)
    if emo_audio:
        payload["emo_audio_prompt"] = emo_audio
    emo_text = _coerce_optional_text(emo_text)
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
        return _pretty({"method": "GET", "url": url}), {
            "status_code": resp.status_code,
            "elapsed": round(elapsed, 3),
            "body": body,
        }
    except Exception as exc:
        return _pretty({"method": "GET", "url": url}), {"error": str(exc)}


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
            {"status_code": resp.status_code, "body": body},
        )
    except Exception as exc:
        return gr.update(), gr.update(), {"error": str(exc)}


def upload_voice(api_base, audio_file, speaker_name):
    if not audio_file:
        return gr.update(), gr.update(), {"error": "请先选择音色参考音频"}
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
    return voice_update, delete_update, {
        "status_code": resp.status_code,
        "elapsed": round(elapsed, 3),
        "body": body,
        "voices": voices_result,
    }


def delete_voice(api_base, voice_id):
    if not voice_id:
        return gr.update(), gr.update(), {"error": "请选择要删除的 voice_id"}
    url = f"{_api_base(api_base)}/v1/audio/voices/{voice_id}"
    resp = requests.delete(url, timeout=60)
    body = resp.json() if resp.content else {}
    voice_update, delete_update, voices_result = refresh_voices(api_base)
    return voice_update, delete_update, {
        "status_code": resp.status_code,
        "body": body,
        "voices": voices_result,
    }


def synthesize_via_api(api_base, *values):
    payload = build_speech_payload(*values)
    url = f"{_api_base(api_base)}/v1/audio/speech"
    request_echo = {"method": "POST", "url": url, "json": payload}
    started = time.perf_counter()
    try:
        resp = requests.post(url, json=payload, timeout=900)
        elapsed = time.perf_counter() - started
        content_type = resp.headers.get("content-type", "")
        if resp.status_code >= 400 or content_type.startswith("application/json"):
            body = resp.json() if resp.content else {}
            return None, None, _pretty(request_echo), {
                "status_code": resp.status_code,
                "elapsed": round(elapsed, 3),
                "headers": _headers_dict(resp.headers),
                "body": body,
            }
        output_path = _save_response(resp.content, payload["response_format"], "api_tts")
        playable_audio = output_path if payload["response_format"] in ("wav", "mp3") else None
        return playable_audio, output_path, _pretty(request_echo), {
            "status_code": resp.status_code,
            "elapsed": round(elapsed, 3),
            "headers": _headers_dict(resp.headers),
            "output_file": output_path,
            "bytes": len(resp.content),
        }
    except Exception as exc:
        return None, None, _pretty(request_echo), {"error": str(exc)}


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
    request_echo = {"url": f"{_api_base(api_base)}/ws", "json": payload}
    started = time.perf_counter()
    try:
        ws_url, connected, message = asyncio.run(_websocket_call(api_base, payload))
        elapsed = time.perf_counter() - started
        audio_path = None
        if message.get("audio_base64"):
            audio_path = _save_response(base64.b64decode(message["audio_base64"]), "wav", "ws_tts")
            message = {k: v for k, v in message.items() if k != "audio_base64"}
            message["audio_file"] = audio_path
        request_echo["url"] = ws_url
        return audio_path, audio_path, _pretty(request_echo), {
            "elapsed": round(elapsed, 3),
            "connected": connected,
            "message": message,
        }
    except Exception as exc:
        return None, None, _pretty(request_echo), {"error": str(exc)}


def apply_fast_preset():
    return 1, False, 10, 0.8, 0.8, 900, 0.0, 10.0, 16, 0.7


def apply_quality_preset():
    return 3, True, 30, 0.8, 0.8, 1500, 0.0, 10.0, 25, 0.7


def create_webui(get_tts):
    css = """
    .api-console textarea, .api-console pre { font-family: ui-monospace, SFMono-Regular, Consolas, monospace; }
    .api-console .gradio-container { max-width: 1440px; }
    """

    with gr.Blocks(title="IndexTTS2 API Console", css=css, fill_height=True, elem_classes=["api-console"]) as demo:
        gr.Markdown("# IndexTTS2 API Console")

        api_base = gr.Textbox(label="API base", value=DEFAULT_API_BASE)

        with gr.Row():
            health_btn = gr.Button("Health")
            refresh_btn = gr.Button("Refresh voices")
        health_request = gr.Code(label="Health request", language="json", lines=4)
        health_response = gr.JSON(label="Health response")

        with gr.Tabs():
            with gr.Tab("Voices"):
                with gr.Row():
                    with gr.Column(scale=2):
                        upload_audio = gr.Audio(label="Reference audio", type="filepath", sources=["upload", "microphone"])
                        speaker_name = gr.Textbox(label="Speaker name", placeholder="intro-01-tts")
                        upload_btn = gr.Button("Upload voice", variant="primary")
                    with gr.Column(scale=2):
                        voice_dropdown = gr.Dropdown(label="Registered voices", choices=[], interactive=True)
                        delete_dropdown = gr.Dropdown(label="Delete voice", choices=[], interactive=True)
                        delete_btn = gr.Button("Delete selected voice", variant="stop")
                        voices_response = gr.JSON(label="Voice API response")

            with gr.Tab("Speech"):
                with gr.Row():
                    with gr.Column(scale=3):
                        text = gr.Textbox(
                            label="Input text",
                            value="这是一次 IndexTTS 二代接口测试。",
                            lines=4,
                        )
                        response_format = gr.Radio(["wav", "mp3", "pcm"], value="wav", label="Response format")
                        emo_audio_prompt = gr.Textbox(label="Emotion audio path", placeholder="optional server-side path")

                        with gr.Accordion("Emotion", open=False):
                            emo_alpha = gr.Slider(0, 1, value=1.0, step=0.05, label="emo_alpha")
                            send_emo_vector = gr.Checkbox(label="Send emo_vector", value=False)
                            with gr.Row():
                                happy = gr.Slider(0, 1, value=0, step=0.05, label="happy")
                                angry = gr.Slider(0, 1, value=0, step=0.05, label="angry")
                                sad = gr.Slider(0, 1, value=0, step=0.05, label="sad")
                                afraid = gr.Slider(0, 1, value=0, step=0.05, label="afraid")
                            with gr.Row():
                                disgusted = gr.Slider(0, 1, value=0, step=0.05, label="disgusted")
                                melancholic = gr.Slider(0, 1, value=0, step=0.05, label="melancholic")
                                surprised = gr.Slider(0, 1, value=0, step=0.05, label="surprised")
                                calm = gr.Slider(0, 1, value=0, step=0.05, label="calm")
                            use_emo_text = gr.Checkbox(label="use_emo_text", value=False)
                            emo_text = gr.Textbox(label="emo_text", lines=2)
                            use_random = gr.Checkbox(label="use_random", value=False)

                        with gr.Accordion("Generation", open=True):
                            with gr.Row():
                                fast_btn = gr.Button("Fast preset")
                                quality_btn = gr.Button("Quality preset")
                            interval_silence = gr.Slider(0, 1000, value=200, step=10, label="interval_silence")
                            max_text_tokens_per_segment = gr.Slider(20, 240, value=120, step=5, label="max_text_tokens_per_segment")
                            num_beams = gr.Slider(1, 10, value=3, step=1, label="num_beams")
                            do_sample = gr.Checkbox(label="do_sample", value=True)
                            top_k = gr.Slider(1, 100, value=30, step=1, label="top_k")
                            top_p = gr.Slider(0, 1, value=0.8, step=0.01, label="top_p")
                            temperature = gr.Slider(0.1, 2.0, value=0.8, step=0.05, label="temperature")
                            max_mel_tokens = gr.Slider(100, 3000, value=1500, step=50, label="max_mel_tokens")
                            length_penalty = gr.Slider(0.0, 2.0, value=0.0, step=0.05, label="length_penalty")
                            repetition_penalty = gr.Slider(0.1, 20.0, value=10.0, step=0.1, label="repetition_penalty")
                            diffusion_steps = gr.Slider(1, 50, value=25, step=1, label="diffusion_steps")
                            inference_cfg_rate = gr.Slider(0, 2, value=0.7, step=0.05, label="inference_cfg_rate")

                    with gr.Column(scale=2):
                        synth_btn = gr.Button("POST /v1/audio/speech", variant="primary")
                        output_audio = gr.Audio(label="Playable output", type="filepath")
                        output_file = gr.File(label="Raw output file")
                        speech_request = gr.Code(label="Request echo", language="json", lines=18)
                        speech_response = gr.JSON(label="Response echo")

            with gr.Tab("WebSocket"):
                ws_type = gr.Radio(["tts", "tts_stream"], value="tts", label="Message type")
                ws_btn = gr.Button("Send WebSocket message", variant="primary")
                ws_audio = gr.Audio(label="WebSocket audio", type="filepath")
                ws_file = gr.File(label="WebSocket output file")
                ws_request = gr.Code(label="WebSocket request echo", language="json", lines=14)
                ws_response = gr.JSON(label="WebSocket response echo")

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
    parser = argparse.ArgumentParser(description="IndexTTS2 API Console")
    parser.add_argument("--server_name", default="0.0.0.0")
    parser.add_argument("--server_port", type=int, default=7860)
    parser.add_argument("--api_base", default=DEFAULT_API_BASE)
    args = parser.parse_args()

    os.environ["INDEXTTS_API_BASE"] = args.api_base.rstrip("/")
    demo = create_webui(lambda: None)
    demo.launch(server_name=args.server_name, server_port=args.server_port)
