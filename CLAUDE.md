# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

IndexTTS2 — Bilibili 开源的零样本（zero-shot）TTS 系统，支持声音克隆和情感控制。本项目包含官方推理引擎（`indextts/`）、FastAPI API 服务器（`api_server.py`）、官方 WebUI（`webui.py`）和增强版 WebUI（`webui_enhanced.py`）。

## Environment & Dependencies

- **Python**: 3.10–3.14（`.python-version` 指定 3.10）
- **包管理**: 必须使用 `uv`（不支持 conda/pip 直接安装）
- **GPU**: 需要 NVIDIA CUDA 12.8+
- **模型文件**: 下载到 `checkpoints/` 目录，来源为 HuggingFace `IndexTeam/IndexTTS-2` 或 ModelScope

```bash
uv sync --all-extras
# 中国镜像加速
uv sync --all-extras --default-index "https://mirrors.aliyun.com/pypi/simple"
```

## Common Commands

```bash
# API 服务器（端口 8002），根路由挂载增强版 WebUI
uv run python api_server.py --host 0.0.0.0 --port 8002 --fp16

# 官方 WebUI（端口 7860）
uv run webui.py --fp16

# 增强版 WebUI 独立运行（含情感控制 + 推理参数调优）
uv run python webui_enhanced.py --use_fp16

# 下载模型
uv run python download_models.py

# GPU 环境检查
uv run tools/gpu_check.py
```

## Architecture

### 核心推理引擎 (`indextts/`)

- `infer_v2.py` → `IndexTTS2`：v2 模型，支持情感控制和拼音标注。`infer()` 接受 `**generation_kwargs` 透传推理参数
- `infer.py` → `IndexTTS`：v1/v1.5 模型，仅声音克隆

推理流程：text → 前端处理 → GPT token 生成 → s2mel → BigVGAN → wav

### API 服务器 (`api_server.py`)

FastAPI 应用，Swagger 文档在 `/docs`。根路由 `/` 挂载增强版 WebUI（Gradio）。

**并发控制**：GPU 推理使用 `asyncio.Semaphore(1)` 串行化，输出文件使用 UUID 命名防冲突，音色元数据使用内存缓存避免磁盘读取。

**核心接口（/v1 规范）**：
- `POST /v1/audio/speech` — 语音合成（`input`/`voice`/`response_format`，支持全部推理参数）
- `POST /v1/audio/voices` — 上传音色，返回 `voice_id`，自动预热
- `GET /v1/audio/voices` — 列出所有已注册音色
- `DELETE /v1/audio/voices/{id}` — 删除音色
- `WS /ws` — WebSocket 流式合成
- `GET /health` — 健康检查

**音色缓存**：`assets/speaker_cache/`，元数据 `meta.json`，音频文件 `spk_{md5[:8]}.wav`

**推理可调参数**（所有 TTS 接口均支持）：`num_beams`(3)、`do_sample`(True)、`top_k`(30)、`top_p`(0.8)、`temperature`(0.8)、`max_mel_tokens`(1500)、`length_penalty`(0.0)、`repetition_penalty`(10.0)

### 增强版 WebUI (`webui_enhanced.py`)

导出 `create_webui(get_tts)` 函数，接受 getter 回调延迟获取模型实例。支持完整情感控制（8 维向量 + 文本情感 + 情感音频）和 8 个推理可调参数。可独立运行或被 `api_server.py` 挂载。

## Key Parameters

- `--fp16`: 半精度推理，推荐启用
- `--deepspeed`: DeepSpeed 加速（效果取决于硬件）
- `emo_alpha` (0.0–1.0): 情感控制强度，文本情感模式建议 ≤0.6
- `emo_vector`: 8 维情感向量 `[happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]`

## Important Notes

- 运行脚本必须通过 `uv run`，不要手动激活虚拟环境
- 拼音控制仅在有效中文拼音范围内生效（参考 `checkpoints/pinyin.vocab`）
- Windows 上 DeepSpeed 可能难以安装，可跳过 `--all-extras`
- HuggingFace 访问慢时设置 `HF_ENDPOINT=https://hf-mirror.com`
- 辅助模型下载已默认走 ModelScope（`network_detection.py`）
- `create_webui(get_tts)` 使用 getter 模式而非直接传 tts 实例，因为 Gradio 在模块加载时初始化而模型在 FastAPI lifespan 中初始化
