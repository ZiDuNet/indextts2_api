# IndexTTS2 快速开始

## 环境要求

- Python 3.10–3.14
- NVIDIA GPU + CUDA 12.8+
- [uv 包管理器](https://docs.astral.sh/uv/)

## 安装

```bash
# 安装依赖
uv sync --all-extras

# 中国镜像加速
uv sync --all-extras --default-index "https://mirrors.aliyun.com/pypi/simple"
```

## 启动

```bash
# API 服务器（端口 8002），首次启动自动下载模型
uv run python api_server.py --host 0.0.0.0 --port 8002 --fp16
```

启动后：
- **WebUI**：http://localhost:8002/
- **API 文档**：http://localhost:8002/docs

## 测试

```bash
# 健康检查
curl http://localhost:8002/health

# 上传音色
curl -X POST http://localhost:8002/v1/audio/voices \
  -F "audio=@examples/voice_01.wav" \
  -F "speaker_name=测试"

# 语音合成
curl -X POST http://localhost:8002/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input":"你好世界","voice":"spk_xxxxxxxx"}' \
  -o test.wav
```

详细 API 文档见 [README_API.md](README_API.md)。

## 其他启动方式

```bash
# 官方 WebUI（端口 7860）
uv run webui.py --fp16

# 增强版 WebUI 独立运行（含情感控制 + 推理参数调优）
uv run python webui_enhanced.py --use_fp16
```

## 常见问题

- **模型下载慢**：已默认走 ModelScope 镜像，首次下载约 2GB
- **Windows DeepSpeed 安装失败**：可跳过，去掉 `--all-extras` 改用 `--extra webui`
- **GPU 检查**：`uv run tools/gpu_check.py`
