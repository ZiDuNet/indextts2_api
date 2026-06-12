# IndexTTS2 快速开始指南

## 项目位置
```
C:\Users\wushuo\Desktop\indextts\index-tts-vllm\
```

## 当前状态

### ✅ 已完成
- API 服务器代码 (简化版可测试)
- x86 和 ARM64/GB10 Docker 配置
- 你的音频文件 (examples/voice_01.wav)

### ❌ 需要你完成
- Python 3.11 安装
- 模型下载

---

## 快速开始

### 方式 1: Python 直接启动 (推荐)

```bash
# 1. 安装 Python 3.11
# 下载: https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe

# 2. 进入项目目录
cd C:\Users\wushuo\Desktop\indextts\index-tts-vllm

# 3. 创建虚拟环境
python3.11 -m venv venv

# 4. 激活并安装依赖
venv\Scripts\activate
pip install uv
uv sync --all-extras

# 5. 启动服务器 (会自动从 ModelScope 下载模型)
python api_server.py --fp16

# 6. 测试接口
# 打开浏览器: http://localhost:8002/health
```

### 方式 2: Docker

```bash
# 1. 安装 Docker Desktop
# https://www.docker.com/products/docker-desktop/

# 2. 构建镜像
cd C:\Users\wushuo\Desktop\indextts\index-tts-vllm
docker build -t indextts2 .

# 3. 运行
docker run --gpus all -p 8002:8002 indextts2
```

### 方式 3: 简化版 (立即可用，但只用模拟音频)

```bash
cd C:\Users\wushuo\Desktop\indextts\index-tts-vllm
python api_server_simple.py
```

---

## 接口列表

| 接口 | 方法 | 功能 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/register_speaker` | POST | 注册角色 |
| `/audio/voices` | GET | 获取角色列表 |
| `/update_speaker` | POST | 修改角色 |
| `/delete_speaker` | POST | 删除角色 |
| `/upload_audio` | POST | 文件上传 |
| `/tts` | POST | 语音合成(二进制) |
| `/tts-wav` | POST | 语音合成(WAV) |
| `/tts_url` | POST | URL合成 |
| `/audio/speech` | POST | OpenAI兼容 |
| `/ws` | WebSocket | 流式合成 |

---

## 测试命令

```bash
# 测试健康
curl http://localhost:8002/health

# 测试 TTS
curl -X POST http://localhost:8002/tts-wav \
  -H "Content-Type: application/json" \
  -d '{"text": "你好", "spk_audio_prompt": "examples/voice_01.wav"}' \
  -o test.wav
```

---

## 问题排查

1. **Python 版本错误**: 确保使用 Python 3.11
2. **模型下载慢**: 首次运行会自动从 ModelScope 下载 (~2GB)
3. **CUDA 错误**: 确保有 NVIDIA GPU 和驱动

---

## 文件说明

| 文件 | 说明 |
|------|------|
| `api_server.py` | 完整版 (需要 Python 3.11 + 模型) |
| `api_server_simple.py` | 简化版 (模拟音频) |
| `Dockerfile` | x86 版本 |
| `Dockerfile.arm64` | ARM64/GB10 版本 |
| `docker-compose.yml` | x86 部署 |
| `docker-compose.arm64.yml` | ARM64 部署 |
| `examples/voice_01.wav` | 你的音频 |