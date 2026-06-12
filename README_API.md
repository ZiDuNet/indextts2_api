# IndexTTS2 API 服务器

基于官方 index-tts 代码 + 自定义接口

## 接口列表

| 接口 | 方法 | 功能 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/register_speaker` | POST | 注册角色 |
| `/audio/voices` | GET | 获取已注册角色 |
| `/update_speaker` | POST | 修改角色 |
| `/delete_speaker` | POST | 删除角色 |
| `/upload_audio` | POST | 文件上传 |
| `/tts` | POST | 语音合成(二进制) |
| `/tts-wav` | POST | 语音合成(WAV) |
| `/tts_url` | POST | 使用音频路径合成 |
| `/audio/speech` | POST | OpenAI兼容接口 |
| `/ws` | WebSocket | 流式合成 |

## 快速开始

### 1. Python 直接启动

```bash
# 安装依赖 (需要 Python 3.10-3.11)
cd C:\Users\wushuo\Desktop\indextts\index-tts-official

# 安装 uv (如果未安装)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 使用 uv 安装依赖
uv sync --all-extras

# 启动服务器
python api_server.py --host 0.0.0.0 --port 8002 --fp16
```

### 2. Docker 启动

```bash
# x86 版本
docker build -t indextts2 .
docker run --gpus all -p 8002:8002 indextts2

# ARM64/GB10 版本
docker build -f Dockerfile.arm64 -t indextts2:arm64 .
docker-compose -f docker-compose.arm64.yml up
```

## 测试

启动服务器后，在另一个终端运行：

```bash
python test_api.py
```

## 请求示例

### 语音合成

```bash
curl -X POST http://localhost:8002/tts-wav \
  -H "Content-Type: application/json" \
  -d '{
    "text": "你好，这是测试",
    "spk_audio_prompt": "examples/voice_01.wav"
  }' \
  -o output.wav
```

### 注册角色

```bash
curl -X POST http://localhost:8002/register_speaker \
  -H "Content-Type: application/json" \
  -d '{
    "name": "我的角色",
    "sample_audios": ["examples/voice_01.wav"]
  }'
```

### 获取角色列表

```bash
curl http://localhost:8002/audio/voices
```

## 注意事项

1. 需要下载模型文件到 `checkpoints/` 目录
2. 需要准备参考音频文件到 `examples/` 目录
3. Python 版本需要 3.10-3.11（官方依赖限制）