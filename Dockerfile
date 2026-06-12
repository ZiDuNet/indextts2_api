# IndexTTS2 x86 Dockerfile

FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_ENDPOINT=https://hf-mirror.com
ENV PATH="/root/.local/bin:$PATH"
ENV UV_CACHE_DIR=/tmp/uv-cache
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
WORKDIR /app

# 阿里云 apt 源
RUN sed -i 's|http://archive.ubuntu.com|https://mirrors.aliyun.com|g' /etc/apt/sources.list && \
    sed -i 's|http://security.ubuntu.com|https://mirrors.aliyun.com|g' /etc/apt/sources.list && \
    apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common curl ca-certificates git tini libsndfile1 ffmpeg && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3.11-venv python3-pip && \
    rm -rf /var/lib/apt/lists/*

# 安装 uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# 复制项目文件
COPY . /app

# 安装 Python 依赖（绝对路径指定 python3.11 避免 fallback 到系统 3.10）
RUN uv sync --locked --python /usr/bin/python3.11 --extra webui --no-dev \
    --default-index "https://mirrors.aliyun.com/pypi/simple" \
    && uv run --locked --extra webui --no-dev python -c "import platform, torch, torchaudio, gradio; print(platform.machine(), torch.__version__, torch.version.cuda, torchaudio.__version__, gradio.__version__)"

EXPOSE 8002

ENTRYPOINT ["tini", "--"]
CMD ["uv", "run", "--locked", "--extra", "webui", "--no-dev", "python", "api_server.py", "--host", "0.0.0.0", "--port", "8002", "--fp16"]
