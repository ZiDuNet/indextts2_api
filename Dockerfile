# IndexTTS2 x86 Dockerfile
# 适用于 x86_64 架构 (CUDA 12.x)

FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_ENDPOINT=https://hf-mirror.com
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    git git-lfs curl python3.11 python3.11-dev python3-pip \
    libsndfile1 ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# 安装 uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# 复制项目文件
COPY . /app

# Git LFS
RUN git lfs install && git lfs pull

# 安装 Python 依赖
RUN uv sync --all-extras

EXPOSE 8002

# 启动 API 服务器
CMD ["uv", "run", "python", "api_server.py", "--host", "0.0.0.0", "--port", "8002", "--fp16"]