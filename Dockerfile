# IndexTTS2 x86 Dockerfile
# 最小化基础镜像 + CUDA runtime 按需安装

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_ENDPOINT=https://hf-mirror.com
ENV PATH="/root/.local/bin:$PATH"
# 让 PyTorch 在 x86_64 + CUDA 12 容器里能找到 .so
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}

WORKDIR /app

# 阿里云 apt 源 + NVIDIA CUDA 12.8 仓库
RUN sed -i 's|http://archive.ubuntu.com|https://mirrors.aliyun.com|g' /etc/apt/sources.list && \
    sed -i 's|http://security.ubuntu.com|https://mirrors.aliyun.com|g' /etc/apt/sources.list && \
    apt-get update && apt-get install -y wget gnupg2 && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && rm cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        cuda-cudart-12-8 cuda-libraries-12-8 libcudnn8 libcublas-12-8 && \
    rm -rf /var/lib/apt/lists/*

# 系统依赖 + Python 3.11
RUN apt-get update && apt-get install -y \
    software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y \
    curl python3.11 python3.11-dev python3.11-venv python3-pip \
    libsndfile1 ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# 安装 uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# 让 uv 显式知道要 3.11，避免 fallback 到系统 Python 3.10
RUN uv python install 3.11

# 复制项目文件
COPY . /app

# 安装 Python 依赖（指定 Python 3.11；只装 webui extra 避开 deepspeed，
# deepspeed 无 ARM64/x86_12.8 预编译 wheel，从源码编译需要 nvcc + 5+ GB 镜像）
RUN uv sync --python 3.11 --extra webui --default-index "https://mirrors.aliyun.com/pypi/simple"

EXPOSE 8002

CMD ["uv", "run", "python", "api_server.py", "--host", "0.0.0.0", "--port", "8002", "--fp16"]
