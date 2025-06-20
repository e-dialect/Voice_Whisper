# 基于CUDA的Python 3.10镜像
FROM nvidia/cuda:11.6.2-cudnn8-runtime-ubuntu20.04

# 避免交互式提示
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai

# 系统依赖安装
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y \
    python3.10 \
    python3.10-distutils \
    python3.10-dev \
    python3-pip \
    build-essential \
    git \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3

# 安装pip
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

# 创建必要的目录结构
RUN mkdir -p /app/uploads /app/outputs /app/model_cache /app/tools/uvr5

# 设置工作目录
WORKDIR /app

# 复制和安装依赖
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY SenseVoice/app.py /app/
COPY SenseVoice/templates/ /app/templates/
COPY SenseVoice/static/ /app/static/

# 复制第三方库Matcha-TTS（如有需要可在此处添加）
# Matcha-TTS 已集成于 SenseVoice/CosyVoice/third_party/Matcha-TTS
# 容器内路径为 /app/SenseVoice/CosyVoice/third_party/Matcha-TTS

# 创建模型目录路径
RUN mkdir -p /app/model_cache/models/iic/

# 设置环境变量
ENV PYTHONPATH=/app:/app/CosyVoice:/app/tools:/app/tools/uvr5
ENV MODEL_DIR=/app/model_cache

# 暴露应用端口
EXPOSE 5000

# 启动命令
CMD ["python", "app.py"]