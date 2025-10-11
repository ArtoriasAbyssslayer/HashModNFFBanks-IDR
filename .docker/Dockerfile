FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# CUDA environment setup
ENV CUDA_HOME=/usr/local/cuda-11.8
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Target multiple GPU architectures for compatibility
ENV TCNN_CUDA_ARCHITECTURES="61;75;86;89"
ENV PENC_CUDA_ARCHITECTURES="61;75;86;89"
ENV PYTORCH3D_CUDA_ARCHITECTURES="61;75;86;89"
ENV TORCH_CUDA_ARCH_LIST="6.1;7.5;8.6;8.9"
ENV FORCE_CUDA=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    ninja-build \
    wget \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 11.8 first
RUN pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# Set working directory
WORKDIR /workspace

# Copy requirements.txt
COPY requirements.txt /workspace/

# Install all requirements including git packages
RUN pip install --no-cache-dir --use-pep517 --no-build-isolation -r requirements.txt

# Copy the entire project into the container
COPY . /workspace/

# Set up non-root user
RUN useradd -m -s /bin/bash developer && \
    chown -R developer:developer /workspace

USER developer

# Label the image
LABEL maintainer="HF3DNeuralReco"
LABEL description="CUDA 11.8 environment for HF3DNeuralReco project with PyTorch and neural rendering libraries"

# Default command
CMD ["/bin/bash"]
