FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

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

# Install PyTorch with CUDA 11.8 first (before requirements.txt)
RUN pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# Set CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda-11.8
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6"
ENV FORCE_CUDA=1

# Set working directory
WORKDIR /workspace

# Copy requirements.txt
COPY requirements.txt /workspace/

# Install all requirements including git packages
RUN pip install --no-cache-dir --use-pep517 --no-build-isolation -r requirements.txt

# Verify installations
RUN python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}, Available: {torch.cuda.is_available()}')" && \
    python -c "import tinycudann; print('tinycudann installed successfully')" && \
    python -c "import pytorch3d; print('pytorch3d installed successfully')" && \
    python -c "import permutohedral_encoding; print('permutohedral_encoding installed successfully')"

# Set up non-root user (optional but recommended)
RUN useradd -m -s /bin/bash developer && \
    chown -R developer:developer /workspace

USER developer

# Label the image
LABEL maintainer="HF3DNeuralReco"
LABEL description="CUDA 11.8 environment for HF3DNeuralReco project with PyTorch and neural rendering libraries"

# Default command
CMD ["/bin/bash"]
