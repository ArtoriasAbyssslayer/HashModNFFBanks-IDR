# Docker Setup Guide for HF3DNeuralReco

## Prerequisites

1. **NVIDIA Docker Runtime** installed on your system:
   ```bash
   # Install nvidia-docker2 on Arch Linux
   paru -S nvidia-docker
   
   # Or install nvidia-container-toolkit
   sudo pacman -S nvidia-container-toolkit
   
   # Restart Docker
   sudo systemctl restart docker
   ```

2. **Verify GPU access in Docker**:
   ```bash
   docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
   ```

## Quick Start

### Option 1: Using Docker Compose (Recommended)

```bash
# Build the image
docker-compose build

# Run container
docker-compose run --rm cuda-dev

# Or run in background and attach
docker-compose up -d
docker-compose exec cuda-dev bash
```

### Option 2: Using Docker Directly

```bash
# Build the image
docker build -t hf3dneuralreco:latest .

# Run container
docker run --gpus all -it --rm \
    -v $(pwd):/workspace \
    -v ~/.cache:/home/developer/.cache \
    --shm-size 8g \
    hf3dneuralreco:latest
```

## Verify Installation Inside Container

```bash
# Check PyTorch and CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

# Check GPU
nvidia-smi

# Check installed packages
python -c "import tinycudann; print('tinycudann OK')"
python -c "import pytorch3d; print('pytorch3d OK')"
python -c "import permutohedral_encoding; print('permutohedral_encoding OK')"
```

## Development Workflow

### Running Your Scripts

```bash
# Inside container
python your_script.py

# Or from host
docker-compose exec cuda-dev python your_script.py
```

### Installing Additional Packages

```bash
# Inside container
pip install package-name

# To make permanent, add to requirements.txt and rebuild
docker-compose build
```

### Using Jupyter Notebook (Optional)

Add to docker-compose.yml:
```yaml
    ports:
      - "8888:8888"
    command: jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

Then install jupyter:
```bash
pip install jupyter
```

## Troubleshooting

### Issue: "could not select device driver"
```bash
# Check nvidia-docker is installed
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# If fails, reinstall nvidia-docker2
paru -S nvidia-docker
sudo systemctl restart docker
```

### Issue: "Out of memory" errors
```bash
# Increase shared memory in docker-compose.yml
shm_size: '16gb'  # or higher
```

### Issue: Compilation fails for CUDA packages
```bash
# Check CUDA is accessible
echo $CUDA_HOME
nvcc --version

# Rebuild with no cache
docker-compose build --no-cache
```

### Issue: Slow build times
```bash
# Use BuildKit for faster builds
DOCKER_BUILDKIT=1 docker-compose build

# Or enable BuildKit permanently
echo 'export DOCKER_BUILDKIT=1' >> ~/.bashrc
```

## Build Time Estimates

- **Initial build**: 15-25 minutes (downloads + compilation)
- **Rebuild after changes**: 2-5 minutes (uses cache)
- **Pull base image**: 2-3 GB download

## Performance Notes

- Docker adds ~5-10% overhead compared to native
- GPU performance is nearly identical to native
- I/O can be slower with mounted volumes (use Docker volumes for better performance if needed)

## Alternative: Build Without Docker

If you prefer native installation on Arch, use Conda:

```bash
# Install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
~/miniconda3/bin/conda init bash
source ~/.bashrc

# Create environment with GCC 11
conda create -n hf3dneuralreco python=3.10 -y
conda activate hf3dneuralreco
conda install -c conda-forge gxx_linux-64=11.2.0 -y

# Set environment variables
export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
export CUDAHOSTCXX=$CXX
export CUDA_HOME=/opt/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Install PyTorch
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# Install from requirements.txt
pip install -r requirements.txt
```
