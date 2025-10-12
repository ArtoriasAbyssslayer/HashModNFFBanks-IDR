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

## Quick Start - Using Pre-built Image (Recommended)

### Pull from GitHub Container Registry

```bash
# Pull the latest version
docker pull ghcr.io/artoriasabyssslayer/hashmod-nffbanks-idr:latest

# Or pull a specific version
docker pull ghcr.io/artoriasabyssslayer/hashmod-nffbanks-idr:v1.0.0
```

### Run the Container

**Basic usage (interactive shell):**
```bash
docker run -it --rm \
  --gpus all \
  -v $(pwd):/workspace \
  ghcr.io/artoriasabyssslayer/hashmod-nffbanks-idr:latest
```

**Full setup with all mounts:**
```bash
docker run -it --rm \
  --gpus all \
  --name hf3d-container \
  -v $(pwd):/workspace \
  -v $(pwd)/data:/data \
  -v $(pwd)/experiments:/experiments \
  -v $(pwd)/checkpoints:/checkpoints \
  --shm-size=8g \
  ghcr.io/artoriasabyssslayer/hashmod-nffbanks-idr:latest
```

**Run training directly:**
```bash
docker run -it --rm \
  --gpus all \
  -v $(pwd):/workspace \
  --shm-size=8g \
  ghcr.io/artoriasabyssslayer/hashmod-nffbanks-idr:latest \
  ./code/scripts/run_training_failsafe.sh --exp HashGrid --scan_id 114
```

**Run evaluation:**
```bashs
docker run -it --rm \
  --gpus all \
  -v $(pwd):/workspace \
  ghcr.io/artoriasabyssslayer/hashmod-nffbanks-idr:latest \
  ./code/scripts/run_evaluation_failsafe.sh --exp HashGrid --scan_id 114
```

## Building from Source

### Option 1: Using Build Script

```bash
# Make script executable
chmod +x build-docker.sh

# Build the image
./build-docker.sh

# Select build option:
# 1. Quick build (use cache) - Recommended
# 2. Clean build (no cache)
# 3. Build and run immediately
# 4. Just run (skip build)
```

### Option 2: Using Docker Compose

```bash
# Build the image
docker-compose build

# Run container
docker-compose run --rm hf3d-neural-reco

# Or run in background and attach
docker-compose up -d
docker-compose exec hf3d-neural-reco bash
```

### Option 3: Using Docker Directly

```bash
# Build the image
docker build -t hf3d-neural-reco:latest .

# Run container
docker run -it --rm \
  --gpus all \
  -v $(pwd):/workspace \
  -v $(pwd)/data:/data \
  -v $(pwd)/experiments:/experiments \
  --shm-size=8g \
  hf3d-neural-reco:latest
```

## Using the Run Script

```bash
# Make script executable
chmod +x run-docker.sh

# Run the script
./run-docker.sh

# Select run mode:
# 1. Interactive shell (default)
# 2. Run training
# 3. Run evaluation
# 4. Run custom command
# 5. Use docker-compose (detached mode)
# 6. Run Jupyter notebook
```

## Verify Installation Inside Container

When you start the container, you'll see a welcome banner with GPU information. You can also verify manually:

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
docker exec -it hf3d-container python your_script.py
```

### Installing Additional Packages

```bash
# Inside container (temporary - lost when container stops)
pip install package-name

# To make permanent:
# 1. Add to requirements.txt
# 2. Rebuild the image
docker build -t hf3d-neural-reco:latest .
```

### Using Jupyter Notebook

```bash
# Using the run script
./run-docker.sh
# Select option 6

# Or manually
docker run -it --rm \
  --gpus all \
  -v $(pwd):/workspace \
  -p 8888:8888 \
  -p 6006:6006 \
  --shm-size=8g \
  ghcr.io/artoriasabyssslayer/hashmod-nffbanks-idr:latest \
  bash -c "pip install jupyter tensorboard && \
           jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root"
```

Access at: `http://localhost:8888`

## Volume Mounts Explained

| Mount | Purpose |
|-------|---------|
| `-v $(pwd):/workspace` | Your code and project files |
| `-v $(pwd)/data:/data` | Dataset directory |
| `-v $(pwd)/experiments:/experiments` | Training outputs and logs |
| `-v $(pwd)/checkpoints:/checkpoints` | Model checkpoints |

**Important:** The container only contains dependencies. Your code is mounted at runtime, so any changes you make locally are immediately reflected in the container.

## Container Management

```bash
# List running containers
docker ps

# List all containers (including stopped)
docker ps -a

# Stop container
docker stop hf3d-container

# Start stopped container
docker start -i hf3d-container

# Execute command in running container
docker exec -it hf3d-container bash

# View logs
docker logs -f hf3d-container

# Remove container
docker rm hf3d-container

# Remove image
docker rmi hf3d-neural-reco:latest
docker rmi ghcr.io/artoriasabyssslayer/hashmod-nffbanks-idr:latest
```

## Troubleshooting

### Issue: "could not select device driver" or GPU not available

```bash
# Check nvidia-docker is installed
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# If fails, reinstall nvidia-container-toolkit
sudo pacman -S nvidia-container-toolkit
sudo systemctl restart docker
```

### Issue: "exec format error" when starting container

```bash
# The entrypoint script has wrong line endings
# Fix by recreating docker-entrypoint.sh with Unix line endings
sed -i 's/\r$//' docker-entrypoint.sh
chmod +x docker-entrypoint.sh

# Rebuild the image
docker build -t hf3d-neural-reco:latest .
```

### Issue: "Out of memory" errors

```bash
# Increase shared memory
docker run --shm-size=16g ...  # Default is 8g

# Or in docker-compose.yml:
shm_size: '16gb'
```

### Issue: Permission denied when writing files

The container runs as user `developer` (not root). If you encounter permission issues:

```bash
# From host, fix ownership
sudo chown -R $USER:$USER .

# Or run specific commands as root (not recommended)
docker exec -u root -it hf3d-container bash
```

### Issue: Compilation fails for CUDA packages

```bash
# Check CUDA is accessible inside container
docker run --rm --gpus all hf3d-neural-reco:latest bash -c "nvcc --version"

# If building from source, rebuild with no cache
docker build --no-cache -t hf3d-neural-reco:latest .
```

### Issue: Slow build times

```bash
# Use BuildKit for faster builds
DOCKER_BUILDKIT=1 docker build -t hf3d-neural-reco:latest .

# Or enable BuildKit permanently
echo 'export DOCKER_BUILDKIT=1' >> ~/.bashrc
source ~/.bashrc
```

### Issue: Image pull fails or is very slow

```bash
# Check Docker Hub rate limits
docker pull --quiet ghcr.io/artoriasabyssslayer/hashmod-nffbanks-idr:latest

# Use a mirror or build locally instead
./build-docker.sh
```

## Build Time Estimates

- **Pull pre-built image**: 5-10 minutes (18.2 GB download)
- **Initial build from source**: 20-30 minutes (downloads + compilation)
- **Rebuild after changes**: 2-5 minutes (uses cache)

## Performance Notes

- Docker adds ~2-5% overhead compared to native
- GPU performance is nearly identical to native (>95%)
- I/O performance with mounted volumes is good for code, but may be slower for large datasets
- Use `--shm-size=8g` or higher for data loaders to avoid shared memory issues

## Image Information

- **Base Image**: `nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04`
- **Python Version**: 3.10
- **CUDA Version**: 11.8
- **PyTorch Version**: 2.0.1
- **Image Size**: ~18.2 GB
- **Default User**: `developer` (non-root)
- **Working Directory**: `/workspace`

## Publishing Your Own Version

If you want to publish a modified version:

```bash
# Make sure you're logged into GHCR
docker login ghcr.io -u YOUR_USERNAME

# Tag with your repository
docker tag hf3d-neural-reco:latest ghcr.io/YOUR_USERNAME/YOUR_REPO:latest

# Push to GHCR
docker push ghcr.io/YOUR_USERNAME/YOUR_REPO:latest

# Or use the publish script
./publish-docker.sh v1.0.0
```


## Support

For issues, questions, or contributions:
- **Repository**: https://github.com/ArtoriasAbyssslayer/HashModNFFBanks-IDR
- **Container Registry**: https://github.com/ArtoriasAbyssslayer/HashModNFFBanks-IDR/pkgs/container/hashmod-nffbanks-idr
- **Issues**: https://github.com/ArtoriasAbyssslayer/HashModNFFBanks-IDR/issues