#!/usr/bin/env bash
# Cross-platform Docker run for HashModNFFBanks-IDR

# Image info
IMAGE="ghcr.io/artoriasabyssslayer/hashmod-nffbanks-idr:latest"
WORKDIR="/workspace"

# Detect OS and fix path for volume mount
if [[ "$OSTYPE" == "msys"* || "$OSTYPE" == "cygwin"* ]]; then
    # Git Bash or Cygwin on Windows
    HOST_DIR=$(pwd -W)      # Windows style path
elif [[ "$OSTYPE" == "linux-gnu"* && -n "$WSL_DISTRO_NAME" ]]; then
    # WSL
    HOST_DIR=$(wslpath -a "$(pwd)")
else
    # Linux or macOS
    HOST_DIR=$(pwd)
fi

echo "Using host directory: $HOST_DIR"

# Pull the latest image
docker pull "$IMAGE"

# Run container with current directory mounted
docker run -it --rm --gpus all -v "${HOST_DIR}:${WORKDIR}" "$IMAGE"
