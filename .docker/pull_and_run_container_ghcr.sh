# Pull the pre-built image
docker pull ghcr.io/artoriasabyssslayer/hashmod-nffbanks-idr:latest

# Run the container
docker run -it --rm \
  --gpus all \
  -v $(pwd):/workspace \
  ghcr.io/artoriasabyssslayer/hashmod-nffbanks-idr:latest