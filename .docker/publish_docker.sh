#!/bin/bash
set -euo pipefail

echo "============================================="
echo "HashModNFFBanks-IDR Docker Publish Script"
echo "============================================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Config
LOCAL_IMAGE="hf3d-neural-reco"
GITHUB_USER="artoriasabyssslayer"
REPO_NAME="hashmod-nffbanks-idr"
VERSION="${1:-latest}"

REMOTE_IMAGE="ghcr.io/${GITHUB_USER}/${REPO_NAME}"

echo -e "${BLUE}Local Image: ${LOCAL_IMAGE}:latest${NC}"
echo -e "${BLUE}Target: ${REMOTE_IMAGE}:${VERSION}${NC}"
echo ""

# --- Step 1: Verify local image exists ---
if ! docker images | grep -q "$LOCAL_IMAGE.*latest"; then
    echo -e "${RED}Error: Image '${LOCAL_IMAGE}:latest' not found${NC}"
    echo "Build it first with: ./build-docker.sh"
    exit 1
fi

# --- Step 2: Verify /workspace is empty ---
echo -e "${YELLOW}Verifying image /workspace...${NC}"
WORKSPACE_FILES=$(docker run --rm "${LOCAL_IMAGE}:latest" sh -c "ls -A /workspace 2>/dev/null || echo 'empty'")
if [ "$WORKSPACE_FILES" != "empty" ] && [ -n "$WORKSPACE_FILES" ]; then
    echo -e "${RED}Warning: Image contains files in /workspace:${NC}"
    echo "$WORKSPACE_FILES"
    echo -e "${YELLOW}Image should be empty and mount workspace at runtime!${NC}"
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# --- Step 3: GHCR Login ---
echo -e "${YELLOW}Checking GHCR login...${NC}"
if [ -f "$HOME/.github_access_token_ghcr" ]; then
    cat "$HOME/.github_access_token_ghcr" | docker login ghcr.io -u "$GITHUB_USER" --password-stdin 2>/dev/null || {
        echo -e "${YELLOW}Already logged in, continuing...${NC}"
    }
else
    echo -e "${YELLOW}Login to GitHub Container Registry...${NC}"
    docker login ghcr.io -u "$GITHUB_USER"
fi

# --- Step 4: Tag images ---
echo -e "${YELLOW}Tagging images...${NC}"
docker tag "${LOCAL_IMAGE}:latest" "${REMOTE_IMAGE}:latest"
docker tag "${LOCAL_IMAGE}:latest" "${REMOTE_IMAGE}:${VERSION}"

echo -e "${GREEN}✓ Tagged as:${NC}"
echo "  ${REMOTE_IMAGE}:latest"
echo "  ${REMOTE_IMAGE}:${VERSION}"
echo ""

# --- Step 5: Push images ---
echo -e "${YELLOW}Pushing to GitHub Container Registry...${NC}"
docker push "${REMOTE_IMAGE}:latest"
echo -e "${GREEN}✓ Pushed ${REMOTE_IMAGE}:latest${NC}"
echo ""

if [ "$VERSION" != "latest" ]; then
    docker push "${REMOTE_IMAGE}:${VERSION}"
    echo -e "${GREEN}✓ Pushed ${REMOTE_IMAGE}:${VERSION}${NC}"
    echo ""
fi

# --- Step 6: Print usage instructions ---
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}✓ Successfully published!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Repository: https://github.com/${GITHUB_USER}/HashModNFFBanks-IDR"
echo "Package URL: https://github.com/${GITHUB_USER}/HashModNFFBanks-IDR/pkgs/container/${REPO_NAME}"
echo ""
echo "Users can pull with:"
echo -e "  ${BLUE}docker pull ${REMOTE_IMAGE}:${VERSION}${NC}"
echo ""
echo "Run with (Linux/macOS):"
echo -e "  ${BLUE}docker run -it --rm --gpus all -v \$(pwd):/workspace ${REMOTE_IMAGE}:${VERSION}${NC}"
echo ""
echo "Run with (Windows PowerShell + WSL2):"
echo -e "  ${BLUE}`$wslPath = wsl -e wslpath (Get-Location).Path; docker run -it --rm --gpus all -v `\$wslPath:/workspace ${REMOTE_IMAGE}:${VERSION}${NC}"
echo ""
echo "Next steps:"
echo "  1. Go to: https://github.com/users/${GITHUB_USER}/packages"
echo "  2. Click on '${REPO_NAME}'"
echo "  3. Click 'Connect repository' → select 'HashModNFFBanks-IDR'"
echo "  4. Package settings → Change visibility → Public"
