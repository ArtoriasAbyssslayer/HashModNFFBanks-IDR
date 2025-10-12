#!/bin/bash

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# ASCII Art Banner
echo ""
echo -e "${GREEN}"
cat << "EOF"
    ╦ ╦╔═╗  ╔═╗╔╦╗  ╔╗╔╔═╗╦ ╦╦═╗╔═╗╦    ╦═╗╔═╗╔═╗╔═╗
    ╠═╣╠╣    ═╗ ║║  ║║║║╣ ║ ║╠╦╝╠═╣║    ╠╦╝║╣ ║  ║ ║
    ╩ ╩╚    ╚═╝═╩╝  ╝╚╝╚═╝╚═╝╩╚═╩ ╩╩═╝  ╩╚═╚═╝╚═╝╚═╝
EOF
echo -e "${CYAN}    ════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}         Hash-based Multi-Resolution Feature Banks${NC}"
echo -e "${BLUE}            3D Neural Surface Reconstruction${NC}"
echo -e "${CYAN}    ════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${GREEN}    ⚙  Environment:${NC}  CUDA 11.8 + PyTorch 2.0.1"
echo -e "${GREEN}    👤 User:${NC}         developer"
echo -e "${GREEN}    📁 Workspace:${NC}    /workspace"
echo ""

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo -e "${YELLOW}🚀 GPU Status:${NC}"
    nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader,nounits 2>/dev/null | \
    while IFS=, read -r idx name driver mem; do
        echo -e "   ${GREEN}[GPU $idx]${NC} $name (Driver: $driver, Memory: ${mem}MB)"
    done
    echo ""
fi

# Check PyTorch CUDA
if python -c "import torch" 2>/dev/null; then
    CUDA_AVAILABLE=$(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)
    CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)" 2>/dev/null)
    GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null)
    
    if [ "$CUDA_AVAILABLE" = "True" ]; then
        echo -e "${GREEN}✓${NC} PyTorch CUDA: ${GREEN}Available${NC} (CUDA $CUDA_VERSION, $GPU_COUNT GPU(s))"
    else
        echo -e "${YELLOW}⚠${NC} PyTorch CUDA: ${YELLOW}Not Available${NC}"
    fi
    echo ""
fi

# Quick start guide
echo -e "${CYAN}Quick Start Commands:${NC}"
echo -e "  ${GREEN}•${NC} Train:      ${BLUE}.code/scripts/run_training_failsafe.sh --exp HashGrid --scan_id 114${NC}"
echo -e "  ${GREEN}•${NC} Evaluate:   ${BLUE}.code/scripts/run_evaluation_failsafe.sh --exp HashGrid --scan_id 114${NC}"
echo -e "  ${GREEN}•${NC} Test CUDA:  ${BLUE}python -c 'import torch; print(torch.cuda.is_available())'${NC}"
echo ""
echo -e "${CYAN}Repository:${NC} https://github.com/ArtoriasAbyssslayer/HashModNFFBanks-IDR"
echo ""

# Execute the command passed to the entrypoint
exec "$@"