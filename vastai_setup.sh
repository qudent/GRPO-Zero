#!/bin/bash
# GRPO-Zero vast.ai setup script
# Run this on a fresh vast.ai instance to get training started
set -euo pipefail

echo "=== GRPO-Zero vast.ai Setup ==="

cd /workspace

# Clone the repo if not already present
if [ ! -d "grpo_zero" ]; then
    git clone https://github.com/policy-gradient/GRPO-Zero.git grpo_zero
fi
cd grpo_zero

# Install uv
if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Install dependencies
uv sync

# Install git-lfs
apt-get update -qq && apt-get install -y -qq git-lfs
git lfs install

# Download dataset
if [ ! -d "Countdown-Tasks-3to4" ]; then
    echo "=== Downloading dataset ==="
    git clone https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4
fi

# Download pretrained model
if [ ! -d "Qwen2.5-3B-Instruct" ]; then
    echo "=== Downloading Qwen2.5-3B-Instruct ==="
    git clone https://huggingface.co/Qwen/Qwen2.5-3B-Instruct
fi

# Detect VRAM and pick config
VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
VRAM_GB=$((VRAM_MB / 1024))
echo "=== Detected ${VRAM_GB}GB VRAM ==="

if [ "$VRAM_GB" -ge 40 ]; then
    CONFIG="config.yaml"
    echo "Using default config (48GB+ VRAM)"
else
    CONFIG="config_24GB.yaml"
    echo "Using 24GB config with CPU optimizer offloading"
fi

echo ""
echo "=== Setup complete! ==="
echo "To start training:"
echo "  cd /workspace/grpo_zero"
echo "  uv run train.py --config $CONFIG"
echo ""
echo "To monitor with tensorboard:"
echo "  uv run tensorboard --logdir logs --bind_all"
