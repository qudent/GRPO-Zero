#!/bin/bash
# Launch a vast.ai instance for GRPO-Zero training
# Usage: ./vastai_launch.sh [gpu_type]
# gpu_type: a40 (default), 4090, a100, h100
set -euo pipefail

GPU_TYPE="${1:-a40}"

case "$GPU_TYPE" in
    3090)
        QUERY='gpu_name=RTX_3090 gpu_ram>=24 num_gpus=1 reliability>0.95 inet_down>200 cuda_vers>=12.0 disk_space>=80'
        echo "Searching for RTX 3090 (~\$0.16/hr, 24GB, needs CPU offload config)"
        ;;
    4090)
        QUERY='gpu_name=RTX_4090 num_gpus=1 reliability>0.95 inet_down>200 cuda_vers>=12.0 disk_space>=80'
        echo "Searching for RTX 4090 (~\$0.39-0.54/hr, 24GB, fastest gen speed per dollar)"
        ;;
    a40)
        QUERY='gpu_name=A40 num_gpus=1 reliability>0.95 inet_down>200 cuda_vers>=12.0 disk_space>=80'
        echo "Searching for A40 (~\$0.40/hr, 48GB, recommended default)"
        ;;
    a100)
        QUERY='gpu_name=A100_SXM4 num_gpus=1 reliability>0.95 inet_down>200 cuda_vers>=12.0 disk_space>=80'
        echo "Searching for A100 SXM4 (~\$0.56-1.0/hr, 80GB, bigger batches possible)"
        ;;
    h100)
        QUERY='gpu_name=H100_SXM num_gpus=1 reliability>0.95 inet_down>200 cuda_vers>=12.0 disk_space>=80'
        echo "Searching for H100 SXM (~\$1.47-1.56/hr, 80GB, fastest)"
        ;;
    *)
        echo "Unknown GPU type: $GPU_TYPE"
        echo "Options: 3090, 4090, a40, a100, h100"
        exit 1
        ;;
esac

echo ""
echo "Available offers:"
vastai search offers "$QUERY" -o 'dph' --limit 5
echo ""

# Find cheapest offer ID
OFFER_ID=$(vastai search offers "$QUERY" -o 'dph' --limit 1 --raw 2>/dev/null | python3 -c "import json,sys; print(json.load(sys.stdin)[0]['id'])")

if [ -z "$OFFER_ID" ]; then
    echo "No offers found for $GPU_TYPE"
    exit 1
fi

echo "Cheapest offer ID: $OFFER_ID"
echo ""
read -p "Create instance with this offer? [y/N] " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    ONSTART='bash -c "cd /workspace && git clone https://github.com/policy-gradient/GRPO-Zero.git grpo_zero 2>/dev/null; cd grpo_zero && bash vastai_setup.sh"'
    vastai create instance "$OFFER_ID" \
        --image pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel \
        --disk 100 \
        --onstart-cmd "$ONSTART" \
        --direct
    echo "Instance created! Monitor with: vastai show instances"
    echo "SSH in with: vastai ssh-url <instance_id>"
else
    echo "Aborted."
fi
