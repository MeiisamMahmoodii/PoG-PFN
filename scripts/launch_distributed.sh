#!/bin/bash

# Launch script for distributed training on multiple GPUs
# 
# Usage:
#   # Train on all available GPUs
#   bash scripts/launch_distributed.sh
#
#   # Train on specific number of GPUs
#   bash scripts/launch_distributed.sh 2
#
#   # Train on specific GPUs
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/launch_distributed.sh 4

# Number of GPUs (default: all available)
NGPUS=${1:-$(nvidia-smi --list-gpus | wc -l)}

echo "========================================="
echo "Launching Distributed Training"
echo "========================================="
echo "Number of GPUs: $NGPUS"
echo "Script: scripts/train_distributed.py"
echo "========================================="
echo ""

# Launch with torchrun (recommended for PyTorch 1.10+)
torchrun \
    --nproc_per_node=$NGPUS \
    --master_port=29500 \
    scripts/train_distributed.py

# Alternative: Use python -m torch.distributed.launch (older PyTorch)
# python -m torch.distributed.launch \
#     --nproc_per_node=$NGPUS \
#     --master_port=29500 \
#     scripts/train_distributed.py

echo ""
echo "========================================="
echo "Distributed Training Complete!"
echo "========================================="
