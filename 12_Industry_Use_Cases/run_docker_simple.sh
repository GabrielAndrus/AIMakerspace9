#!/bin/bash
# Simplified Docker training script

set -e

echo "=========================================="
echo "LLM Training with GPU Acceleration"
echo "=========================================="

docker run --rm \
  --runtime=nvidia \
  --gpus all \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e METAFLOW_DEFAULT_METADATA=local \
  -v $(pwd):/workspace \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  nvcr.io/nvidia/pytorch:26.01-py3 \
  bash -c "
    cd /workspace &&
    
    echo '=== Installing Flash Attention ===' &&
    pip install --no-cache-dir flash-attn --no-build-isolation &&
    
    echo '=== Installing project dependencies ===' &&
    pip install --no-cache-dir -e . &&
    
    echo '=== Running GPU check ===' &&
    python check_gpu.py &&
    
    echo '' &&
    echo '=== Starting training test ===' &&
    python test_docker_training.py
  "