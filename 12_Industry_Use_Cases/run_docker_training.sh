#!/bin/bash
# Script to run LLM training in NVIDIA Docker container

set -e

echo "=========================================="
echo "LLM Training with GPU Acceleration"
echo "=========================================="

# Check if nvidia-docker is available
if ! docker run --rm nvcr.io/nvidia/pytorch:26.01-py3 nvidia-smi &> /dev/null; then
    echo "ERROR: NVIDIA Docker runtime not available!"
    echo "Please ensure:"
    echo "  1. NVIDIA GPU drivers are installed"
    echo "  2. NVIDIA Container Toolkit is installed"
    echo "  3. Docker is configured to use nvidia runtime"
    exit 1
fi

echo "✓ NVIDIA Docker runtime detected"

# Run the container
docker compose -f docker-compose.training.yml run --rm llm-training