#!/bin/bash
# Validation script for NVIDIA PyTorch Docker configuration

set -e

echo "=========================================="
echo "Docker GPU Support Validation"
echo "=========================================="
echo ""

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

pass_count=0
fail_count=0

check_pass() {
    echo -e "${GREEN}✓ PASS${NC}: $1"
    ((pass_count++))
}

check_fail() {
    echo -e "${RED}✗ FAIL${NC}: $1"
    ((fail_count++))
}

check_info() {
    echo -e "${YELLOW}ℹ INFO${NC}: $1"
}

# Test 1: Verify base image
echo "Checking Dockerfile configuration..."
echo ""

if grep -q "^FROM nvcr.io/nvidia/pytorch:26.01-py3 AS builder" docker/Dockerfile; then
    check_pass "Builder stage uses correct NVIDIA base image"
else
    check_fail "Builder stage does not use nvcr.io/nvidia/pytorch:26.01-py3"
fi

if grep -q "^FROM nvcr.io/nvidia/pytorch:26.01-py3 AS production" docker/Dockerfile; then
    check_pass "Production stage uses correct NVIDIA base image"
else
    check_fail "Production stage does not use nvcr.io/nvidia/pytorch:26.01-py3"
fi

if grep -q "python:3.11-slim" docker/Dockerfile; then
    check_fail "Dockerfile still contains python:3.11-slim reference"
else
    check_pass "No python:3.11-slim references found (correct)"
fi

echo ""

# Test 2: Verify UV installation
if grep -q "RUN which uv || pip install --no-cache-dir uv" docker/Dockerfile; then
    check_pass "UV conditional installation logic present"
else
    check_fail "UV installation logic missing or incorrect"
fi

echo ""

# Test 3: Verify multi-stage build
if grep -q "COPY --from=builder /opt/venv /opt/venv" docker/Dockerfile; then
    check_pass "Multi-stage build with venv copy configured"
else
    check_fail "Venv copy from builder to production missing"
fi

echo ""

# Test 4: Verify entrypoint GPU checks
if grep -q "command -v nvidia-smi" docker/entrypoint.sh; then
    check_pass "Entrypoint includes nvidia-smi command check"
else
    check_fail "Entrypoint missing nvidia-smi check"
fi

if grep -q "nvidia-smi --query-gpu=name,memory.total,memory.free,compute_cap" docker/entrypoint.sh; then
    check_pass "GPU query includes all required fields (name, memory, compute_cap)"
else
    check_fail "GPU query missing required fields"
fi

if grep -q "WARNING: No NVIDIA GPU detected" docker/entrypoint.sh; then
    check_pass "CPU fallback warning message present"
else
    check_fail "CPU fallback warning missing"
fi

if grep -q "nvidia-container-toolkit" docker/entrypoint.sh; then
    check_pass "Actionable advice for GPU support present"
else
    check_fail "Missing advice about nvidia-container-toolkit installation"
fi

echo ""

# Test 5: Verify file permissions
if [ -x "docker/entrypoint.sh" ]; then
    check_pass "Entrypoint script is executable"
else
    check_fail "Entrypoint script lacks execute permissions"
fi

echo ""

# Test 6: Verify no Python installation
if grep -qi "apt-get.*python\|yum.*python" docker/Dockerfile; then
    check_fail "Found Python installation command (should use NVIDIA image's Python)"
else
    check_pass "No Python installation commands (correct - using NVIDIA base image's Python)"
fi

echo ""

# Test 7: Check for proper environment setup
if grep -q "PYTHONUNBUFFERED=1" docker/entrypoint.sh; then
    check_pass "PYTHONUNBUFFERED environment variable set"
else
    check_fail "PYTHONUNBUFFERED not set in entrypoint"
fi

if grep -q "PYTHONDONTWRITEBYTECODE=1" docker/entrypoint.sh; then
    check_pass "PYTHONDONTWRITEBYTECODE environment variable set"
else
    check_fail "PYTHONDONTWRITEBYTECODE not set in entrypoint"
fi

echo ""

# Test 8: Verify security (non-root user)
if grep -q "useradd --create-home" docker/Dockerfile; then
    check_pass "Non-root user (appuser) created"
else
    check_fail "Missing non-root user creation"
fi

if grep -q "^USER appuser" docker/Dockerfile; then
    check_pass "Container runs as non-root user"
else
    check_fail "Missing USER directive for appuser"
fi

echo ""

# Summary
echo "=========================================="
echo "Validation Summary"
echo "=========================================="
echo ""
echo -e "${GREEN}Passed:${NC} $pass_count"
echo -e "${RED}Failed:${NC} $fail_count"
echo ""

if [ $fail_count -eq 0 ]; then
    echo -e "${GREEN}All checks passed!${NC}"
    echo ""
    check_info "The Docker configuration is correctly set up for GPU support."
    check_info "Base image: nvcr.io/nvidia/pytorch:26.01-py3"
    check_info "CUDA version: 13.x (supports Blackwell GPUs)"
    echo ""
    echo "Next steps:"
    echo "  1. Build the container: docker build -f docker/Dockerfile . -t automl-platform:latest"
    echo "  2. Run with GPU: docker run --gpus all -p 7860:7860 automl-platform:latest"
    echo "  3. Run without GPU (CPU mode): docker run -p 7860:7860 automl-platform:latest"
    exit 0
else
    echo -e "${RED}Some checks failed!${NC}"
    echo ""
    check_info "Please review the failures above and fix them."
    exit 1
fi