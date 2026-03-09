#!/bin/bash
# Test script for error investigator
# Run inside the NVIDIA PyTorch container

cd /workspace

python3 << 'EOF'
import traceback
from src.agent.error_investigator import investigate_error

# Test 1: CUDA Out of Memory Error
print("=" * 60)
print("TEST 1: CUDA Out of Memory Error")
print("=" * 60)

try:
    raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB. GPU 0: out of memory")
except Exception as e:
    tb = traceback.format_exc()
    for msg in investigate_error(
        error=e,
        traceback_str=tb,
        task_type="llm_training",
        flow_name="LLMTrainingFlow",
        flow_args={
            "data_path": "data/train.jsonl",
            "training_method": "SFT",
            "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
            "epochs": 3,
        },
    ):
        print(msg)

print("\n" + "=" * 60)
print("TEST 2: Data Format Error")
print("=" * 60)

try:
    raise ValueError("JSONDecodeError: Expecting property name enclosed in double quotes")
except Exception as e:
    tb = traceback.format_exc()
    for msg in investigate_error(
        error=e,
        traceback_str=tb,
        task_type="llm_training",
        flow_name="LLMTrainingFlow",
        flow_args={
            "data_path": "data/bad_format.jsonl",
            "training_method": "DPO",
            "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
        },
    ):
        print(msg)

print("\n" + "=" * 60)
print("TEST 3: File Not Found Error")
print("=" * 60)

try:
    raise FileNotFoundError("Dataset file not found: /workspace/data/missing.csv")
except Exception as e:
    tb = traceback.format_exc()
    for msg in investigate_error(
        error=e,
        traceback_str=tb,
        task_type="ml_training",
        flow_name="MLTrainingFlow",
        flow_args={
            "data_path": "/workspace/data/missing.csv",
            "target_column": "price",
        },
    ):
        print(msg)

EOF
