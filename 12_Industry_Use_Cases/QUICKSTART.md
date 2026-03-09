# Quickstart Guide

Get up and running with the AI Makerspace platform in minutes.

---

## Prerequisites

- **Docker & Docker Compose**
- **NVIDIA GPU** (optional, for LLM training)

---

## Step 1: Start Docker Services

Start all services (Qdrant vector DB, Langfuse observability, WebUI):

```bash
docker compose up -d
docker compose exec app python -m src.app
```

Verify services are running:
- **Qdrant**: http://localhost:6333/dashboard (vector database UI)
- **Langfuse**: http://localhost:3000 (LLM observability dashboard)
- **WebUI**: http://localhost:7860
- **Metaflow Dashboard**: http://localhost:3001

First-time Langfuse setup: Create an account at http://localhost:3000

### WebUI Tabs

The WebUI includes:
- **RAG Chat**: Query indexed documentation
- **AutoML**: Automated model selection for tabular data
- **RAGAS Evaluation**: Test retrieval quality (dense/sparse/hybrid methods) with metrics like faithfulness, context_precision, context_recall
- **LLM Fine-tuning**: Agent-based dataset analysis (recommends SFT/DPO/GRPO), training file upload, and error diagnosis

---

## Step 2: Build Knowledge Base Index

Index the scikit-learn documentation into Qdrant:

```bash
cd /home/imjonezz/Desktop/AIMakerspace9/12_Industry_Use_Cases

docker compose exec app python -m src.retrieval.indexer
```

This loads documents from `data/knowledge_base/`, chunks them, generates embeddings, and stores in Qdrant.

---

## Step 3: LLM Training (GPU Required)

For GPU-accelerated LLM training, use the NVIDIA PyTorch container.

**IMPORTANT**: The container `nvcr.io/nvidia/pytorch:26.01-py3` already has the correct PyTorch version installed with CUDA 13.1 support. DO NOT reinstall torch inside the container.

# SFT Training
python src/flows/llm_training_flow.py run \
  --model Qwen/Qwen3.5-9B \
  --dataset data/examples/sft_example.jsonl \
  --method SFT

# DPO Training
python src/flows/llm_training_flow.py run \
  --model Qwen/Qwen3.5-9B \
  --dataset data/examples/dpo_example.jsonl \
  --method DPO

# GRPO Training
python src/flows/llm_training_flow.py run \
  --model Qwen/Qwen3.5-9B \
  --dataset data/examples/grpo_example.jsonl \
  --method GRPO
```

### Viewing Metaflow Runs

Use CLI commands to inspect training runs:

```bash
# List all runs for a flow
metaflow list

# Show detailed information for a specific run
metaflow show <run_id>

# Check the status of the latest run
metaflow status
```

> **Note**: Metaflow UI requires x86_64 architecture. On ARM64 (Apple Silicon), use CLI commands above.

---

## Quick Test Examples

### Tabular ML (AutoML)

Use the WebUI at http://localhost:7860:
1. Navigate to **Tabular ML** tab
2. Click **Download Sample Data** for regression/classification examples
3. Upload data and configure training

Sample data files available:
- `data/examples/regression_example.csv`
- `data/examples/classification_example.csv`

### LLM Training

Example datasets in `data/examples/`:
| File | Method | Description |
|------|--------|-------------|
| `sft_example.jsonl` | SFT | Supervised fine-tuning with conversation pairs |
| `dpo_example.jsonl` | DPO | Direct preference optimization with chosen/rejected pairs |
| `grpo_example.jsonl` | GRPO | Group relative policy optimization with ground truth |

---

## Troubleshooting

### Port Conflicts

If ports are already in use:
```bash
# Check what's using the port
sudo lsof -i :3000   # Langfuse
sudo lsof -i :6333   # Qdrant
sudo lsof -i :7860   # WebUI

# Stop conflicting services or modify ports in docker-compose.yml
```

### Docker Not Running

```bash
# Start Docker daemon
sudo systemctl start docker

# Verify Docker is running
docker ps
```

### CUDA Version Issues

This system uses **CUDA 13.1**. The `bitsandbytes` quantization library only supports up to CUDA 12.x.

**Solution**: Disable quantization and use float16/bfloat16:
```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,  # Use bfloat16 instead of quantization
    device_map="auto"
)
# Do NOT use load_in_4bit=True or load_in_8bit=True
```

With 128GB VRAM, most models run without quantization.

### Qdrant Dimension Mismatch

If changing embedding models, recreate the collection:
```python
from qdrant_client import QdrantClient
client = QdrantClient(host="localhost", port=6333)
client.delete_collection(collection_name="sklearn_docs")
```

Then re-run `docker-compose exec app python -m src.retrieval.indexer`

### Metaflow Username Error

Add `-e USERNAME=docker_user` to Docker run command.

### Torch Import Errors in Container

If you see errors like `AttributeError: module 'torch._C' has no attribute '_dlpack_exchange_api'`, this means torch was incorrectly reinstalled inside the container, conflicting with the pre-installed version.

**Solution**: Reinstall without dependencies:

```bash
pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
pip install --no-deps -e .
```

### Queue ImportError

If you see `cannot import name 'Queue' from 'queue'`, this means Python is importing the local `src/queue/` directory instead of the standard library. This was fixed by renaming to `src/job_queue/`. Make sure you're using the latest code.

### WebUI Fails to Start

Make sure Docker services are running:
```bash
docker-compose up -d
curl http://localhost:6333  # Should return Qdrant info
```

---

## Quick Reference

| Task | Command |
|------|---------|
| Start services | `docker-compose up -d` |
| Stop services | `docker-compose down` |
| Build index | `docker-compose exec app python -m src.retrieval.indexer` |
| Start WebUI | Already running via docker-compose (http://localhost:7860) |
| GPU training container | See Step 3 Docker command |

---

## Next Steps

- Explore the **RAG Chat** tab to query indexed documentation
- Try **AutoML** for automated model selection on your data
- Use **RAGAS Evaluation** tab to compare retrieval methods and metrics
- Use **LLM Fine-tuning** tab for dataset analysis and training configuration
- Fine-tune LLMs with SFT/DPO/GRPO using example datasets
- Monitor experiments in Langfuse at http://localhost:3000