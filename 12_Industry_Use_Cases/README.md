# Agentic AutoML Platform

An intelligent platform for ML model training and LLM fine-tuning with agentic RAG-based model selection.

## Features

- **Tabular ML (Classification/Regression)**
  - Ensemble training with RandomForest + XGBoost + LightGBM
  - Automated hyperparameter optimization
  
- **LLM Fine-tuning Methods**
  - SFT (Supervised Fine-Tuning)
  - DPO (Direct Preference Optimization)
  - GRPO (Group Relative Policy Optimization)

- **Vector Search Knowledge Base**
  - Qdrant vector database for semantic search
  - Qwen3-Embedding-4B for embeddings

- **Observability**
  - Langfuse integration for experiment tracking and monitoring

- **Web Interface**
  - Gradio WebUI for interactive model training and inference

## Requirements

- Python 3.10+
- Docker (for Qdrant, Langfuse)
- NVIDIA GPU container (`nvcr.io/nvidia/pytorch:26.01-py3`) for LLM training with Flash Attention
- 128GB+ VRAM recommended for larger models like Qwen3.5-9B

## Quick Start

See [quickstart.md](quickstart.md) for detailed setup instructions.

```bash
uv pip install -e .
docker-compose up -d
python -m src.retrieval.indexer
python src/app.py
```

## Project Structure

```
├── src/
│   ├── app.py              # Main Gradio WebUI application
│   ├── retrieval/          # Vector search and indexing
│   │   └── indexer.py      # Knowledge base indexer
│   ├── training/           # Training pipelines
│   └── agents/             # Agentic components
├── data/
│   └── examples/           # Sample datasets for testing
├── docker-compose.yml      # Docker services (Qdrant, Langfuse)
├── quickstart.md           # Detailed setup guide
└── README.md               # This file
```

## Training Methods

### Tabular ML Ensemble Training

Automated ensemble training combining multiple gradient boosting and tree-based models for optimal tabular data predictions.

### SFT (Supervised Fine-Tuning)

Standard supervised fine-tuning for adapting LLMs to specific tasks using labeled datasets.

### DPO (Direct Preference Optimization)

Preference-based alignment method that trains models directly on preference pairs without a separate reward model.

### GRPO (Group Relative Policy Optimization)

Advanced RLHF method for group-wise policy optimization with relative rewards.

## Example Data

Sample datasets are available in `data/examples/` for testing and experimentation.

## WebUI Access

After starting the application, access the Gradio interface at:

```
http://localhost:7860
```

## License

MIT