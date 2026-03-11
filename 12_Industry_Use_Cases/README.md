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

- **Deep Error Investigation Agent**
   - Multi-agent LangGraph system for debugging ML/LLM errors
   - Semantic memory with Qdrant for learning from past errors
   - Automatic search query refinement for better solutions
   - User-friendly error explanations in the WebUI with actionable recommendations

- **Intelligent Model Selection**
   - RAG-based automatic model selection from sklearn documentation
   - Dataset analysis with training method recommendations (SFT/DPO/GRPO)
   - Zero ML knowledge required for non-technical users

- **Full Observability**
   - Langfuse integration with comprehensive tracing
   - LLM call tracking, retrieval spans, and training flow monitoring

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
```

## Project Structure

```
├── src/
│   ├── app.py              # Main Gradio WebUI application (5 tabs)
│   ├── config.py           # Centralized configuration
│   ├── retrieval/          # Vector search and indexing
│   │   └── indexer.py      # Knowledge base indexer
│   ├── flows/              # Metaflow training pipelines
│   │   ├── runner.py       # Flow execution handlers
│   │   └── llm_training_flow.py  # LLM fine-tuning flow
│   ├── agent/              # Agentic components
│   │   ├── investigation_graph.py   # LangGraph orchestration for errors
│   │   └── sub_agents/     # Specialized agent nodes (query, search, synthesis)
│   ├── llm/                # LLM inference and training
│   │   ├── local_inference.py     # Local model loading with LoRA support
│   │   └── inference_server.py    # LLM generation API
│   ├── ml/                 # Tabular ML components
│   │   └── data_validator.py      # Data validation and model recommendation
│   ├── evaluation/         # RAGAS evaluation
│   │   └── ragas_evaluator.py     # Retrieval quality metrics
│   └── utils/              # Utilities
│       ├── langfuse_client.py     # Langfuse tracing integration
│       └── error_handling.py      # User-friendly error formatting
├── data/
│   ├── examples/           # Sample datasets for testing
│   └── knowledge_base/     # Documentation for RAG indexing
├── docker-compose.yml      # Docker services (Qdrant, Langfuse)
├── QUICKSTART.md           # Detailed setup guide
└── README.md               # This file
```

## Training Methods

### Tabular ML Ensemble Training

Automated ensemble training combining multiple gradient boosting and tree-based models for optimal tabular data predictions. Features:

- RAG-powered model selection from sklearn documentation
- Automatic task type detection (classification/regression)
- Ensemble: RandomForest + XGBoost + LightGBM
- Hyperparameter optimization

### SFT (Supervised Fine-Tuning)

Standard supervised fine-tuning for adapting LLMs to specific tasks using labeled datasets. Supports:

- JSONL format with messages field or Q:/A: text format
- LoRA adapter training for efficient fine-tuning
- Automatic model and tokenizer saving

### DPO (Direct Preference Optimization)

Preference-based alignment method that trains models directly on preference pairs without a separate reward model. Requires:

- JSONL format with `prompt`, `chosen`, and `rejected` fields
- Comparison-based learning for alignment

### GRPO (Group Relative Policy Optimization)

Advanced RLHF method for group-wise policy optimization with relative rewards. Features:

- JSONL format with `prompt`, `ground_truth` (optional), and `pattern` (optional)
- Built-in reward templates: math, format_check
- Custom reward function support

## Example Data

Sample datasets are available in `data/examples/` for testing and experimentation.

## WebUI Access

After starting the application, access the Gradio interface at:

```
http://localhost:7860
```

The WebUI provides 5 tabs:
1. **Tabular ML** - Automated model selection and ensemble training
2. **LLM Fine-tuning** - SFT/DPO/GRPO with agent-based dataset analysis
3. **Inference Playground** - Test trained tabular models
4. **LLM Inference** - Test fine-tuned LLMs with LoRA adapters
5. **RAGAS Evaluation** - Evaluate retrieval quality

## Recent Improvements

- ✅ Corrected all LLM server configurations (model name, URLs)
- ✅ Fixed streaming generation and import bugs
- ✅ Added full Langfuse observability with comprehensive tracing
- ✅ Improved error investigation results now appear in Gradio UI
- ✅ Consolidated configuration in `src/config.py`
- ✅ Removed unused code and development artifacts

## License

MIT
