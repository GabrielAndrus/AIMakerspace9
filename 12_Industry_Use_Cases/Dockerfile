FROM nvcr.io/nvidia/pytorch:26.01-py3

LABEL maintainer="AI Maker Space <contact@aimakerspace.io>"
LABEL description="Agentic AutoML Platform for training ML models and fine-tuning LLMs"

ENV METAFLOW_DEFAULT_METADATA=local
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./

RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
    "gradio>=5.0" \
    "metaflow>=2.12" \
    "scikit-learn>=1.5" \
    "sentence-transformers>=2.2" \
    "xgboost>=2.0" \
    "lightgbm>=4.3" \
    "pandas>=2.0" \
    "numpy>=1.24" \
    "trl>=0.12" \
    "transformers>=4.40" \
    "accelerate>=0.30" \
    "peft>=0.10" \
    "datasets>=2.14" \
    "openpyxl>=3.1" \
    "pypdf>=4.0" \
    "qdrant-client>=1.7" \
    "ragas>=0.1" \
    "openai>=1.0" \
    "langchain>=0.1.0" \
    "langchain-text-splitters>=0.0.0" \
    "langfuse>=3.0" \
    "langgraph>=0.0.50" \
    "matplotlib>=3.7" \
    "joblib>=1.3" \
    "python-dotenv>=1.0"

COPY src/ ./src/
COPY data/ ./data/

RUN mkdir -p /workspace/data /workspace/models

EXPOSE 7860

ENV HF_HOME=/root/.cache/huggingface

CMD ["bash"]