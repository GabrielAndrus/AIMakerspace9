# Agentic AutoML Platform - Project Plan

## Overview

Building an Agentic AutoML platform that allows users to upload datasets (CSV, TXT, PDF) and train ML models or fine-tune LLMs through a Gradio web interface. The system uses Metaflow for pipeline orchestration and runs in Docker containers on NVIDIA Spark infrastructure.

---

## Project Vision

### Problem Statement
Users need a simple way to train ML models on their tabular data or fine-tune LLMs on custom datasets without writing code. Current solutions require significant ML expertise.

### Solution
A web-based AutoML platform that:
1. Accepts file uploads (CSV for tabular ML, TXT/PDF for LLM fine-tuning)
2. Auto-detects task types and recommends appropriate models
3. Trains ensemble ML models or fine-tunes LLMs with minimal configuration
4. Provides downloadable trained models and an inference playground

### Target Users
- Data scientists who want quick baselines
- Domain experts without ML coding experience
- Teams needing rapid prototyping of ML solutions

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Docker Container                          │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    Gradio Web Interface                    │  │
│  │  ┌─────────────┐ ┌──────────────┐ ┌──────────────────┐   │  │
│  │  │ Tabular ML  │ │ LLM Training │ │ Inference        │   │  │
│  │  │ Tab         │ │ Tab          │ │ Playground       │   │  │
│  │  └─────────────┘ └──────────────┘ └──────────────────┘   │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              │                                   │
│                              ▼                                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                   Job Queue (SQLite)                       │  │
│  │              Handles 5-20 concurrent users                 │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              │                                   │
│                              ▼                                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                  Metaflow Pipelines                        │  │
│  │  ┌─────────────────┐    ┌──────────────────────────────┐ │  │
│  │  │ ML Training     │    │ LLM Fine-tuning              │ │  │
│  │  │ Flow            │    │ Flow                         │ │  │
│  │  └─────────────────┘    └──────────────────────────────┘ │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              │                                   │
│                              ▼                                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                  Model Storage (Ephemeral)                 │  │
│  │         Models downloaded after training, not persisted    │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  Base Image: nvcr.io/nvidia/pytorch:26.01-py3                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
12_Industry_Use_Cases/
├── src/
│   ├── __init__.py
│   ├── app.py                    # Main Gradio application entry point
│   ├── config.py                 # Configuration management (env vars, paths)
│   │
│   ├── ml/                       # Tabular ML components
│   │   ├── __init__.py
│   │   ├── data_validator.py     # CSV validation, type detection, error messages
│   │   ├── auto_ensemble.py      # Ensemble training (Voting/Stacking)
│   │   └── inference_server.py   # ML model inference for playground
│   │
│   ├── llm/                      # LLM fine-tuning components  
│   │   ├── __init__.py
│   │   ├── dataset_converter.py  # Convert TXT/PDF to training formats
│   │   ├── trainers/             # Training scripts by method
│   │   │   ├── __init__.py
│   │   │   ├── sft_trainer.py    # Supervised Fine-Tuning with Unsloth
│   │   │   ├── dpo_trainer.py    # Direct Preference Optimization
│   │   │   └── grpo_trainer.py   # Group Relative Policy Optimization
│   │   └── inference_server.py   # LLM inference for playground
│   │
│   ├── flows/                    # Metaflow pipelines
│   │   ├── __init__.py
│   │   ├── ml_training_flow.py   # End-to-end ML training pipeline
│   │   └── llm_training_flow.py  # End-to-end LLM training pipeline
│   │
│   ├── queue/                    # Job management for concurrency
│   │   ├── __init__.py
│   │   ├── job_manager.py        # SQLite-backed job queue
│   │   └── task_status.py        # Status tracking and progress updates
│   │
│   └── utils/
│       ├── __init__.py
│       ├── serialization.py      # Model saving/loading utilities
│       └── error_handling.py     # Descriptive, actionable error messages
│
├── docker/
│   ├── Dockerfile                # Multi-stage production build
│   └── entrypoint.sh             # Container startup with signal handling
│
├── tests/
│   ├── __init__.py
│   ├── test_ml_pipeline.py       # Tests for tabular ML pipeline
│   └── test_llm_pipeline.py      # Tests for LLM fine-tuning pipeline
│
├── data/                         # Temporary data storage (gitignored)
├── models/                       # Temporary model storage (gitignored)
│
├── pyproject.toml                # UV project configuration
├── uv.lock                       # Dependency lock file
├── README.md                     # Project documentation
├── AGENTS.md                     # Design patterns and instructions (update this)
└── todo.md                       # This file
```

---

## Technical Research Summary

### 1. SKLearn Ensemble Training Best Practices

#### VotingClassifier (Recommended for simplicity)
```python
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Soft voting averages probabilities - better for calibrated models
ensemble = VotingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('xgb', XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss')),
        ('lgbm', LGBMClassifier(n_estimators=100, verbose=-1)),
    ],
    voting='soft',
    weights=[1, 2, 2]  # XGBoost and LightGBM often perform better
)
```

#### StackingClassifier (For learning optimal combinations)
```python
from sklearn.ensemble import StackingClassifier

stacking = StackingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(n_estimators=100)),
        ('xgb', XGBClassifier()),
        ('lgbm', LGBMClassifier(verbose=-1)),
    ],
    final_estimator=LogisticRegression(),
    cv=5,
    passthrough=False  # Set True to include original features
)
```

#### Regression Ensembles
```python
from sklearn.ensemble import VotingRegressor, StackingRegressor

# For regression tasks (continuous target)
voting_reg = VotingRegressor([
    ('rf', RandomForestRegressor(n_estimators=100)),
    ('xgb', XGBRegressor()),
    ('lgbm', LGBMRegressor(verbose=-1)),
])
```

#### Key Principles
- **Diversity is key**: Use models with different inductive biases
- **Uncorrelated errors**: Models should make mistakes on different samples
- **Performance baseline**: Each model should outperform random guessing
- **3-7 diverse models**: Diminishing returns beyond that

---

### 2. Metaflow for ML Pipeline Orchestration

#### Basic Flow Structure
```python
from metaflow import FlowSpec, step, Parameter

class MLTrainingFlow(FlowSpec):
    # Parameters for configuration
    data_path = Parameter('data', required=True)
    target_column = Parameter('target', required=True)
    
    @step
    def start(self):
        self.random_seed = 42
        self.next(self.load_data)
    
    @step
    def load_data(self):
        import pandas as pd
        self.df = pd.read_csv(self.data_path)
        self.next(self.validate_data)
    
    @step
    def validate_data(self):
        # Validation logic
        self.next(self.train_model)
    
    @step
    def train_model(self):
        # Training logic
        self.model = None  # Trained model
        self.next(self.evaluate)
    
    @step
    def evaluate(self):
        # Evaluation logic
        self.accuracy = 0.95
        self.next(self.save_model)
    
    @step
    def save_model(self):
        import joblib
        self.model_path = "model.pkl"
        joblib.dump(self.model, self.model_path)
        self.next(self.end)
    
    @step
    def end(self):
        print(f"Model saved to {self.model_path}")

if __name__ == '__main__':
    MLTrainingFlow()
```

#### Key Decorators
- `@step` - Defines a workflow step
- `@resources(gpu=1, memory=32000)` - Request compute resources
- `@card` - Generate visual reports for observability
- `@conda_base` - Manage dependencies

#### Running Flows
```bash
# Local execution
python flow.py run --data /path/to/data.csv --target label

# With parameters
python flow.py run --param value
```

---

### 3. LLM Dataset Formats for Fine-Tuning

#### SFT (Supervised Fine-Tuning)
```python
# Conversational format (preferred for instruction-following)
{"messages": [
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "The answer is 4."}
]}

# Plain text format (for language modeling)
{"text": "The sky is blue."}
```

#### DPO (Direct Preference Optimization)
```python
# Preference pairs format
{
    "prompt": [{"role": "user", "content": "What is AI?"}],
    "chosen": [{"role": "assistant", "content": "AI is artificial intelligence..."}],
    "rejected": [{"role": "assistant", "content": "I don't know."}]
}
```

#### GRPO (Group Relative Policy Optimization)
```python
# Prompt-only format with reward function
{"prompt": [{"role": "user", "content": "What is 15 + 27?"}]}

# Reward function computes score based on correctness
def accuracy_reward_func(completions, ground_truth, **kwargs):
    import re
    rewards = []
    for completion, gt in zip(completions, ground_truth):
        match = re.search(r"\\boxed\{(.*?)\}", completion)
        answer = match.group(1) if match else ""
        rewards.append(1.0 if answer == gt else 0.0)
    return rewards
```

#### Dataset Conversion Patterns

**TXT to SFT format:**
```python
# Line-by-line text → messages format
def txt_to_messages(txt_content: str) -> list:
    lines = txt_content.strip().split('\n')
    return [{"messages": [{"role": "user", "content": line}]} for line in lines if line]
```

**PDF to SFT format:**
```python
from pypdf import PdfReader

def pdf_to_chunks(pdf_path: str, chunk_size: int = 512) -> list:
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    
    # Chunk into training examples
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i+chunk_size]
        if len(chunk) > 100:  # Minimum chunk size
            chunks.append({"text": chunk})
    return chunks
```

---

### 4. Gradio for ML Web Interfaces

#### File Upload Interface
```python
import gradio as gr

with gr.Blocks() as demo:
    file_input = gr.File(
        label="Upload Dataset",
        file_count="single",
        file_types=[".csv", ".txt", ".pdf"],
        type="filepath"
    )
    
    output = gr.File(label="Download Trained Model")
    
    submit_btn = gr.Button("Train Model", variant="primary")
```

#### Concurrent Users (Queue Configuration)
```python
# Enable queue for handling concurrent requests
demo.queue(
    max_size=50,              # Maximum requests in queue
    default_concurrency_limit=5  # Max concurrent processing
)
demo.launch()
```

#### Progress Tracking for Long Operations
```python
def train_model(data_path, progress=gr.Progress()):
    progress(0.1, desc="Loading data...")
    # ... load data
    
    for i in range(epochs):
        progress((i + 1) / epochs, desc=f"Epoch {i+1}/{epochs}")
        # ... training
    
    progress(1.0, desc="Complete!")
    return model_path

# Progress tracking requires queue to be enabled
demo.queue()
```

#### Per-Event Concurrency Control
```python
# Different queues for different resource types
with gr.Blocks() as demo:
    btn_gpu = gr.Button("Train on GPU")
    btn_cpu = gr.Button("Process Data")
    
    # GPU tasks share queue (limited to 2 concurrent)
    btn_gpu.click(
        fn=gpu_task,
        concurrency_limit=2,
        concurrency_id="gpu_queue"
    )
    
    # CPU tasks can run more concurrently
    btn_cpu.click(
        fn=cpu_task,
        concurrency_limit=4,
        concurrency_id="cpu_queue"
    )
```

---

### 5. Docker Best Practices for ML

#### Multi-Stage Dockerfile
```dockerfile
# ===========================================
# Stage 1: Builder with CUDA devel tools
# ===========================================
FROM nvcr.io/nvidia/pytorch:26.01-py3 AS builder

WORKDIR /app
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install build dependencies
RUN pip install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

COPY pyproject.toml .
RUN pip install --no-cache-dir -e .

# ===========================================
# Stage 2: Production runtime
# ===========================================
FROM nvcr.io/nvidia/pytorch:26.01-py3 AS production

# Security: non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy virtual environment
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Copy application code
COPY --chown=appuser:appuser . .

USER appuser

# Health check for model serving
HEALTHCHECK --interval=30s --timeout=10s \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 7860
ENTRYPOINT ["./docker/entrypoint.sh"]
CMD ["python", "-m", "src.app"]
```

#### Entrypoint Script
```bash
#!/bin/bash
set -e

# Signal handling for graceful shutdown
trap 'kill -TERM $APP_PID 2>/dev/null; wait $APP_PID 2>/dev/null' SIGTERM SIGINT

# Pre-flight checks
echo "Starting Agentic AutoML Platform..."

# GPU check (optional - can run on CPU for testing)
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Status:"
    nvidia-smi --query-gpu=name,memory.free --format=csv
fi

# Initialize database (job queue)
mkdir -p /app/data
sqlite3 /app/data/jobs.db "CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    status TEXT,
    progress REAL,
    result_path TEXT,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)"

# Start application
exec python -m src.app &
APP_PID=$!
wait $APP_PID
```

#### Docker Compose for Development
```yaml
version: '3.8'

services:
  automl:
    build: .
    ports:
      - "7860:7860"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    environment:
      - LOG_LEVEL=INFO
```

---

### 6. Unsloth Integration for LLM Training

#### Basic SFT with Unsloth
```python
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig

# Load model with 4-bit quantization
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Phi-3.5-mini-instruct",
    max_seq_length=2048,
    load_in_4bit=True,  # 70% less VRAM
)

# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=16,
    lora_dropout=0,  # Optimized for Unsloth
)

# Train with SFTTrainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=SFTConfig(
        output_dir="outputs",
        num_train_epochs=3,
        per_device_train_batch_size=2,
    ),
)
trainer.train()
```

---

## Implementation Phases

### Phase 1: Project Foundation (2-3 hours)

#### Tasks
- [ ] Initialize UV project with `uv init`
- [ ] Create directory structure
- [ ] Configure `pyproject.toml` with all dependencies
- [ ] Create `.gitignore` for data/models
- [ ] Set up basic logging configuration

#### Dependencies (pyproject.toml)
```toml
[project]
name = "agentic-automl"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    # Web interface
    "gradio>=5.0",
    
    # ML pipeline orchestration
    "metaflow>=2.12",
    
    # Tabular ML
    "scikit-learn>=1.5",
    "xgboost>=2.0",
    "lightgbm>=4.3",
    "pandas>=2.0",
    "numpy>=1.24",
    
    # LLM Training
    "trl>=0.12",
    "transformers>=4.40",
    "accelerate>=0.30",
    "peft>=0.10",
    "bitsandbytes>=0.43",
    
    # Data processing
    "pypdf>=4.0",  # PDF text extraction
    
    # Utilities
    "joblib>=1.3",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "ruff>=0.3",
]

# Unsloth requires special installation
# pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

---

### Phase 2: Tabular ML Pipeline (4-5 hours)

#### Component: data_validator.py

**Responsibilities:**
1. Validate CSV file structure and encoding
2. Detect target column type (classification vs regression)
3. Check for missing values, invalid types
4. Generate descriptive error messages with actionable fixes

**Auto-Detection Logic:**
```python
def detect_task_type(y: pd.Series) -> str:
    """
    Detect if target is classification or regression.
    
    Rules:
    - If dtype is object/string → classification
    - If unique values < 10 OR unique/values ratio < 0.05 → classification
    - Otherwise → regression
    """
    if y.dtype == 'object':
        return 'classification'
    
    unique_ratio = y.nunique() / len(y)
    if y.nunique() < 10 or unique_ratio < 0.05:
        return 'classification'
    
    return 'regression'
```

**Error Message Template:**
```python
ERROR_MESSAGES = {
    "missing_target": """
The target column '{column}' contains missing values.

To fix this issue:
1. Open your CSV file
2. Either remove rows with missing target values, or
3. Fill missing values with an appropriate default

Rows affected: {count} ({percent:.1f}%)
""",
    "invalid_types": """
Column '{column}' has inconsistent data types.

Found these types: {types}

Please ensure all values in this column are the same type.
Check row {example_row} for inconsistent value: '{example_value}'
""",
}
```

#### Component: auto_ensemble.py

**Responsibilities:**
1. Create diverse ensemble based on task type
2. Train models in parallel where possible
3. Evaluate and select best ensemble configuration

**Ensemble Configurations:**

| Task Type | Ensemble | Models |
|-----------|----------|--------|
| Binary Classification | VotingClassifier (soft) | RF, XGB, LGBM |
| Multiclass Classification | VotingClassifier (soft) | RF, XGB, LGBM |
| Regression | VotingRegressor | RF, XGB, LGBM |

**Training Function:**
```python
def train_ensemble(X_train, y_train, task_type: str) -> dict:
    """Train ensemble model and return with metrics."""
    
    if task_type == 'classification':
        ensemble = VotingClassifier(
            estimators=[
                ('rf', RandomForestClassifier(n_estimators=100, n_jobs=-1)),
                ('xgb', XGBClassifier(n_estimators=100, verbosity=0)),
                ('lgbm', LGBMClassifier(n_estimators=100, verbose=-1)),
            ],
            voting='soft'
        )
    else:  # regression
        ensemble = VotingRegressor([
            ('rf', RandomForestRegressor(n_estimators=100, n_jobs=-1)),
            ('xgb', XGBRegressor()),
            ('lgbm', LGBMRegressor(verbose=-1)),
        ])
    
    ensemble.fit(X_train, y_train)
    
    return {
        'model': ensemble,
        'estimators': {name: est for name, est in ensemble.estimators_}
    }
```

#### Component: ml_training_flow.py (Metaflow)

```python
from metaflow import FlowSpec, step, Parameter

class MLTrainingFlow(FlowSpec):
    data_path = Parameter('data_path', required=True)
    target_column = Parameter('target', required=True)
    
    @step
    def start(self):
        """Initialize pipeline."""
        self.random_seed = 42
        self.next(self.load_data)
    
    @step  
    def load_data(self):
        """Load and validate CSV."""
        import pandas as pd
        self.df = pd.read_csv(self.data_path)
        
        # Validate target column exists
        if self.target_column not in self.df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found")
        
        self.next(self.preprocess)
    
    @step
    def preprocess(self):
        """Preprocess features."""
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        
        X = self.df.drop(columns=[self.target_column])
        y = self.df[self.target_column]
        
        # Simple preprocessing
        X = pd.get_dummies(X)
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_seed
        )
        
        # Detect task type
        from src.ml.data_validator import detect_task_type
        self.task_type = detect_task_type(y)
        
        self.next(self.train_model)
    
    @step
    def train_model(self):
        """Train ensemble model."""
        from src.ml.auto_ensemble import train_ensemble
        
        result = train_ensemble(self.X_train, self.y_train, self.task_type)
        self.model = result['model']
        
        self.next(self.evaluate)
    
    @step
    def evaluate(self):
        """Evaluate model performance."""
        from sklearn.metrics import accuracy_score, mean_squared_error
        
        predictions = self.model.predict(self.X_test)
        
        if self.task_type == 'classification':
            self.metric = accuracy_score(self.y_test, predictions)
            self.metric_name = 'accuracy'
        else:
            self.metric = mean_squared_error(self.y_test, predictions)
            self.metric_name = 'mse'
        
        self.next(self.serialize_model)
    
    @step
    def serialize_model(self):
        """Serialize model for download."""
        import joblib
        
        self.model_path = "model.joblib"
        joblib.dump(self.model, self.model_path)
        
        self.next(self.end)
    
    @step
    def end(self):
        """Pipeline complete."""
        print(f"Model trained. {self.metric_name}: {self.metric:.4f}")
```

---

### Phase 3: LLM Fine-tuning Pipeline (5-6 hours)

#### Component: dataset_converter.py

**Supported Conversions:**

| Input Format | SFT Output | DPO Output | GRPO Output |
|--------------|------------|------------|-------------|
| TXT (line-by-line) | `{"text": "..."}` | N/A | N/A |
| TXT (Q&A pairs) | `{"messages": [...]}` | Partial | `{"prompt": [...]}` |
| PDF (text) | `{"text": "..."}` | N/A | N/A |

**Conversion Functions:**
```python
def convert_txt_to_sft(filepath: str, format_type: str = "line_by_line") -> list:
    """Convert TXT file to SFT training format."""
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    if format_type == "line_by_line":
        # Each line is a training example
        return [
            {"text": line.strip()}
            for line in content.split('\n')
            if line.strip()
        ]
    
    elif format_type == "qa_pairs":
        # Format: Q: ... \n A: ...
        examples = []
        blocks = content.split('\n\n')
        
        for block in blocks:
            if 'Q:' in block and 'A:' in block:
                q_start = block.find('Q:') + 2
                a_start = block.find('A:') + 2
                
                question = block[q_start:block.find('A:')].strip()
                answer = block[a_start:].strip()
                
                examples.append({
                    "messages": [
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": answer}
                    ]
                })
        
        return examples

def convert_pdf_to_sft(filepath: str, chunk_size: int = 512) -> list:
    """Extract text from PDF and create training chunks."""
    
    from pypdf import PdfReader
    
    reader = PdfReader(filepath)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    
    # Create overlapping chunks
    examples = []
    for i in range(0, len(text), chunk_size // 2):
        chunk = text[i:i + chunk_size]
        if len(chunk) > 100:
            examples.append({"text": chunk})
    
    return examples
```

#### Component: trainers/sft_trainer.py

```python
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

def train_sft(
    dataset: Dataset,
    base_model: str = "unsloth/Phi-3.5-mini-instruct",
    output_dir: str = "outputs",
    epochs: int = 3,
    learning_rate: float = 2e-4,
):
    """Train SFT model with Unsloth optimizations."""
    
    # Load base model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=2048,
        load_in_4bit=True,
    )
    
    # Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=16,
        lora_dropout=0,
    )
    
    # Train
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        tokenizer=tokenizer,
        args=SFTConfig(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=learning_rate,
            logging_steps=10,
        ),
    )
    
    trainer.train()
    
    # Save LoRA adapters
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    return output_dir
```

#### Component: trainers/dpo_trainer.py

```python
from trl import DPOTrainer, DPOConfig
from unsloth import FastLanguageModel

def train_dpo(
    dataset,
    base_model: str = "unsloth/Phi-3.5-mini-instruct",
    output_dir: str = "outputs",
    beta: float = 0.1,
):
    """Train with Direct Preference Optimization."""
    
    # Load model (must be instruct-tuned for DPO)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=2048,
        load_in_4bit=True,
    )
    
    trainer = DPOTrainer(
        model=model,
        train_dataset=dataset,
        args=DPOConfig(
            output_dir=output_dir,
            beta=beta,  # KL penalty coefficient
            num_train_epochs=1,
        ),
    )
    
    trainer.train()
    model.save_pretrained(output_dir)
    
    return output_dir
```

---

### Phase 4: Job Queue System (3 hours)

#### Component: job_manager.py

```python
import sqlite3
import uuid
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

class JobStatus(Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class Job:
    id: str
    job_type: str  # "ml_training" or "llm_training"
    params: dict
    status: JobStatus
    progress: float
    result_path: str | None
    error_message: str | None
    created_at: datetime

class JobManager:
    """SQLite-backed job queue for concurrent training requests."""
    
    def __init__(self, db_path: str = "/app/data/jobs.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                job_type TEXT,
                params TEXT,
                status TEXT,
                progress REAL,
                result_path TEXT,
                error_message TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()
    
    def submit_job(self, job_type: str, params: dict) -> str:
        """Queue a training job. Returns job_id."""
        
        import json
        
        job_id = str(uuid.uuid4())
        
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO jobs (id, job_type, params, status, progress) VALUES (?, ?, ?, ?, ?)",
            [job_id, job_type, json.dumps(params), JobStatus.QUEUED.value, 0.0]
        )
        conn.commit()
        conn.close()
        
        return job_id
    
    def get_job(self, job_id: str) -> Job | None:
        """Get job details."""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            "SELECT * FROM jobs WHERE id = ?",
            [job_id]
        )
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        
        return Job(
            id=row[0],
            job_type=row[1],
            params=json.loads(row[2]),
            status=JobStatus(row[3]),
            progress=row[4],
            result_path=row[5],
            error_message=row[6],
            created_at=datetime.fromisoformat(row[7]),
        )
    
    def update_progress(self, job_id: str, progress: float):
        """Update job progress (called by training workers)."""
        
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "UPDATE jobs SET progress = ? WHERE id = ?",
            [progress, job_id]
        )
        conn.commit()
        conn.close()
    
    def complete_job(self, job_id: str, result_path: str):
        """Mark job as completed with result path."""
        
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "UPDATE jobs SET status = ?, progress = 1.0, result_path = ? WHERE id = ?",
            [JobStatus.COMPLETED.value, result_path, job_id]
        )
        conn.commit()
        conn.close()
    
    def fail_job(self, job_id: str, error_message: str):
        """Mark job as failed with error message."""
        
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "UPDATE jobs SET status = ?, error_message = ? WHERE id = ?",
            [JobStatus.FAILED.value, error_message, job_id]
        )
        conn.commit()
        conn.close()
```

---

### Phase 5: Gradio Web Interface (4-5 hours)

#### Component: app.py

```python
import gradio as gr
from src.queue.job_manager import JobManager
from src.ml.data_validator import validate_csv, detect_task_type

# Initialize job manager
job_manager = JobManager()

def create_interface():
    """Create the main Gradio interface."""
    
    with gr.Blocks(
        title="Agentic AutoML",
        theme=gr.themes.Soft()
    ) as demo:
        
        gr.Markdown("""
        # Agentic AutoML Platform
        
        Train ML models on your tabular data or fine-tune LLMs with minimal configuration.
        """)
        
        # ===========================================
        # Tab 1: Tabular ML Training
        # ===========================================
        with gr.Tab("Tabular ML"):
            
            gr.Markdown("""
            ### Train an Ensemble Model on Your CSV Data
            
            Upload a CSV file and select the target column. We'll automatically detect 
            whether your task is classification or regression, then train an ensemble
            of RandomForest, XGBoost, and LightGBM models.
            """)
            
            with gr.Row():
                csv_file = gr.File(
                    label="Upload CSV",
                    file_types=[".csv"],
                    type="filepath"
                )
            
            with gr.Row():
                target_column = gr.Dropdown(
                    label="Target Column",
                    choices=[],
                    interactive=True
                )
            
            with gr.Accordion("Advanced Options", open=False):
                model_choice = gr.Radio(
                    label="Model Selection",
                    choices=[
                        "Auto Ensemble (Recommended)",
                        "Random Forest Only",
                        "XGBoost Only",
                    ],
                    value="Auto Ensemble (Recommended)"
                )
            
            train_ml_btn = gr.Button("Train Model", variant="primary")
            
            with gr.Row():
                ml_status = gr.Textbox(label="Status", interactive=False)
                ml_progress = gr.Slider(
                    label="Progress",
                    minimum=0,
                    maximum=100,
                    interactive=False
                )
            
            ml_download = gr.File(label="Download Trained Model")
        
        # ===========================================
        # Tab 2: LLM Fine-tuning
        # ===========================================
        
        with gr.Tab("LLM Fine-tuning"):
            
            gr.Markdown("""
            ### Fine-tune an LLM on Your Custom Data
            
            Upload a TXT or PDF file. For text files, each line becomes a training example.
            For PDFs, we'll extract text and chunk it for training.
            """)
            
            with gr.Row():
                llm_file = gr.File(
                    label="Upload Dataset",
                    file_types=[".txt", ".pdf"],
                    type="filepath"
                )
            
            with gr.Row():
                dataset_type = gr.Radio(
                    label="Dataset Type",
                    choices=[
                        "SFT (Supervised Fine-Tuning)",
                        "DPO (Direct Preference Optimization)",
                        "GRPO (Online RL)",
                    ],
                    value="SFT (Supervised Fine-Tuning)"
                )
            
            with gr.Row():
                base_model = gr.Dropdown(
                    label="Base Model",
                    choices=[
                        "unsloth/Phi-3.5-mini-instruct",
                        "unsloth/Llama-3.2-1B-Instruct",
                        "unsloth/gemma-2-9b-bnb-4bit",
                    ],
                    value="unsloth/Phi-3.5-mini-instruct"
                )
            
            with gr.Accordion("Training Configuration", open=False):
                epochs = gr.Slider(1, 10, value=3, step=1, label="Epochs")
                learning_rate = gr.Slider(
                    1e-5, 1e-3, value=2e-4,
                    label="Learning Rate",
                    step=1e-5
                )
            
            train_llm_btn = gr.Button("Start Training", variant="primary")
            
            with gr.Row():
                llm_status = gr.Textbox(label="Status", interactive=False)
                llm_progress = gr.Slider(
                    label="Progress",
                    minimum=0,
                    maximum=100,
                    interactive=False
                )
            
            llm_download = gr.File(label="Download LoRA Adapter")
        
        # ===========================================
        # Tab 3: Inference Playground
        # ===========================================
        
        with gr.Tab("Inference Playground"):
            
            gr.Markdown("""
            ### Test Your Trained Models
            
            Load a model you've trained and test predictions.
            """)
            
            with gr.Row():
                model_to_load = gr.File(
                    label="Upload Trained Model",
                    type="filepath"
                )
            
            # ML Inference Section
            with gr.Accordion("ML Model Inference", open=True):
                ml_input = gr.Dataframe(
                    label="Input Features",
                    headers=["Feature 1", "Feature 2"],
                    datatype=["number", "number"],
                    row_count=1,
                )
                
                ml_predict_btn = gr.Button("Predict")
                ml_output = gr.Textbox(label="Prediction")
            
            # LLM Inference Section
            with gr.Accordion("LLM Generation", open=False):
                llm_prompt = gr.Textbox(
                    label="Prompt",
                    lines=3
                )
                
                llm_generate_btn = gr.Button("Generate")
                llm_output = gr.Textbox(
                    label="Generated Text",
                    lines=5
                )
        
        # ===========================================
        # Event Handlers
        # ===========================================
        
        def update_columns(filepath):
            """Update target column dropdown when CSV is uploaded."""
            import pandas as pd
            if filepath:
                df = pd.read_csv(filepath)
                return gr.Dropdown(choices=list(df.columns))
            return gr.Dropdown(choices=[])
        
        csv_file.change(
            fn=update_columns,
            inputs=[csv_file],
            outputs=[target_column]
        )
        
        def train_ml_model(filepath, target, progress=gr.Progress()):
            """Handle ML training request."""
            
            if not filepath or not target:
                return "Error: Please upload file and select target", 0, None
            
            try:
                progress(0.1, desc="Validating data...")
                
                # Validate CSV
                validation = validate_csv(filepath)
                if not validation['valid']:
                    return f"Error: {validation['message']}", 0, None
                
                # Submit job
                job_id = job_manager.submit_job('ml_training', {
                    'data_path': filepath,
                    'target_column': target
                })
                
                progress(0.2, desc="Training model...")
                
                # Poll for completion
                import time
                while True:
                    job = job_manager.get_job(job_id)
                    
                    if job.status.value == 'completed':
                        progress(1.0, desc="Complete!")
                        return "Training complete!", 100, job.result_path
                    
                    elif job.status.value == 'failed':
                        return f"Error: {job.error_message}", 0, None
                    
                    progress(0.2 + job.progress * 0.8)
                    time.sleep(1)
                    
            except Exception as e:
                return f"Error: {str(e)}", 0, None
        
        train_ml_btn.click(
            fn=train_ml_model,
            inputs=[csv_file, target_column],
            outputs=[ml_status, ml_progress, ml_download]
        )
        
        return demo

# Create and launch
demo = create_interface()
demo.queue(max_size=50, default_concurrency_limit=5)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
```

---

### Phase 6: Docker Containerization (2-3 hours)

#### Component: Dockerfile

```dockerfile
# ===========================================
# Agentic AutoML Platform - Production Image
# ===========================================

# Stage 1: Builder with CUDA devel tools
FROM nvcr.io/nvidia/pytorch:26.01-py3 AS builder

WORKDIR /app

# Install UV if not present
RUN which uv || pip install uv

# Create virtual environment
RUN uv venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy dependency files
COPY pyproject.toml .

# Install dependencies with UV
RUN uv pip install --system -e . && \
    uv pip install --system "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# ===========================================
# Stage 2: Production runtime
# ===========================================
FROM nvcr.io/nvidia/pytorch:26.01-py3 AS production

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# Security: non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Copy application code
COPY --chown=appuser:appuser . .

# Create directories for data and models
RUN mkdir -p /app/data /app/models && \
    chown -R appuser:appuser /app

USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s \
    CMD curl -f http://localhost:7860/ || exit 1

# Expose Gradio port
EXPOSE 7860

# Use entrypoint script
ENTRYPOINT ["./docker/entrypoint.sh"]
CMD ["python", "-m", "src.app"]
```

#### Component: entrypoint.sh

```bash
#!/bin/bash
set -e

# ===========================================
# Agentic AutoML Platform Startup Script
# ===========================================

echo "======================================"
echo "  Agentic AutoML Platform"
echo "======================================"

# Signal handling for graceful shutdown
cleanup() {
    echo ""
    echo "Received shutdown signal, cleaning up..."
    
    if [ -n "$APP_PID" ] && kill -0 "$APP_PID" 2>/dev/null; then
        kill -TERM "$APP_PID" 2>/dev/null
        
        # Wait for graceful shutdown (max 30 seconds)
        local timeout=30
        local count=0
        
        while kill -0 "$APP_PID" 2>/dev/null && [ $count -lt $timeout ]; do
            sleep 1
            count=$((count + 1))
        done
        
        # Force kill if still running
        if kill -0 "$APP_PID" 2>/dev/null; then
            echo "Force killing stubborn process..."
            kill -KILL "$APP_PID" 2>/dev/null
        fi
        
        wait "$APP_PID" 2>/dev/null
    fi
    
    echo "Cleanup complete"
    exit 0
}

# Register signal handlers
trap cleanup SIGTERM SIGINT SIGQUIT

# ===========================================
# Environment Setup
# ===========================================

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

# ===========================================
# Pre-flight Checks
# ===========================================

echo "Performing pre-flight checks..."

# Check GPU availability (optional - can run on CPU)
if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "GPU Status:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
    echo ""
else
    echo "No GPU detected - running in CPU mode (slower training)"
fi

# ===========================================
# Database Initialization
# ===========================================

echo "Initializing job queue database..."

mkdir -p /app/data
sqlite3 /app/data/jobs.db "CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    job_type TEXT,
    params TEXT,
    status TEXT DEFAULT 'queued',
    progress REAL DEFAULT 0.0,
    result_path TEXT,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)"

echo "Database initialized at /app/data/jobs.db"

# ===========================================
# Start Application
# ===========================================

echo ""
echo "Starting application..."
echo ""

# Execute the command passed to docker run
exec "$@" &
APP_PID=$!

# Wait for application
wait $APP_PID
```

---

### Phase 7: Testing & Documentation (2-3 hours)

#### Test Data

Create sample datasets for testing:

**data/sample_classification.csv:**
```csv
feature_1,feature_2,feature_3,target
1.5,2.3,0.8,A
2.1,1.9,1.2,B
0.8,3.1,0.5,A
...
```

**data/sample_sft.txt:**
```
What is machine learning? Machine learning is a subset of artificial intelligence...
How do neural networks work? Neural networks are computing systems inspired by biological neurons...
```

#### Unit Tests

```python
# tests/test_ml_pipeline.py

import pytest
from src.ml.data_validator import validate_csv, detect_task_type
import pandas as pd

def test_detect_classification():
    """Test detection of classification task."""
    
    y = pd.Series(['A', 'B', 'A', 'C', 'B'] * 100)
    assert detect_task_type(y) == 'classification'

def test_detect_regression():
    """Test detection of regression task."""
    
    y = pd.Series([1.5, 2.3, 0.8, 3.1, 4.2] * 100)
    assert detect_task_type(y) == 'regression'

def test_validate_missing_target():
    """Test validation catches missing target column."""
    
    # Create temp CSV without expected target
    ...
```

---

## Design Decisions Summary

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Task Detection** | Auto-detect from target column | Better UX, reduces user friction |
| **Concurrency Model** | SQLite job queue (5-20 users) | Scales well without Redis complexity |
| **Model Persistence** | Download-only | Stateless design, easier horizontal scaling |
| **LLM Training Methods** | SFT + DPO + GRPO | Full flexibility for users |
| **Ensemble Strategy** | Soft voting with diverse models | Best accuracy vs complexity tradeoff |
| **Base Image** | nvcr.io/nvidia/pytorch:26.01-py3 | Pre-configured CUDA + PyTorch |
| **Package Manager** | UV | Fast dependency resolution, built into image |

---

## User Flow Diagrams

### Tabular ML Training Flow
```
User                    Gradio              Job Manager          Metaflow
  │                       │                      │                    │
  ├──── Upload CSV ──────►│                      │                    │
  │                       ├── Validate file ────►│                    │
  │◄── Show columns ──────┤                      │                    │
  │                       │                      │                    │
  ├── Select target col ──►│                      │                    │
  │                       ├── Queue job ─────────►│                    │
  │◄── Job ID ────────────┤                      │                    │
  │                       │                      ├── Start flow ─────►│
  │                       │                      │                    ├─ load_data
  │                       │◄── Poll status ──────┤                    ├─ preprocess  
  │◄── Progress updates ──┤                      │                    ├─ train_model
  │                       │                      │                    ├─ evaluate
  │                       │◄── Job complete ─────┤◄── Flow done ──────┤
  │◄── Download model ────┤                      │                    │
```

### LLM Fine-tuning Flow
```
User                    Gradio              Dataset Converter     Trainer
  │                       │                      │                   │
  ├──── Upload TXT/PDF ──►│                      │                   │
  │                       ├── Convert to format ►│                   │
  │◄── Preview data ──────┤                      │                   │
  │                       │                      │                   │
  ├── Select base model ──►│                      │                  │
  ├── Configure params ───►│                      │                  │
  │                       ├── Start training ──────────────────────►│
  │◄── Progress updates ──┤                      │                   ├─ load_model
  │                       │                      │                   ├─ train_epochs
  │◄── Download LoRA ─────┤◄─────────────────────────────────────────┤
```

---

## Success Metrics

| Metric | Target |
|--------|--------|
| Training completion rate | > 95% |
| Error message clarity | User can fix without docs |
| Concurrent users supported | 5-20 |
| Average ML training time | < 5 minutes (small dataset) |
| Average LLM fine-tuning time | < 30 minutes (1K examples) |

---

## Next Steps

1. Run `/init` to initialize the project structure
2. Run `/compact` to create compact task files for each component  
3. Begin implementation following the phases above

---

## Implementation Progress (Updated Feb 2026)

### Completed Phases
| Phase | Status | Notes |
|-------|--------|-------|
| Phase 1: Project Foundation | ✅ Complete | UV project initialized, dependencies installed |
| Phase 2: Tabular ML Pipeline | ✅ Complete | data_validator.py, auto_ensemble.py, ml_training_flow.py |
| Phase 3: LLM Fine-tuning Pipeline | ✅ Code Complete | SFT/DPO/GRPO trainers written, needs GPU testing |
| Phase 4: Job Queue System | ✅ Complete | job_manager.py with SQLite backend |
| Phase 5: Gradio Web Interface | ✅ Complete | app.py (743 lines), all three tabs functional |
| Phase 6: Docker Containerization | ✅ Complete | Multi-stage Dockerfile, entrypoint.sh |
| Phase 7: Testing & Documentation | ✅ Complete | 22 tests passing (8 ML + 14 LLM) |

### Completed Tasks

1. **Gradio app integration test** ✅
   - App loads successfully with 68 blocks
   - Tabular ML tab verified with sample CSV training
   - LLM Fine-tuning tab loads (training requires GPU)
   - Inference Playground functional with model loading

2. **Created `tests/test_llm_pipeline.py`** ✅
   - 14 tests for dataset_converter functions (SFT/DPO/GRPO formats)
   - Edge case handling tested (empty files, missing data)

3. **UX Quality Review** ✅
   - Error handling uses actionable messages with fix suggestions
   - Progress indicators for training (5-step progress bar)
   - Edge cases handled gracefully (empty files, invalid formats)

### Remaining Tasks (Optional Enhancements)

1. **Test LLM training flow** (requires GPU)
   - Use `data/sample_sft.txt` for quick SFT test
   - Verify LoRA adapter saves correctly
   - Test inference with fine-tuned model

### Key Implementation Discoveries

1. **Metaflow FlowSpec Limitation**: Cannot instantiate with parameters directly - designed for CLI execution via `python flow.py run --param value`. Created `train_ml_model_direct()` in app.py as workaround for synchronous web requests.

2. **VotingClassifier.estimators_ Type**: After `fit()`, returns fitted estimators in format that may not unpack as tuples. Fixed with explicit type checking during iteration.

3. **Gradio Progress Auto-Injection**: `gr.Progress` is auto-injected at runtime - function signatures show warnings but work correctly.

4. **Gradio HTML Components**: `gr.HTML()` enables rich formatting with colored alert backgrounds for status messages.

### Sample Data Created

- `data/sample_classification.csv` - 200 rows, 3 classes (credit risk)
- `data/sample_regression.csv` - 200 rows, housing data  
- `data/sample_sft.txt` - 20 Q&A pairs for SFT testing
- `data/sample_sft_lineByLine.txt` - 30 training lines

### Trained Models Location

- `models/ensemble_classification_model.joblib`
- `models/ensemble_regression_model.joblib`