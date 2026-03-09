import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

os.environ.setdefault("HF_HOME", "/root/.cache/huggingface")
if token := os.environ.get("HF_TOKEN"):
    os.environ["HF_TOKEN"] = token


@dataclass(frozen=True)
class Settings:
    DATA_DIR: str = "data"
    MODEL_DIR: str = "models"
    JOB_DB_PATH: str = "data/jobs.db"

    HF_HOME: str = os.environ.get("HF_HOME", "/root/.cache/huggingface")
    HF_TOKEN: str = os.environ.get("HF_TOKEN", "")

    QDRANT_URL: str = os.environ.get("QDRANT_URL", "http://host.docker.internal:6333")
    QDRANT_COLLECTION: str = os.environ.get("QDRANT_COLLECTION", "automl_knowledge")

    LANGFUSE_HOST: str = os.environ.get("LANGFUSE_HOST", "http://localhost:3000")
    LANGFUSE_BASE_URL: str = os.environ.get("LANGFUSE_BASE_URL", os.environ.get("LANGFUSE_HOST", "http://localhost:3000"))
    LANGFUSE_PUBLIC_KEY: str = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
    LANGFUSE_SECRET_KEY: str = os.environ.get("LANGFUSE_SECRET_KEY", "")

    LLM_INFERENCE_URL: str = os.environ.get("LLM_INFERENCE_URL", "http://192.168.1.185:8080/v1")
    LLM_INFERENCE_KEY: str = os.environ.get("LLM_INFERENCE_KEY", "not-needed")

    DEFAULT_BASE_MODEL: str = os.environ.get("DEFAULT_BASE_MODEL", "Qwen/Qwen2.5-0.5B")
    MAX_SEQ_LENGTH: int = 2048

    MAX_REFINEMENT_ITERATIONS: int = 3
    INVESTIGATION_COLLECTION_NAME: str = "error_investigations"
    INVESTIGATION_VECTOR_SIZE: int = 2560

    def __post_init__(self):
        os.makedirs(self.DATA_DIR, exist_ok=True)
        os.makedirs(self.MODEL_DIR, exist_ok=True)


settings = Settings()
