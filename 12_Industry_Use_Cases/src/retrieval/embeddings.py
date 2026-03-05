import os
from typing import Optional
from sentence_transformers import SentenceTransformer


_embedding_model: Optional[SentenceTransformer] = None


def get_embedding_model() -> SentenceTransformer:
    """Get or create the embedding model."""
    global _embedding_model

    if _embedding_model is None:
        model_name = os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-4B")
        _embedding_model = SentenceTransformer(model_name)

    return _embedding_model


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed a list of documents."""
    model = get_embedding_model()
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings.tolist()


def embed_query(text: str) -> list[float]:
    """Embed a single query."""
    model = get_embedding_model()
    embedding = model.encode(text)
    return embedding.tolist()
