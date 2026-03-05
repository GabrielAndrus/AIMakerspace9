import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, Filter, FieldCondition, MatchValue


class QdrantRetriever:
    """Retriever for Qdrant vector database."""

    def __init__(self, collection_name: str = None, url: str = None, vector_size: int = None):
        self.collection_name = collection_name or os.getenv("QDRANT_COLLECTION", "automl_knowledge")
        self.url = url or os.getenv("QDRANT_URL", "http://host.docker.internal:6333")
        self.vector_size = vector_size
        self.client = QdrantClient(url=self.url)
        self.use_hybrid = False
        self.hybrid_retriever = None

    def create_collection(self, recreate: bool = False) -> None:
        """Create the collection if it doesn't exist."""
        collections = self.client.get_collections().collections
        collection_exists = any(c.name == self.collection_name for c in collections)

        if recreate and collection_exists:
            self.client.delete_collection(collection_name=self.collection_name)
            collection_exists = False

        if not collection_exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
            )

    def add_documents(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        payloads: list[dict[str, Any]],
        batch_size: int = 100,
    ) -> None:
        """Add documents to the collection in batches to avoid payload size limits."""
        total = len(ids)
        for i in range(0, total, batch_size):
            batch_ids = ids[i : i + batch_size]
            batch_embeddings = embeddings[i : i + batch_size]
            batch_payloads = payloads[i : i + batch_size]

            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    {"id": abs(hash(idx)) % (2**63), "vector": vec, "payload": payload}
                    for idx, vec, payload in zip(batch_ids, batch_embeddings, batch_payloads)
                ],
            )

    def enable_hybrid_retrieval(self) -> None:
        """Enable hybrid retrieval (dense + sparse)."""
        self.use_hybrid = True
        if not self.hybrid_retriever:
            from src.retrieval.hybrid_retriever import HybridRetriever
            self.hybrid_retriever = HybridRetriever(self)

    def build_hybrid_index(self, documents: list[dict[str, Any]]) -> None:
        """Build hybrid index for combined dense and sparse search."""
        if not self.use_hybrid or not self.hybrid_retriever:
            raise ValueError("Hybrid retrieval not enabled. Call enable_hybrid_retrieval() first.")
        self.hybrid_retriever.build_index(documents)

    def set_retrieval_method(self, method: str) -> None:
        """Set retrieval method: 'dense', 'sparse', or 'hybrid'."""
        if not self.use_hybrid:
            raise ValueError("Hybrid retrieval not enabled. Call enable_hybrid_retrieval() first.")
        self.retrieval_method = method

    def search(
        self, query_embedding: list[float], limit: int = 5, source_filter: str = None, method: str = "dense"
    ) -> list[dict[str, Any]]:
        """Search for similar documents."""
        if self.use_hybrid and method != "dense":
            from src.retrieval.embeddings import embed_query
            query_text = ""
            for payload in self.client.scroll(collection_name=self.collection_name, limit=1)[0]:
                if "content" in payload.payload:
                    query_text = payload.payload["content"]
                    break

            results = self.hybrid_retriever.search(query_text, limit=limit, method=method)
            return results

        search_params = {"limit": limit}

        if source_filter:
            search_params["query_filter"] = Filter(
                must=[FieldCondition(key="source", match=MatchValue(value=source_filter))]
            )

        results = self.client.query_points(
            collection_name=self.collection_name, query=query_embedding, **search_params
        )

        return [{"id": hit.id, "score": hit.score, "payload": hit.payload} for hit in results.points]
