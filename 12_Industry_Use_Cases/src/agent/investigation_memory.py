"""Investigation Memory - Semantic memory for past error investigations using Qdrant.

Enables:
1. Episodic memory - Store successful resolutions for similar errors
2. Procedural memory - Improve investigation prompts based on feedback  
3. Semantic search - Find similar past investigations by meaning
"""

from uuid import uuid4
from datetime import datetime
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

from src.config import settings
from src.retrieval.embeddings import embed_query


COLLECTION_NAME = settings.INVESTIGATION_COLLECTION_NAME
VECTOR_SIZE = settings.INVESTIGATION_VECTOR_SIZE


class InvestigationMemory:
    """Semantic memory for error investigations using Qdrant."""
    
    def __init__(self, url: Optional[str] = None):
        self.url = url or settings.QDRANT_URL
        self.client = QdrantClient(url=self.url)
        self._ensure_collection()
    
    def _ensure_collection(self):
        """Create the error_investigations collection if it doesn't exist."""
        try:
            self.client.get_collection(COLLECTION_NAME)
        except Exception:
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=VECTOR_SIZE,
                    distance=Distance.COSINE
                )
            )
    
    def save_resolution(
        self,
        error_type: str,
        error_message: str,
        task_type: Optional[str],
        training_method: Optional[str],
        base_model: Optional[str],
        recommendation: str,
        successful: bool = True
    ) -> str:
        """Store an investigation result for future semantic search.
        
        Args:
            error_type: Exception class name (e.g., "RuntimeError")
            error_message: The error message text
            task_type: ml_training, llm_training, inference
            training_method: SFT, DPO, GRPO if applicable
            base_model: Model name if applicable
            recommendation: The recommendation that was given
            successful: Whether the recommendation helped resolve the issue
        
        Returns:
            ID of the stored point
        """
        text = f"{error_type} {error_message}"
        embedding = embed_query(text)
        
        point_id = str(uuid4())
        
        self.client.upsert(
            collection_name=COLLECTION_NAME,
            points=[PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "error_type": error_type,
                    "error_message": error_message[:500],
                    "task_type": task_type,
                    "training_method": training_method,
                    "base_model": base_model,
                    "recommendation": recommendation[:1000],
                    "successful": successful,
                    "timestamp": datetime.now().isoformat()
                }
            )]
        )
        
        return point_id
    
    def find_similar_errors(
        self,
        error_type: str,
        error_message: str,
        limit: int = 3,
        min_score: float = 0.7
    ) -> list[dict]:
        """Find past similar investigations via semantic search.
        
        Args:
            error_type: Exception class name
            error_message: Error message text
            limit: Max results to return
            min_score: Minimum similarity score
        
        Returns:
            List of payload dicts with past resolutions
        """
        query_text = f"{error_type} {error_message}"
        embedding = embed_query(query_text)
        
        results = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=embedding,
            limit=limit,
            score_threshold=min_score
        )
        
        return [r.payload for r in results.points if r.payload]
    
    def get_successful_resolutions(
        self,
        task_type: Optional[str] = None,
        limit: int = 10
    ) -> list[dict]:
        """Get all successful resolutions, optionally filtered by task type.
        
        Uses scroll (no vector search) to retrieve stored data.
        
        Args:
            task_type: Optional filter for task type
            limit: Max results to return
        
        Returns:
            List of payload dicts with successful resolutions
        """
        query_filter = None
        if task_type:
            query_filter = Filter(
                must=[
                    FieldCondition(key="successful", match=MatchValue(value=True)),
                    FieldCondition(key="task_type", match=MatchValue(value=task_type))
                ]
            )
        else:
            query_filter = Filter(
                must=[FieldCondition(key="successful", match=MatchValue(value=True))]
            )
        
        results = self.client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=query_filter,
            limit=limit
        )
        
        return [r.payload for r in results[0] if r.payload]
    
    def get_recent_investigations(self, limit: int = 20) -> list[dict]:
        """Get most recent investigations regardless of success.
        
        Args:
            limit: Max results to return
        
        Returns:
            List of payload dicts sorted by timestamp (newest first)
        """
        results = self.client.scroll(
            collection_name=COLLECTION_NAME,
            limit=limit
        )
        
        investigations = [r.payload for r in results[0] if r.payload]
        investigations.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return investigations
    
    def delete_collection(self) -> None:
        """Delete the entire collection (use with caution)."""
        try:
            self.client.delete_collection(collection_name=COLLECTION_NAME)
        except Exception:
            pass


_investigation_memory: Optional[InvestigationMemory] = None


def get_investigation_memory() -> InvestigationMemory:
    """Get or create the singleton investigation memory instance."""
    global _investigation_memory
    if _investigation_memory is None:
        _investigation_memory = InvestigationMemory()
    return _investigation_memory


def save_successful_resolution(error_context: dict, recommendation: str) -> str:
    """Convenience function to save a successful resolution."""
    mem = get_investigation_memory()
    return mem.save_resolution(
        error_type=error_context.get("error_type", ""),
        error_message=error_context.get("error_message", ""),
        task_type=error_context.get("task_type"),
        training_method=error_context.get("training_method"),
        base_model=error_context.get("base_model"),
        recommendation=recommendation,
        successful=True
    )


def find_similar_past_errors(error_context: dict) -> list[dict]:
    """Convenience function to find similar past errors."""
    mem = get_investigation_memory()
    return mem.find_similar_errors(
        error_type=error_context.get("error_type", ""),
        error_message=error_context.get("error_message", "")
    )