# HybridRetriever Fix Summary

## Changes Made

### 1. Added Factory Method to `HybridRetriever`
**File:** `src/retrieval/hybrid_retriever.py`

Added a static factory method that creates a complete `HybridRetriever` instance:
```python
@staticmethod
def create(collection_name: str = None, url: str = None, vector_size: int = 2560, k1: float = 1.5, b: float = 0.75, rrf_k: int = 60):
    """Factory method to create a HybridRetriever with its own QdrantRetriever."""
    from .qdrant_client import QdrantRetriever
    qdrant_retriever = QdrantRetriever(collection_name=collection_name, url=url, vector_size=vector_size)
    return HybridRetriever(qdrant_retriever, k1=k1, b=b, rrf_k=rrf_k)
```

### 2. Updated `RAGASEvaluator` Class
**File:** `src/evaluation/ragas_evaluator.py`

#### Added hybrid_retriever attribute:
```python
def __init__(self, collection_name: str = None, vector_size: int = 2560):
    self.retriever = QdrantRetriever(collection_name=collection_name, vector_size=vector_size)
    self.hybrid_retriever = None  # NEW
    self.retrieval_method = "dense"
```

#### Enhanced `set_retrieval_method()`:
Now automatically creates hybrid retriever when switching to sparse or hybrid methods:
```python
def set_retrieval_method(self, method: str) -> None:
    self.retrieval_method = method
    
    if method in ["sparse", "hybrid"] and not self.hybrid_retriever:
        from src.retrieval.hybrid_retriever import HybridRetriever
        self.hybrid_retriever = HybridRetriever.create(
            collection_name=self.retriever.collection_name,
            url=self.retriever.url,
            vector_size=self.retriever.vector_size
        )
```

#### Added `build_hybrid_index_from_collection()`:
Loads documents from the existing Qdrant collection and builds the hybrid index:
```python
def build_hybrid_index_from_collection(self) -> None:
    """Build hybrid index from existing Qdrant collection."""
    if not self.hybrid_retriever:
        raise ValueError("Hybrid retriever not initialized.")
    
    documents = self._load_documents_from_collection()
    if not documents:
        raise ValueError("No documents found in collection.")
    
    print(f"Building hybrid index from {len(documents)} documents...")
    self.hybrid_retriever.build_index(documents)
```

#### Added `_load_documents_from_collection()` helper:
```python
def _load_documents_from_collection(self) -> list[dict[str, Any]]:
    """Load all documents from Qdrant collection."""
    points, _ = self.retriever.client.scroll(
        collection_name=self.retriever.collection_name,
        limit=10000,
        with_payload=True
    )
    
    documents = []
    for point in points:
        payload = point.payload or {}
        if "content" not in payload:
            continue
        documents.append({
            "id": str(point.id),
            "title": payload.get("title", ""),
            "content": payload["content"],
            "source": payload.get("source", ""),
        })
    
    return documents
```

#### Updated `retrieve_context()`:
Now properly handles all three retrieval methods:
```python
def retrieve_context(self, question: str, limit: int = 5) -> list[str]:
    if self.retrieval_method == "dense":
        query_embedding = embed_query(question)
        results = self.retriever.search(query_embedding, limit=limit, method="dense")
    elif self.retrieval_method in ["sparse", "hybrid"]:
        if not self.hybrid_retriever or not self.hybrid_retriever.indexed:
            raise ValueError("Hybrid retriever not initialized. Index must be built first.")
        results = self.hybrid_retriever.search(question, limit=limit, method=self.retrieval_method)
    else:
        raise ValueError(f"Unknown retrieval method: {self.retrieval_method}")
    
    return [r["payload"]["content"] for r in results]
```

#### Updated `compare_retrieval_methods()`:
Now supports all three methods (dense, sparse, hybrid) and builds the hybrid index automatically:
```python
def compare_retrieval_methods(self, dataset_path: str = None) -> dict[str, Any]:
    methods = ["dense", "sparse", "hybrid"]
    
    for method in methods:
        print(f"\nEvaluating with {method} retrieval...")
        self.set_retrieval_method(method)
        
        if method in ["sparse", "hybrid"]:
            self.build_hybrid_index_from_collection()
        
        results = self.evaluate_dataset(dataset_path)
        comparison_results[method] = results
```

#### Updated `main()` CLI:
- Now shows all three methods in comparison output
- Builds hybrid index when needed for sparse/hybrid evaluation

## Usage Examples

### Creating a Hybrid Retriever Directly:
```python
from src.retrieval.hybrid_retriever import HybridRetriever

# Create via factory method
retriever = HybridRetriever.create(
    collection_name="automl_knowledge",
    vector_size=2560
)

# Build index from documents
documents = [{"id": "1", "content": "...", ...}]
retriever.build_index(documents)

# Search with different methods
dense_results = retriever.search("query", limit=5, method="dense")
sparse_results = retriever.search("query", limit=5, method="sparse")
hybrid_results = retriever.search("query", limit=5, method="hybrid")
```

### Using with RAGASEvaluator:
```python
from src.evaluation.ragas_evaluator import RAGASEvaluator

evaluator = RAGASEvaluator(collection_name="automl_knowledge", vector_size=2560)

# Evaluate with specific method
evaluator.set_retrieval_method("hybrid")
evaluator.build_hybrid_index_from_collection()
results = evaluator.evaluate_dataset("data/evaluation/test_questions.jsonl")

# Or compare all methods
comparison = evaluator.compare_retrieval_methods()
```

### Running from CLI:

```bash
# Evaluate with specific retrieval method
python src/evaluation/ragas_evaluator.py --method hybrid

# Compare all three methods
python src/evaluation/ragas_evaluator.py --compare
```

## Testing

Run the structure verification test:
```bash
python3 test_structure.py
```

For full functional testing (requires Docker environment with all dependencies):
```bash
# Start the container and run tests
docker exec -it <container> python /workspace/test_hybrid_fix.py
```

## Key Benefits

1. **Easy Initialization**: The factory method makes it simple to create a HybridRetriever without needing to instantiate QdrantRetriever separately
2. **Automatic Setup**: The evaluator automatically creates and configures the hybrid retriever when needed
3. **Index Loading**: Documents are loaded directly from the existing Qdrant collection - no need to manually load documents
4. **Method Switching**: Easily switch between dense, sparse, and hybrid retrieval methods
5. **Complete Comparison**: The evaluator now supports comparing all three retrieval methods side-by-side