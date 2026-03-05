#!/usr/bin/env python3
"""Test script for HybridRetriever factory method."""

import sys
sys.path.insert(0, '/home/imjonezz/Desktop/AIMakerspace9/12_Industry_Use_Cases')

print("Testing HybridRetriever factory method...")
print("=" * 50)

try:
    from src.retrieval.qdrant_client import QdrantRetriever
    print("✓ QdrantRetriever imported successfully")
except Exception as e:
    print(f"✗ Failed to import QdrantRetriever: {e}")
    sys.exit(1)

try:
    from src.retrieval.hybrid_retriever import HybridRetriever
    print("✓ HybridRetriever imported successfully")
except Exception as e:
    print(f"✗ Failed to import HybridRetriever: {e}")
    sys.exit(1)

# Test factory method
try:
    print("\nTesting HybridRetriever.create()...")
    retriever = HybridRetriever.create(
        collection_name="automl_knowledge",
        vector_size=2560
    )
    print("✓ HybridRetriever created successfully via factory method")
except Exception as e:
    print(f"✗ Failed to create HybridRetriever: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test attributes
print("\nChecking retriever structure...")
assert hasattr(retriever, 'qdrant_retriever'), "Missing qdrant_retriever attribute"
print("✓ Has qdrant_retriever")

assert hasattr(retriever, 'bm25'), "Missing bm25 attribute"
print("✓ Has bm25")

assert hasattr(retriever, 'rrf_k'), "Missing rrf_k attribute"
print("✓ Has rrf_k")

assert hasattr(retriever, 'indexed'), "Missing indexed attribute"
print("✓ Has indexed")

assert hasattr(retriever, 'documents'), "Missing documents attribute"
print("✓ Has documents")

assert hasattr(retriever, 'build_index'), "Missing build_index method"
print("✓ Has build_index method")

assert hasattr(retriever, 'search'), "Missing search method"
print("✓ Has search method")

# Test that search methods exist
from inspect import signature
search_sig = signature(retriever.search)
params = list(search_sig.parameters.keys())
assert 'query' in params, "search() missing query parameter"
assert 'limit' in params, "search() missing limit parameter"
assert 'method' in params, "search() missing method parameter"
print("✓ search() has correct parameters (query, limit, method)")

# Test that qdrant_retriever is properly initialized
print("\nChecking QdrantRetriever initialization...")
assert retriever.qdrant_retriever is not None, "qdrant_retriever is None"
print("✓ qdrant_retriever is initialized")

assert retriever.qdrant_retriever.collection_name == "automl_knowledge", \
    f"Unexpected collection_name: {retriever.qdrant_retriever.collection_name}"
print("✓ Collection name is correct")

print("\n" + "=" * 50)
print("All tests passed! ✓")
print("=" * 50)

# Show usage example
print("\nUsage Example:")
print("-" * 50)
print("""
from src.retrieval.hybrid_retriever import HybridRetriever

# Create via factory method
retriever = HybridRetriever.create(
    collection_name="automl_knowledge",
    vector_size=2560
)

# Build index (requires documents)
documents = [...]
retriever.build_index(documents)

# Search with different methods
results_dense = retriever.search("query", limit=5, method="dense")
results_sparse = retriever.search("query", limit=5, method="sparse")
results_hybrid = retriever.search("query", limit=5, method="hybrid")
""")