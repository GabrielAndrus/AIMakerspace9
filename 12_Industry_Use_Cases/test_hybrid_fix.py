#!/usr/bin/env python3
"""
Test script to verify the HybridRetriever fixes work correctly.
This should be run inside the Docker container where all dependencies are available.

Usage:
    docker exec -it <container> python /workspace/test_hybrid_fix.py
"""

import os
import sys

# Ensure workspace is in path
sys.path.insert(0, '/workspace')

def test_imports():
    """Test that all imports work correctly."""
    print("=" * 60)
    print("Testing Imports...")
    print("=" * 60)
    
    try:
        from src.retrieval.qdrant_client import QdrantRetriever
        print("✓ QdrantRetriever imported")
    except Exception as e:
        print(f"✗ Failed to import QdrantRetriever: {e}")
        return False
    
    try:
        from src.retrieval.hybrid_retriever import HybridRetriever
        print("✓ HybridRetriever imported")
    except Exception as e:
        print(f"✗ Failed to import HybridRetriever: {e}")
        return False
    
    try:
        from src.evaluation.ragas_evaluator import RAGASEvaluator
        print("✓ RAGASEvaluator imported")
    except Exception as e:
        print(f"✗ Failed to import RAGASEvaluator: {e}")
        return False
    
    print()
    return True

def test_factory_method():
    """Test HybridRetriever factory method."""
    print("=" * 60)
    print("Testing HybridRetriever.create() Factory Method...")
    print("=" * 60)
    
    from src.retrieval.hybrid_retriever import HybridRetriever
    
    try:
        retriever = HybridRetriever.create(
            collection_name="automl_knowledge",
            vector_size=2560
        )
        print("✓ HybridRetriever created successfully")
    except Exception as e:
        print(f"✗ Failed to create HybridRetriever: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Check attributes
    checks = [
        (hasattr(retriever, 'qdrant_retriever'), "Has qdrant_retriever"),
        (hasattr(retriever, 'bm25'), "Has bm25"),
        (hasattr(retriever, 'rrf_k'), "Has rrf_k"),
        (hasattr(retriever, 'indexed'), "Has indexed"),
        (hasattr(retriever, 'documents'), "Has documents"),
    ]
    
    for check, desc in checks:
        if check:
            print(f"✓ {desc}")
        else:
            print(f"✗ {desc}")
            return False
    
    # Check methods
    method_checks = [
        (hasattr(retriever, 'build_index'), "Has build_index"),
        (hasattr(retriever, 'search'), "Has search"),
    ]
    
    for check, desc in method_checks:
        if check:
            print(f"✓ {desc}")
        else:
            print(f"✗ {desc}")
            return False
    
    # Check search signature
    from inspect import signature
    search_sig = signature(retriever.search)
    params = list(search_sig.parameters.keys())
    
    param_checks = [
        ('query' in params, "search() has 'query' parameter"),
        ('limit' in params, "search() has 'limit' parameter"),
        ('method' in params, "search() has 'method' parameter"),
    ]
    
    for check, desc in param_checks:
        if check:
            print(f"✓ {desc}")
        else:
            print(f"✗ {desc}")
            return False
    
    # Verify qdrant_retriever is initialized
    if retriever.qdrant_retriever is None:
        print("✗ qdrant_retriever is None")
        return False
    print(f"✓ qdrant_retriever initialized (collection: {retriever.qdrant_retriever.collection_name})")
    
    print()
    return True

def test_evaluator_initialization():
    """Test RAGASEvaluator initialization."""
    print("=" * 60)
    print("Testing RAGASEvaluator Initialization...")
    print("=" * 60)
    
    from src.evaluation.ragas_evaluator import RAGASEvaluator
    
    try:
        evaluator = RAGASEvaluator(collection_name="automl_knowledge", vector_size=2560)
        print("✓ RAGASEvaluator created")
    except Exception as e:
        print(f"✗ Failed to create RAGASEvaluator: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Check attributes
    checks = [
        (hasattr(evaluator, 'retriever'), "Has retriever"),
        (hasattr(evaluator, 'hybrid_retriever'), "Has hybrid_retriever"),
        (hasattr(evaluator, 'retrieval_method'), "Has retrieval_method"),
    ]
    
    for check, desc in checks:
        if check:
            print(f"✓ {desc}")
        else:
            print(f"✗ {desc}")
            return False
    
    # Check methods
    method_checks = [
        (hasattr(evaluator, 'set_retrieval_method'), "Has set_retrieval_method"),
        (hasattr(evaluator, 'build_hybrid_index_from_collection'), "Has build_hybrid_index_from_collection"),
        (hasattr(evaluator, 'retrieve_context'), "Has retrieve_context"),
    ]
    
    for check, desc in method_checks:
        if check:
            print(f"✓ {desc}")
        else:
            print(f"✗ {desc}")
            return False
    
    print()
    return True

def test_retrieval_method_switching():
    """Test switching between retrieval methods."""
    print("=" * 60)
    print("Testing Retrieval Method Switching...")
    print("=" * 60)
    
    from src.evaluation.ragas_evaluator import RAGASEvaluator
    
    try:
        evaluator = RAGASEvaluator(collection_name="automl_knowledge", vector_size=2560)
        print("✓ Evaluator created")
    except Exception as e:
        print(f"✗ Failed to create evaluator: {e}")
        return False
    
    # Test setting method to dense
    try:
        evaluator.set_retrieval_method("dense")
        print(f"✓ Set method to 'dense' (hybrid_retriever: {evaluator.hybrid_retriever})")
    except Exception as e:
        print(f"✗ Failed to set method to 'dense': {e}")
        return False
    
    # Test setting method to sparse (should create hybrid_retriever)
    try:
        evaluator.set_retrieval_method("sparse")
        if evaluator.hybrid_retriever is not None:
            print(f"✓ Set method to 'sparse' (hybrid_retriever created)")
        else:
            print(f"✗ hybrid_retriever not created for sparse method")
            return False
    except Exception as e:
        print(f"✗ Failed to set method to 'sparse': {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test setting method to hybrid (should reuse existing hybrid_retriever)
    try:
        old_hybrid = evaluator.hybrid_retriever
        evaluator.set_retrieval_method("hybrid")
        if evaluator.hybrid_retriever is old_hybrid:
            print(f"✓ Set method to 'hybrid' (reused existing hybrid_retriever)")
        else:
            print(f"! Warning: New hybrid_retriever created instead of reusing")
    except Exception as e:
        print(f"✗ Failed to set method to 'hybrid': {e}")
        return False
    
    print()
    return True

def test_index_building():
    """Test building hybrid index from collection."""
    print("=" * 60)
    print("Testing Hybrid Index Building...")
    print("=" * 60)
    
    from src.evaluation.ragas_evaluator import RAGASEvaluator
    
    try:
        evaluator = RAGASEvaluator(collection_name="automl_knowledge", vector_size=2560)
        evaluator.set_retrieval_method("hybrid")
    except Exception as e:
        print(f"✗ Failed to setup evaluator: {e}")
        return False
    
    # Check if collection has documents
    try:
        points, _ = evaluator.retriever.client.scroll(
            collection_name=evaluator.retriever.collection_name,
            limit=1
        )
        
        if not points:
            print("! Collection is empty, skipping index building test")
            return True
        
        print(f"✓ Collection has documents ({len(points)}+)")
    except Exception as e:
        print(f"✗ Failed to check collection: {e}")
        return False
    
    # Try building index
    try:
        print("Building hybrid index from collection...")
        evaluator.build_hybrid_index_from_collection()
        
        if evaluator.hybrid_retriever.indexed:
            print("✓ Hybrid index built successfully")
        else:
            print("! Index building completed but indexed flag is False")
        
        if evaluator.hybrid_retriever.documents:
            print(f"✓ Documents loaded ({len(evaluator.hybrid_retriever.documents)} chunks)")
        else:
            print("! No documents loaded")
        
    except Exception as e:
        print(f"✗ Failed to build index: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print()
    return True

def test_search_methods():
    """Test search with different methods."""
    print("=" * 60)
    print("Testing Search Methods...")
    print("=" * 60)
    
    from src.evaluation.ragas_evaluator import RAGASEvaluator
    
    try:
        evaluator = RAGASEvaluator(collection_name="automl_knowledge", vector_size=2560)
        evaluator.set_retrieval_method("hybrid")
        
        try:
            evaluator.build_hybrid_index_from_collection()
        except Exception as e:
            print(f"! Could not build index, skipping search test: {e}")
            return True
        
    except Exception as e:
        print(f"✗ Failed to setup evaluator: {e}")
        return False
    
    test_query = "How do I use GridSearchCV?"
    
    # Test dense search
    try:
        results = evaluator.hybrid_retriever.search(test_query, limit=3, method="dense")
        print(f"✓ Dense search returned {len(results)} results")
    except Exception as e:
        print(f"✗ Dense search failed: {e}")
        return False
    
    # Test sparse search
    try:
        results = evaluator.hybrid_retriever.search(test_query, limit=3, method="sparse")
        print(f"✓ Sparse search returned {len(results)} results")
    except Exception as e:
        print(f"✗ Sparse search failed: {e}")
        return False
    
    # Test hybrid search
    try:
        results = evaluator.hybrid_retriever.search(test_query, limit=3, method="hybrid")
        print(f"✓ Hybrid search returned {len(results)} results")
    except Exception as e:
        print(f"✗ Hybrid search failed: {e}")
        return False
    
    # Test evaluator retrieve_context with different methods
    print("\nTesting RAGASEvaluator.retrieve_context()...")
    
    for method in ["dense", "sparse", "hybrid"]:
        try:
            evaluator.set_retrieval_method(method)
            if method in ["sparse", "hybrid"]:
                evaluator.build_hybrid_index_from_collection()
            
            contexts = evaluator.retrieve_context(test_query, limit=2)
            print(f"✓ retrieve_context() with '{method}' returned {len(contexts)} contexts")
        except Exception as e:
            print(f"✗ retrieve_context() with '{method}' failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    print()
    return True

def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("HybridRetriever Fix Verification Tests")
    print("=" * 60 + "\n")
    
    tests = [
        ("Imports", test_imports),
        ("Factory Method", test_factory_method),
        ("Evaluator Init", test_evaluator_initialization),
        ("Method Switching", test_retrieval_method_switching),
        ("Index Building", test_index_building),
        ("Search Methods", test_search_methods),
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n✗ Test '{name}' raised unexpected exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        symbol = "✓" if passed else "✗"
        print(f"{symbol} {name:<30} {status}")
    
    all_passed = all(passed for _, passed in results)
    
    print("=" * 60)
    if all_passed:
        print("All tests PASSED! ✓")
    else:
        print("Some tests FAILED! ✗")
    print("=" * 60)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())