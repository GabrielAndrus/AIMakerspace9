#!/usr/bin/env python3
"""
Minimal test to verify the HybridRetriever factory method works.
Run this inside Docker with: python /workspace/test_minimal.py
"""

import sys
sys.path.insert(0, '/workspace')

def main():
    print("Testing HybridRetriever Factory Method")
    print("=" * 50)
    
    # Test 1: Import
    try:
        from src.retrieval.hybrid_retriever import HybridRetriever
        print("✓ Import successful")
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return 1
    
    # Test 2: Create via factory
    try:
        retriever = HybridRetriever.create(
            collection_name="automl_knowledge",
            vector_size=2560
        )
        print("✓ Created via factory method")
    except Exception as e:
        print(f"✗ Factory create failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Test 3: Verify structure
    assert retriever.qdrant_retriever is not None, "qdrant_retriever is None"
    print("✓ qdrant_retriever initialized")
    
    assert retriever.bm25 is not None, "bm25 is None"
    print("✓ bm25 initialized")
    
    assert not retriever.indexed, "indexed should be False initially"
    print("✓ indexed state correct")
    
    # Test 4: Check methods
    assert callable(retriever.build_index), "build_index not callable"
    print("✓ build_index method exists")
    
    assert callable(retriever.search), "search not callable"
    print("✓ search method exists")
    
    # Test 5: Check search signature
    from inspect import signature
    sig = signature(retriever.search)
    params = list(sig.parameters.keys())
    assert 'query' in params, "Missing query parameter"
    assert 'limit' in params, "Missing limit parameter"  
    assert 'method' in params, "Missing method parameter"
    print("✓ search() has correct signature")
    
    # Test 6: RAGASEvaluator initialization
    try:
        from src.evaluation.ragas_evaluator import RAGASEvaluator
        evaluator = RAGASEvaluator(collection_name="automl_knowledge", vector_size=2560)
        print("✓ RAGASEvaluator created")
    except Exception as e:
        print(f"✗ Evaluator creation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Test 7: Check evaluator attributes
    assert hasattr(evaluator, 'hybrid_retriever'), "Missing hybrid_retriever"
    print("✓ Evaluator has hybrid_retriever attribute")
    
    assert hasattr(evaluator, 'build_hybrid_index_from_collection'), \
        "Missing build_hybrid_index_from_collection"
    print("✓ Evaluator has index building method")
    
    # Test 8: Method switching
    try:
        evaluator.set_retrieval_method("dense")
        print("✓ Set method to 'dense'")
        
        evaluator.set_retrieval_method("sparse")
        assert evaluator.hybrid_retriever is not None, \
            "hybrid_retriever should be created for sparse method"
        print("✓ Set method to 'sparse' (hybrid_retriever created)")
        
        evaluator.set_retrieval_method("hybrid")
        print("✓ Set method to 'hybrid'")
        
    except Exception as e:
        print(f"✗ Method switching failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("=" * 50)
    print("All tests PASSED!")
    return 0

if __name__ == "__main__":
    sys.exit(main())