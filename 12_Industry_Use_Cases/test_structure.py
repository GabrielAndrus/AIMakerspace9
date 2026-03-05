#!/usr/bin/env python3
"""
Quick test to verify code structure (runs outside Docker).
"""

import sys
sys.path.insert(0, '/home/imjonezz/Desktop/AIMakerspace9/12_Industry_Use_Cases')

def check_code_structure():
    """Verify code structure without importing."""
    
    print("=" * 60)
    print("Checking Code Structure")
    print("=" * 60 + "\n")
    
    # Check HybridRetriever
    with open('src/retrieval/hybrid_retriever.py', 'r') as f:
        hybrid_content = f.read()
    
    print("HybridRetriever:")
    checks = [
        ('@staticmethod' in hybrid_content, "Has @staticmethod decorator"),
        ('def create(' in hybrid_content, "Has create() method"),
        ('from src.retrieval.qdrant_client import QdrantRetriever' in hybrid_content, 
         "create() imports QdrantRetriever"),
        ('return HybridRetriever(' in hybrid_content, 
         "create() returns HybridRetriever instance"),
        ('collection_name' in hybrid_content and 'vector_size' in hybrid_content,
         "create() accepts collection_name and vector_size"),
    ]
    
    for check, desc in checks:
        print(f"  {'✓' if check else '✗'} {desc}")
    
    # Check RAGASEvaluator
    with open('src/evaluation/ragas_evaluator.py', 'r') as f:
        eval_content = f.read()
    
    print("\nRAGASEvaluator:")
    checks = [
        ('self.hybrid_retriever' in eval_content, "Has hybrid_retriever attribute"),
        ('HybridRetriever.create(' in eval_content, "Uses factory method"),
        ('build_hybrid_index_from_collection' in eval_content, 
         "Has build_hybrid_index_from_collection method"),
        ('_load_documents_from_collection' in eval_content,
         "Has _load_documents_from_collection helper"),
        ("methods = [\"dense\", \"sparse\", \"hybrid\"]" in eval_content,
         "compare_retrieval_methods supports all 3 methods"),
        ('sparse_scores' in eval_content, 
         "_create_comparison_summary includes sparse scores"),
    ]
    
    for check, desc in checks:
        print(f"  {'✓' if check else '✗'} {desc}")
    
    # Check set_retrieval_method
    print("\nset_retrieval_method logic:")
    
    # Extract the method to check its logic
    import re
    set_method_match = re.search(
        r'def set_retrieval_method\(self.*?\n(?=    def |\Z)', 
        eval_content, 
        re.DOTALL
    )
    
    if set_method_match:
        method_code = set_method_match.group(0)
        checks = [
            ('if method in ["sparse", "hybrid"]' in method_code,
             "Creates hybrid_retriever for sparse/hybrid"),
            ('not self.hybrid_retriever' in method_code,
             "Only creates if not already created"),
            ('HybridRetriever.create(' in method_code,
             "Uses factory method"),
        ]
        
        for check, desc in checks:
            print(f"  {'✓' if check else '✗'} {desc}")
    
    # Check retrieve_context
    print("\nretrieve_context logic:")
    
    retrieve_match = re.search(
        r'def retrieve_context\(self.*?\n(?=    def |\Z)', 
        eval_content, 
        re.DOTALL
    )
    
    if retrieve_match:
        method_code = retrieve_match.group(0)
        checks = [
            ('if self.retrieval_method == "dense":' in method_code,
             "Handles dense retrieval"),
            ('elif self.retrieval_method in ["sparse", "hybrid"]:' in method_code,
             "Handles sparse/hybrid retrieval"),
            ('self.hybrid_retriever.search(' in method_code,
             "Uses hybrid_retriever for sparse/hybrid"),
        ]
        
        for check, desc in checks:
            print(f"  {'✓' if check else '✗'} {desc}")
    
    # Check main function
    print("\nCLI main() logic:")
    
    main_match = re.search(r'if args\.compare:(?=\n|$)', eval_content, re.DOTALL)
    if main_match:
        compare_code = eval_content[eval_content.find('if args.compare:'):]
        
        checks = [
            ('Dense vs Sparse vs Hybrid' in compare_code,
             "Updated comparison message"),
            ("'dense', 'sparse', 'hybrid'" in compare_code or 
             "['dense', 'sparse', 'hybrid']" in compare_code,
             "Compares all 3 methods"),
            ('for method in ["dense", "sparse", "hybrid"]' in compare_code,
             "Iterates over all 3 methods"),
            ('build_hybrid_index_from_collection' in compare_code,
             "Builds index for sparse/hybrid"),
        ]
        
        for check, desc in checks:
            print(f"  {'✓' if check else '✗'} {desc}")
    
    print("\n" + "=" * 60)
    print("Structure check complete!")
    print("=" * 60)

if __name__ == "__main__":
    check_code_structure()