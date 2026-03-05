#!/usr/bin/env python3
"""Test script to verify LangFuse tracing is working with the DatasetAnalyzer."""

from pathlib import Path

print("=" * 60)
print("LangFuse Integration Test for DatasetAnalyzer")
print("=" * 60)

try:
    from langfuse import get_client, Langfuse
    from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler
    
    from src.config import settings
    
    print(f"✓ Settings loaded:")
    print(f"  LANGFUSE_HOST: {settings.LANGFUSE_HOST}")
    print(f"  PUBLIC_KEY: {'*' * 10}{settings.LANGFUSE_PUBLIC_KEY[-6:]}")
    
    Langfuse(
        public_key=settings.LANGFUSE_PUBLIC_KEY,
        secret_key=settings.LANGFUSE_SECRET_KEY,
        host=settings.LANGFUSE_HOST,
    )
    
    client = get_client()
    print(f"✓ LangFuse client initialized")
    print(f"  Auth check: {client.auth_check()}")
    
    handler = LangfuseCallbackHandler()
    print(f"✓ LangChain callback handler created")
    print()
    
    test_file = Path("data/test_sample.jsonl")
    if not test_file.exists():
        import json
        test_data = [
            {"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there!"}]},
            {"messages": [{"role": "user", "content": "How are you?"}, {"role": "assistant", "content": "I'm doing well, thanks!"}]},
        ]
        test_file.parent.mkdir(parents=True, exist_ok=True)
        with open(test_file, 'w') as f:
            for item in test_data:
                f.write(json.dumps(item) + '\n')
        print(f"✓ Created test file: {test_file}")
    
    from src.agent.dataset_analyzer import DatasetAnalyzer
    
    analyzer = DatasetAnalyzer()
    if analyzer.langfuse_handler:
        print("✓ LangFuse handler attached to analyzer")
    else:
        print("⚠ No LangFuse handler (check keys)")
    
    analysis = analyzer.analyze(str(test_file))
    print(f"✓ Analysis complete:")
    print(f"  Recommended method: {analysis['recommended_method']}")
    print(f"  Detected format: {analysis['detected_format']}")
    
    client.flush()
    print()
    print("=" * 60)
    print("SUCCESS! Check LangFuse at http://localhost:3000")
    print("You should see traces from the dataset analysis.")
    print("=" * 60)

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()