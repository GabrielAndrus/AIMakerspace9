#!/usr/bin/env python3
"""
Standalone test script for Activity #1 - RAGAS testset generation and retriever evaluation.
Run this to verify the notebook code works before running in marimo.
"""

import os

os.environ.setdefault("OPENAI_API_KEY", "dummy-key")
os.environ.setdefault("LLM_BASE_URL", "http://192.168.1.79:8080/v1")
os.environ.setdefault("LLM_MODEL", "minimax-m2.5-mlx@4bit")
os.environ.setdefault("EMBEDDING_MODEL", "text-embedding-qwen3-embedding-4b")
os.environ.setdefault("EMBEDDING_BASE_URL", "http://192.168.1.79:8080/v1")

print("=" * 60)
print("STEP 1: Loading documents")
print("=" * 60)

from langchain_community.document_loaders import TextLoader

loader = TextLoader("data/HealthWellnessGuide.txt")
docs = loader.load()
print(f"Loaded {len(docs)} documents")

print("\n" + "=" * 60)
print("STEP 2: Setting up generator LLM and embeddings")
print("=" * 60)

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

generator_llm = LangchainLLMWrapper(
    ChatOpenAI(
        model="minimax-m2.5-mlx@4bit",
        base_url="http://192.168.1.79:8080/v1",
    )
)
generator_embeddings = LangchainEmbeddingsWrapper(
    OpenAIEmbeddings(
        model="text-embedding-qwen3-embedding-4b",
        base_url="http://192.168.1.79:8080/v1",
        check_embedding_ctx_length=False,
    )
)
print("Generator LLM and embeddings ready")

print("\n" + "=" * 60)
print("STEP 3: Building Knowledge Graph")
print("=" * 60)

from ragas.testset.graph import KnowledgeGraph, Node, NodeType

kg = KnowledgeGraph()

for doc in docs:
    kg.nodes.append(
        Node(
            type=NodeType.DOCUMENT,
            properties={
                "page_content": doc.page_content,
                "document_metadata": doc.metadata,
            },
        )
    )

print(f"Added {len(kg.nodes)} document nodes to knowledge graph")

print("\n" + "=" * 60)
print("STEP 4: Applying transforms (SLOW - ~1-2 hours)")
print("=" * 60)

from ragas.testset.transforms import default_transforms, apply_transforms

print("Applying transforms to build knowledge graph relationships...")
default_transforms_list = default_transforms(
    documents=docs, llm=generator_llm, embedding_model=generator_embeddings
)
apply_transforms(kg, default_transforms_list)

print(
    f"Knowledge graph built: {len(kg.nodes)} nodes, {len(kg.relationships)} relationships"
)

print("\n" + "=" * 60)
print("STEP 5: Creating testset generator")
print("=" * 60)

from ragas.testset import TestsetGenerator

generator = TestsetGenerator(
    llm=generator_llm,
    embedding_model=generator_embeddings,
    knowledge_graph=kg,
)
print("Testset generator ready")

print("\n" + "=" * 60)
print("STEP 6: Generating testset (SLOW - ~2 hours)")
print("=" * 60)

from ragas.testset.synthesizers.single_hop.specific import (
    SingleHopSpecificQuerySynthesizer,
)

query_distribution = [
    (SingleHopSpecificQuerySynthesizer(llm=generator.llm), 1.0),
]

print("Generating testset...")
testset = generator.generate(testset_size=10, query_distribution=query_distribution)
print(f"Generated {len(testset.samples)} test samples")

# Show the questions
print("\nGenerated questions:")
for i, sample in enumerate(testset.samples):
    print(f"  {i + 1}. {sample.eval_sample.user_input}")

print("\n" + "=" * 60)
print("SUCCESS: All RAGAS generation steps completed!")
print("=" * 60)
