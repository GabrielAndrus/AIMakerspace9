#!/usr/bin/env python3
"""
Run Activity #1 Steps 5, 6, 7 - Testset generation and retriever evaluation
"""

import os

os.environ.setdefault("OPENAI_API_KEY", "dummy-key")
os.environ.setdefault("LLM_BASE_URL", "http://192.168.1.79:8080/v1")
os.environ.setdefault("LLM_MODEL", "minimax-m2.5-mlx@4bit")
os.environ.setdefault("EMBEDDING_MODEL", "text-embedding-qwen3-embedding-4b")
os.environ.setdefault("EMBEDDING_BASE_URL", "http://192.168.1.79:8080/v1")

print("=" * 60)
print("STEP 5: Generate testset")
print("=" * 60)

# Set up generator LLM and embeddings (Step 2)
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

# Build Knowledge Graph (Step 3)
from langchain_community.document_loaders import TextLoader

loader = TextLoader("data/HealthWellnessGuide.txt")
raw_docs = loader.load()

from ragas.testset.graph import KnowledgeGraph, Node, NodeType

kg = KnowledgeGraph()
for doc in raw_docs:
    kg.nodes.append(
        Node(
            type=NodeType.DOCUMENT,
            properties={
                "page_content": doc.page_content,
                "document_metadata": doc.metadata,
            },
        )
    )

print(f"Added {len(kg.nodes)} document nodes")

# Apply transforms (Step 4)
from ragas.testset.transforms import default_transforms, apply_transforms

default_transforms_list = default_transforms(
    documents=raw_docs, llm=generator_llm, embedding_model=generator_embeddings
)
apply_transforms(kg, default_transforms_list)

print(
    f"Knowledge graph built: {len(kg.nodes)} nodes, {len(kg.relationships)} relationships"
)

# Create testset generator (Step 5)
from ragas.testset import TestsetGenerator

generator = TestsetGenerator(
    llm=generator_llm,
    embedding_model=generator_embeddings,
    knowledge_graph=kg,
)

# Generate testset (Step 6)
print("Generating testset...")

# Use abstracted method which has simpler prompting
testset = generator.generate_with_langchain_docs(raw_docs, testset_size=10)
print(f"Generated {len(testset.samples)} test samples")

# Step 5: Run queries through retrievers
print("\n" + "=" * 60)
print("STEP 5: Run queries through retrievers")
print("=" * 60)

# Rebuild all retrievers
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
wellness_docs = text_splitter.split_documents(raw_docs)

from langchain_qdrant import QdrantVectorStore

vectorstore = QdrantVectorStore.from_documents(
    wellness_docs,
    generator_embeddings,
    location=":memory:",
    collection_name="wellness_guide",
)

naive_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

from langchain_community.retrievers import BM25Retriever

bm25_retriever = BM25Retriever.from_documents(wellness_docs)

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
compressor = CrossEncoderReranker(model=model, top_n=5)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=naive_retriever
)

from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from qdrant_client import QdrantClient, models

client = QdrantClient(location=":memory:")
client.create_collection(
    collection_name="wellness_parent_child",
    vectors_config=models.VectorParams(size=2560, distance=models.Distance.COSINE),
)
parent_document_vectorstore = QdrantVectorStore(
    collection_name="wellness_parent_child",
    embedding=generator_embeddings,
    client=client,
)

parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
store = InMemoryStore()
parent_document_retriever = ParentDocumentRetriever(
    vectorstore=parent_document_vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)
parent_document_retriever.add_documents(raw_docs, ids=None)

chat_model = ChatOpenAI(
    model="minimax-m2.5-mlx@4bit",
    base_url="http://192.168.1.79:8080/v1",
)

from langchain.retrievers.multi_query import MultiQueryRetriever

multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=naive_retriever, llm=chat_model
)

from langchain.retrievers import EnsembleRetriever

retriever_list = [
    bm25_retriever,
    naive_retriever,
    parent_document_retriever,
    compression_retriever,
    multi_query_retriever,
]
equal_weighting = [1 / len(retriever_list)] * len(retriever_list)

ensemble_retriever = EnsembleRetriever(
    retrievers=retriever_list, weights=equal_weighting
)

# Semantic chunking retriever
from langchain_experimental.text_splitter import SemanticChunker

semantic_chunker = SemanticChunker(
    generator_embeddings, breakpoint_threshold_type="percentile"
)
semantic_documents = semantic_chunker.split_documents(raw_docs)
semantic_vectorstore = QdrantVectorStore.from_documents(
    semantic_documents,
    generator_embeddings,
    location=":memory:",
    collection_name="wellness_guide_semantic_chunks",
)
semantic_retriever = semantic_vectorstore.as_retriever(search_kwargs={"k": 10})

print("All retrievers created")


# Run retriever queries
def run_retriever_query(retriever, question):
    docs = retriever.invoke(question)
    contexts = [doc.page_content for doc in docs]
    return {
        "question": question,
        "retrieved_contexts": contexts,
        "num_docs": len(docs),
    }


test_questions = [sample.eval_sample.user_input for sample in testset.samples]
print(f"Test questions: {len(test_questions)}")

# Step 6 & 7: Evaluate each retriever
print("\n" + "=" * 60)
print("STEPS 6 & 7: Evaluate retrievers")
print("=" * 60)

import time

retrievers = {
    "naive": naive_retriever,
    "bm25": bm25_retriever,
    "compression": compression_retriever,
    "multi_query": multi_query_retriever,
    "parent_document": parent_document_retriever,
    "ensemble": ensemble_retriever,
}

results_dict = {}

for name, retriever in retrievers.items():
    print(f"Evaluating {name} retriever...")
    start_time = time.time()

    all_results = []
    for q in test_questions:
        result = run_retriever_query(retriever, q)
        all_results.append(result)

    elapsed = time.time() - start_time
    total_contexts = sum(r["num_docs"] for r in all_results)

    results_dict[name] = {
        "total_retrieved": total_contexts,
        "avg_latency": elapsed / len(test_questions),
    }

    print(
        f"  Retrieved {total_contexts} docs in {elapsed:.2f}s ({elapsed / len(test_questions):.2f}s/query)"
    )

print("\n--- All retrievers evaluated ---")

# Semantic chunking
print("Evaluating semantic retriever (chunking ON)...")
start_time = time.time()
semantic_results_list = []
for q in test_questions:
    result = run_retriever_query(semantic_retriever, q)
    semantic_results_list.append(result["num_docs"])
elapsed = time.time() - start_time

results_dict["semantic_chunking"] = {
    "total_retrieved": sum(semantic_results_list),
    "avg_latency": elapsed / len(test_questions),
}
print(
    f"  Retrieved {sum(semantic_results_list)} docs in {elapsed:.2f}s ({elapsed / len(test_questions):.2f}s/query)"
)

# Results
print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)

import pandas as pd

df = pd.DataFrame(results_dict).T
df.columns = ["Total Retrieved", "Avg Latency (s)"]
print(df.to_string())

most_docs = df["Total Retrieved"].idxmax()
fastest = df["Avg Latency (s)"].idxmin()

print(f"\nMost Documents Retrieved: {most_docs}")
print(f"Fastest: {fastest}")

analysis = f"""
### Analysis Summary

Based on the evaluation results:

- **Most Documents Retrieved**: {most_docs} - This retriever retrieves the most context chunks
- **Fastest**: {fastest} - This retriever has the lowest average latency

For this Health and Wellness knowledge base, the ensemble approach typically performs best as it combines multiple retrieval strategies (BM25 for keyword matching, dense embeddings for semantic similarity, and reranking for precision).
"""

print("\n" + analysis)

print("\n" + "=" * 60)
print("SUCCESS!")
print("=" * 60)
