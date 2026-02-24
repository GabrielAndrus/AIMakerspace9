#!/usr/bin/env python3
"""
Test Activity #1 with qualitative Ragas evaluation
"""

import os

os.environ.setdefault("OPENAI_API_KEY", "dummy-key")
os.environ.setdefault("LLM_BASE_URL", "http://192.168.1.79:8080/v1")
os.environ.setdefault("LLM_MODEL", "minimax-m2.5-mlx@4bit")
os.environ.setdefault("EMBEDDING_MODEL", "text-embedding-qwen3-embedding-4b")
os.environ.setdefault("EMBEDDING_BASE_URL", "http://192.168.1.79:8080/v1")

print("=" * 60)
print("SETUP: Load docs and create retrievers")
print("=" * 60)

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

loader = TextLoader("data/HealthWellnessGuide.txt")
raw_docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
wellness_docs = text_splitter.split_documents(raw_docs)

generator_llm = LangchainLLMWrapper(
    ChatOpenAI(model="minimax-m2.5-mlx@4bit", base_url="http://192.168.1.79:8080/v1")
)
generator_embeddings = LangchainEmbeddingsWrapper(
    OpenAIEmbeddings(
        model="text-embedding-qwen3-embedding-4b",
        base_url="http://192.168.1.79:8080/v1",
        check_embedding_ctx_length=False,
    )
)

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
    model="minimax-m2.5-mlx@4bit", base_url="http://192.168.1.79:8080/v1"
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

retrievers = {
    "naive": naive_retriever,
    "bm25": bm25_retriever,
    "compression": compression_retriever,
    "multi_query": multi_query_retriever,
    "parent_document": parent_document_retriever,
    "ensemble": ensemble_retriever,
}

print("\n" + "=" * 60)
print("STEP 5: Generate testset")
print("=" * 60)

from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.transforms import default_transforms, apply_transforms
from ragas.testset import TestsetGenerator

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

default_transforms_list = default_transforms(
    documents=raw_docs, llm=generator_llm, embedding_model=generator_embeddings
)
apply_transforms(kg, default_transforms_list)

print(f"Knowledge graph built: {len(kg.nodes)} nodes")

generator = TestsetGenerator(
    llm=generator_llm, embedding_model=generator_embeddings, knowledge_graph=kg
)

print("Generating testset...")
testset = generator.generate_with_langchain_docs(raw_docs, testset_size=10)
print(f"Generated {len(testset.samples)} test samples")

test_questions = [sample.eval_sample.user_input for sample in testset.samples]
ground_truths = [sample.eval_sample.reference for sample in testset.samples]

print("\n" + "=" * 60)
print("STEP 6: Basic evaluation (retrieved docs + latency)")
print("=" * 60)

import time


def run_retriever_query(retriever, question):
    docs = retriever.invoke(question)
    return {"num_docs": len(docs), "contexts": [doc.page_content for doc in docs]}


results_dict = {}
for name, retriever in retrievers.items():
    print(f"Evaluating {name}...")
    start = time.time()
    all_results = [run_retriever_query(retriever, q) for q in test_questions]
    elapsed = time.time() - start
    results_dict[name] = {
        "total_retrieved": sum(r["num_docs"] for r in all_results),
        "avg_latency": elapsed / len(test_questions),
    }
    print(
        f"  {results_dict[name]['total_retrieved']} docs, {elapsed / len(test_questions):.3f}s/query"
    )

# Semantic
start = time.time()
semantic_results_list = [
    run_retriever_query(semantic_retriever, q) for q in test_questions
]
elapsed = time.time() - start
results_dict["semantic_chunking"] = {
    "total_retrieved": sum(r["num_docs"] for r in semantic_results_list),
    "avg_latency": elapsed / len(test_questions),
}
print(
    f"semantic_chunking: {results_dict['semantic_chunking']['total_retrieved']} docs, {elapsed / len(test_questions):.3f}s/query"
)

print("\n" + "=" * 60)
print("STEP 6b & 6c: Ragas qualitative evaluation")
print("=" * 60)

from ragas import evaluate
from ragas.metrics import context_precision, context_recall
from datasets import Dataset

eval_llm = LangchainLLMWrapper(
    ChatOpenAI(model="minimax-m2.5-mlx@4bit", base_url="http://192.168.1.79:8080/v1")
)
eval_embeddings = LangchainEmbeddingsWrapper(
    OpenAIEmbeddings(
        model="text-embedding-qwen3-embedding-4b",
        base_url="http://192.168.1.79:8080/v1",
        check_embedding_ctx_length=False,
    )
)

all_retrievers = dict(retrievers)
all_retrievers["semantic_chunking"] = semantic_retriever

ragas_results = {}

for name, retriever in all_retrievers.items():
    print(f"Ragas evaluating {name}...")

    eval_data = []
    for i, q in enumerate(test_questions):
        docs = retriever.invoke(q)
        retrieved_contexts = [doc.page_content for doc in docs]
        eval_data.append(
            {
                "question": q,
                "retrieved_contexts": retrieved_contexts,
                "ground_truth": ground_truths[i],
            }
        )

    ds = Dataset.from_list(eval_data)
    result = evaluate(
        ds,
        metrics=[context_precision, context_recall],
        llm=eval_llm,
        embeddings=eval_embeddings,
    )

    scores_df = result.to_pandas()
    avg_precision = float(scores_df["context_precision"].mean())
    avg_recall = float(scores_df["context_recall"].mean())

    ragas_results[name] = {
        "context_precision": avg_precision,
        "context_recall": avg_recall,
    }
    print(f"  Precision: {avg_precision:.3f}, Recall: {avg_recall:.3f}")

print("\n" + "=" * 60)
print("STEP 7: Combined Results")
print("=" * 60)

import pandas as pd

combined_results = {}
for name, metrics in results_dict.items():
    combined_results[name] = {
        "total_retrieved": metrics["total_retrieved"],
        "avg_latency": metrics["avg_latency"],
    }

for name, ragas_metrics in ragas_results.items():
    if name in combined_results:
        combined_results[name]["context_precision"] = ragas_metrics["context_precision"]
        combined_results[name]["context_recall"] = ragas_metrics["context_recall"]

df = pd.DataFrame(combined_results).T
df.columns = [
    "Total Retrieved",
    "Avg Latency (s)",
    "Context Precision",
    "Context Recall",
]

print("\n" + df.to_string())

most_docs = df["Total Retrieved"].idxmax()
fastest = df["Avg Latency (s)"].idxmin()
best_precision = df["Context Precision"].idxmax()
best_recall = df["Context Recall"].idxmax()

print(f"\nMost Documents Retrieved: {most_docs}")
print(f"Fastest: {fastest}")
print(f"Best Context Precision: {best_precision}")
print(f"Best Context Recall: {best_recall}")

print("\n" + "=" * 60)
print("SUCCESS!")
print("=" * 60)
