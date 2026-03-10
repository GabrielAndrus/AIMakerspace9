import re
from collections import Counter
from typing import Any

"""
Hybrid Retrieval Implementation for AutoML RAG System.

This implementation combines dense retrieval (semantic search using embeddings)
with sparse retrieval (BM25 keyword matching) and merges results using Reciprocal
Rank Fusion (RRF). This approach improves RAG accuracy by capturing both semantic
meaning and exact keyword matches, which is crucial for technical queries involving
specific AutoML terminology, configuration parameters, and error messages.
"""

from src.utils.langfuse_client import get_langfuse_client


class BM25Retriever:
    """Sparse retrieval using BM25 keyword matching."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_freqs = {}
        self.idf = {}
        self.doc_len = []
        self.avgdl = 0
        self.corpus = []

    def build_index(self, documents: list[dict[str, Any]]) -> None:
        """Build BM25 index from document chunks."""
        self.corpus = documents
        n_docs = len(documents)
        doc_freqs = Counter()
        self.doc_len = []

        for i, doc in enumerate(documents):
            text = doc.get("content", "").lower()
            tokens = self._tokenize(text)
            self.doc_len.append(len(tokens))

            for token in set(tokens):
                doc_freqs[token] += 1

        self.doc_freqs = dict(doc_freqs)
        self.avgdl = sum(self.doc_len) / n_docs if n_docs > 0 else 0

        self.idf = {}
        for word, freq in doc_freqs.items():
            self.idf[word] = ((n_docs - freq + 0.5) / (freq + 0.5)) + 1

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text into words."""
        text = re.sub(r'[^\w\s]', ' ', text)
        return [t for t in text.split() if len(t) > 1]

    def search(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        """Search using BM25 scoring."""
        query_tokens = self._tokenize(query.lower())
        scores = []

        for i, doc in enumerate(self.corpus):
            text = doc.get("content", "").lower()
            tokens = self._tokenize(text)
            token_freqs = Counter(tokens)

            score = 0
            for token in query_tokens:
                if token not in self.idf:
                    continue

                tf = token_freqs.get(token, 0)
                numerator = self.idf[token] * tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (len(tokens) / self.avgdl))
                score += numerator / denominator

            scores.append((i, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


class HybridRetriever:
    """Hybrid retriever combining dense and sparse search with RRF."""

    @staticmethod
    def create(collection_name: str = None, url: str = None, vector_size: int = 2560, k1: float = 1.5, b: float = 0.75, rrf_k: int = 60):
        """Factory method to create a HybridRetriever with its own QdrantRetriever."""
        from .qdrant_client import QdrantRetriever
        qdrant_retriever = QdrantRetriever(collection_name=collection_name, url=url, vector_size=vector_size)
        return HybridRetriever(qdrant_retriever, k1=k1, b=b, rrf_k=rrf_k)

    def __init__(self, qdrant_retriever, k1: float = 1.5, b: float = 0.75, rrf_k: int = 60):
        self.qdrant_retriever = qdrant_retriever
        self.bm25 = BM25Retriever(k1=k1, b=b)
        self.rrf_k = rrf_k
        self.documents = []
        self.indexed = False

    def build_index(self, documents: list[dict[str, Any]]) -> None:
        """Build both dense and sparse indexes."""
        self.documents = documents
        id_texts = [doc["id"] for doc in documents]
        contents = [doc.get("content", "") for doc in documents]

        from src.retrieval.embeddings import embed_documents
        embeddings = embed_documents(contents)

        payloads = [
            {
                "id": doc["id"],
                "title": doc.get("title", ""),
                "content": doc.get("content", ""),
                "source": doc.get("source", ""),
            }
            for doc in documents
        ]

        self.qdrant_retriever.add_documents(id_texts, embeddings, payloads)
        self.bm25.build_index(documents)
        self.indexed = True

    def reciprocal_rank_fusion(
        self, dense_results: list[tuple[int, float]], sparse_results: list[tuple[int, float]]
    ) -> list[tuple[int, float]]:
        """Combine rankings using Reciprocal Rank Fusion."""
        rrf_scores = {}

        for rank, (doc_idx, score) in enumerate(dense_results):
            rrf_score = 1 / (self.rrf_k + rank + 1)
            if doc_idx not in rrf_scores:
                rrf_scores[doc_idx] = 0
            rrf_scores[doc_idx] += rrf_score

        for rank, (doc_idx, score) in enumerate(sparse_results):
            rrf_score = 1 / (self.rrf_k + rank + 1)
            if doc_idx not in rrf_scores:
                rrf_scores[doc_idx] = 0
            rrf_scores[doc_idx] += rrf_score

        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return [(doc_idx, score) for doc_idx, score in sorted_results]

    def search(self, query: str, limit: int = 5, method: str = "hybrid") -> list[dict[str, Any]]:
        """Search using specified method: 'dense', 'sparse', or 'hybrid'."""
        if not self.indexed:
            raise ValueError("Index not built. Call build_index() first.")

        trace = get_langfuse_client().start_span(name="hybrid_search", input={"query": query, "method": method, "limit": limit})

        if method == "dense":
            from src.retrieval.embeddings import embed_query
            span = trace.start_span(name="dense_only")
            query_embedding = embed_query(query)
            results = self.qdrant_retriever.search(query_embedding, limit=limit)
            span.update(output={"num_results": len(results)})
            span.end()

        elif method == "sparse":
            span = trace.start_span(name="sparse_only")
            sparse_results = self.bm25.search(query, top_k=limit * 2)
            results = []
            for doc_idx, score in sparse_results[:limit]:
                doc = self.documents[doc_idx]
                results.append({
                    "id": doc["id"],
                    "score": score,
                    "payload": {
                        "id": doc["id"],
                        "title": doc.get("title", ""),
                        "content": doc.get("content", ""),
                        "source": doc.get("source", ""),
                    },
                })
            span.update(output={"num_results": len(results)})
            span.end()

        elif method == "hybrid":
            from src.retrieval.embeddings import embed_query
            query_embedding = embed_query(query)

            dense_span = trace.start_span(name="dense_retrieval")
            dense_results_full = self.qdrant_retriever.search(query_embedding, limit=limit * 2)
            dense_results = []
            doc_id_to_idx = {str(doc["id"]): i for i, doc in enumerate(self.documents)}
            for hit in dense_results_full:
                if str(hit["id"]) in doc_id_to_idx:
                    doc_idx = doc_id_to_idx[str(hit["id"])]
                    dense_results.append((doc_idx, hit["score"]))
            dense_span.update(output={"num_results": len(dense_results)})
            dense_span.end()

            sparse_span = trace.start_span(name="sparse_retrieval")
            sparse_results = self.bm25.search(query, top_k=limit * 2)
            sparse_span.update(output={"num_results": len(sparse_results)})
            sparse_span.end()

            fusion_span = trace.start_span(name="rrf_fusion")
            fused_results = self.reciprocal_rank_fusion(dense_results, sparse_results)
            results = []
            for doc_idx, score in fused_results[:limit]:
                doc = self.documents[doc_idx]
                results.append({
                    "id": doc["id"],
                    "score": score,
                    "payload": {
                        "id": doc["id"],
                        "title": doc.get("title", ""),
                        "content": doc.get("content", ""),
                        "source": doc.get("source", ""),
                    },
                })
            fusion_span.update(output={"num_results": len(results)})
            fusion_span.end()

        else:
            raise ValueError(f"Unknown method: {method}. Use 'dense', 'sparse', or 'hybrid'.")

        trace.update(output={"num_results": len(results)})
        trace.end()
        return results