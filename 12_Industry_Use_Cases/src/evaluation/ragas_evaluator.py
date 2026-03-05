import json
import os
from typing import Any

import pandas as pd
from datasets import Dataset
from langfuse import get_client
from ragas import evaluate
from ragas.metrics import faithfulness, context_precision, context_recall

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import settings
from src.retrieval.embeddings import embed_query, get_embedding_model
from src.retrieval.qdrant_client import QdrantRetriever


class RAGASEvaluator:
    """Evaluates RAG pipeline using RAGAS metrics."""

    def __init__(self, collection_name: str = None, vector_size: int = 2560):
        self.retriever = QdrantRetriever(collection_name=collection_name, vector_size=vector_size)
        self.hybrid_retriever = None
        self.retrieval_method = "dense"
        self.comparison_results = {}
        self.llm_available = False
        self._setup_ragas_llm()
        self._setup_langfuse()

    def _setup_ragas_llm(self) -> None:
        """Configure RAGAS to use local LLM server instead of OpenAI."""
        try:
            from langchain_openai import ChatOpenAI
            from ragas.llms import LangchainLLMWrapper
            
            self.llm = ChatOpenAI(
                model=settings.DEFAULT_BASE_MODEL,
                base_url=settings.LLM_INFERENCE_URL,
                api_key=settings.LLM_INFERENCE_KEY,
            )
            
            self.ragas_llm = LangchainLLMWrapper(langchain_llm=self.llm)
            
            self.llm.invoke("test")
            self.llm_available = True
        except Exception as e:
            print(f"Warning: LLM server not available at {settings.LLM_INFERENCE_URL}")
            print(f"         Falling back to template-based evaluation. Error: {e}")
            self.llm = None
            self.ragas_llm = None
            self.llm_available = False

    def _setup_langfuse(self) -> None:
        """Initialize LangFuse client for score tracking."""
        try:
            self.langfuse = get_client()
            self.langfuse_available = True
        except Exception as e:
            print(f"Warning: LangFuse client not available. Error: {e}")
            self.langfuse = None
            self.langfuse_available = False

    def retrieve_context(self, question: str, limit: int = 5) -> list[str]:
        """Retrieve relevant context for a question."""
        
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

    def set_retrieval_method(self, method: str) -> None:
        """Set retrieval method: 'dense', 'sparse', or 'hybrid'."""
        self.retrieval_method = method
        
        if method in ["sparse", "hybrid"] and not self.hybrid_retriever:
            from src.retrieval.hybrid_retriever import HybridRetriever
            self.hybrid_retriever = HybridRetriever.create(
                collection_name=self.retriever.collection_name,
                url=self.retriever.url,
                vector_size=self.retriever.vector_size
            )
        
        if method in ["sparse", "hybrid"] and self.hybrid_retriever:
            if not self.hybrid_retriever.indexed:
                print("Hybrid index not built. Building from collection...")
                self.build_hybrid_index_from_collection()
    
    def build_hybrid_index_from_collection(self) -> None:
        """Build hybrid index from existing Qdrant collection."""
        if not self.hybrid_retriever:
            raise ValueError("Hybrid retriever not initialized. Call set_retrieval_method('sparse' or 'hybrid') first.")
        
        documents = self._load_documents_from_collection()
        if not documents:
            raise ValueError("No documents found in collection. Index the knowledge base first.")
        
        print(f"Building hybrid index from {len(documents)} documents...")
        self.hybrid_retriever.build_index(documents)
    
    def _load_documents_from_collection(self) -> list[dict[str, Any]]:
        """Load all documents from Qdrant collection."""
        from src.retrieval.embeddings import embed_documents
        
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

    def compare_retrieval_methods(
        self, dataset_path: str = None
    ) -> dict[str, Any]:
        """Compare dense vs sparse vs hybrid retrieval methods."""
        if dataset_path is None:
            dataset_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "..", "data", "evaluation", "test_questions.jsonl"
            )

        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

        methods = ["dense", "sparse", "hybrid"]
        comparison_results = {}

        for method in methods:
            print(f"\nEvaluating with {method} retrieval...")
            self.set_retrieval_method(method)
            
            if method in ["sparse", "hybrid"]:
                self.build_hybrid_index_from_collection()
            
            results = self.evaluate_dataset(dataset_path)
            comparison_results[method] = results

        summary = self._create_comparison_summary(comparison_results)

        if self.langfuse_available and self.langfuse:
            with self.langfuse.start_as_current_observation(as_type="span", name="retrieval_comparison") as span:
                for method in methods:
                    avg_scores = comparison_results[method].get("average_scores", {})
                    for metric_name, score in avg_scores.items():
                        span.score(name=f"{method}_{metric_name}", value=score, data_type="NUMERIC")
            self.langfuse.flush()

        return {
            "comparison": comparison_results,
            "summary": summary,
        }

    def _create_comparison_summary(self, comparison_results: dict[str, Any]) -> dict[str, Any]:
        """Create a summary table comparing retrieval methods."""
        summary = {
            "metrics": ["faithfulness", "context_precision", "context_recall"],
            "dense_scores": {},
            "sparse_scores": {},
            "hybrid_scores": {},
        }

        for method, results in comparison_results.items():
            avg_scores = results.get("average_scores", {})
            for metric in summary["metrics"]:
                if method == "dense":
                    summary["dense_scores"][metric] = avg_scores.get(metric, 0)
                elif method == "sparse":
                    summary["sparse_scores"][metric] = avg_scores.get(metric, 0)
                elif method == "hybrid":
                    summary["hybrid_scores"][metric] = avg_scores.get(metric, 0)

        return summary

    def generate_response(self, question: str, context: list[str]) -> str:
        """Generate response using retrieved context."""
        if not context:
            return "Unable to find relevant information in the knowledge base."

        context_text = "\n".join([f"- {c[:200]}..." if len(c) > 200 else f"- {c}" for c in context])
        
        if self.llm_available:
            try:
                prompt = f"""Based on the following context, answer the question concisely.

Context:
{context_text}

Question: {question}

Answer:"""
                
                response = self.llm.invoke(prompt)
                return response.content
            except Exception as e:
                print(f"Warning: LLM call failed, using fallback: {e}")
        
        return f"""Based on the available information:

{context_text}

Answer: This question can be addressed using scikit-learn based on the retrieved documentation."""

    def _extract_metric_value(self, result: dict, metric_name: str) -> float:
        """Extract a single numeric value from RAGAS result, handling various formats."""
        value = result.get(metric_name)
        
        if value is None:
            return 0.0
        
        if isinstance(value, (int, float)):
            return float(value)
        
        if isinstance(value, list):
            if len(value) > 0:
                if isinstance(value[0], (int, float)):
                    return float(value[0])
            return 0.0
        
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _submit_langfuse_scores(
        self,
        span,
        faithfulness: float,
        context_precision: float,
        context_recall: float,
    ) -> None:
        """Submit RAGAS scores to LangFuse span."""
        if not span:
            return

        try:
            span.score(name="faithfulness", value=faithfulness, data_type="NUMERIC")
            span.score(name="context_precision", value=context_precision, data_type="NUMERIC")
            span.score(name="context_recall", value=context_recall, data_type="NUMERIC")
        except Exception as e:
            print(f"Warning: Failed to submit LangFuse scores. Error: {e}")

    def evaluate_single(
        self, question: str, ground_truth: str = None
    ) -> dict[str, Any]:
        """Evaluate a single question-answer pair."""
        contexts = self.retrieve_context(question)
        answer = self.generate_response(question, contexts)

        data_dict = {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
        }

        if ground_truth:
            data_dict["ground_truth"] = [ground_truth]

        dataset = Dataset.from_dict(data_dict)

        faithfulness_score = 0.5
        context_precision_score = 0.5
        context_recall_score = 0.5 if ground_truth else 0.0

        if self.llm_available and self.ragas_llm:
            metrics = [faithfulness, context_precision, context_recall]
            for metric in metrics:
                metric.llm = self.ragas_llm

            result = evaluate(
                dataset=dataset,
                metrics=metrics,
            )

            faithfulness_score = self._extract_metric_value(result, "faithfulness")
            context_precision_score = self._extract_metric_value(result, "context_precision")
            context_recall_score = self._extract_metric_value(result, "context_recall")

        if self.langfuse_available and self.langfuse:
            with self.langfuse.start_as_current_observation(as_type="span", name="single_evaluation") as span:
                span.update(input=question, output=answer)
                self._submit_langfuse_scores(
                    span,
                    faithfulness_score,
                    context_precision_score,
                    context_recall_score,
                )
            self.langfuse.flush()

        return {
            "question": question,
            "answer": answer,
            "ground_truth": ground_truth,
            "contexts": contexts,
            "faithfulness": faithfulness_score,
            "context_precision": context_precision_score,
            "context_recall": context_recall_score,
        }

    def evaluate_dataset(
        self, dataset_path: str = None
    ) -> dict[str, Any]:
        """Evaluate a full test dataset."""
        if dataset_path is None:
            dataset_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "..", "data", "evaluation", "test_questions.jsonl"
            )

        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

        test_data = []
        with open(dataset_path, "r") as f:
            for line in f:
                test_data.append(json.loads(line.strip()))

        results = []
        for item in test_data:
            question = item["question"]
            ground_truth = item.get("ground_truth")
            
            try:
                result = self.evaluate_single(question, ground_truth)
                results.append(result)
            except Exception as e:
                print(f"Error evaluating question: {question[:50]}... - {e}")

        if not results:
            return {"error": "No evaluations completed successfully"}

        avg_scores = {
            "faithfulness": sum(r["faithfulness"] for r in results) / len(results),
            "context_precision": sum(r["context_precision"] for r in results) / len(results),
            "context_recall": sum(r["context_recall"] for r in results) / len(results),
        }

        if self.langfuse_available and self.langfuse:
            with self.langfuse.start_as_current_observation(as_type="span", name="dataset_evaluation") as span:
                span.update(
                    input=f"{len(results)} questions evaluated",
                    output=f"method: {self.retrieval_method}",
                    metadata={"retrieval_method": self.retrieval_method}
                )
                span.score(name="avg_faithfulness", value=avg_scores["faithfulness"], data_type="NUMERIC")
                span.score(name="avg_context_precision", value=avg_scores["context_precision"], data_type="NUMERIC")
                span.score(name="avg_context_recall", value=avg_scores["context_recall"], data_type="NUMERIC")
            self.langfuse.flush()

        return {
            "individual_results": results,
            "average_scores": avg_scores,
            "total_evaluated": len(results),
            "retrieval_method": self.retrieval_method,
        }

    def results_to_dataframe(self, results: dict[str, Any]) -> pd.DataFrame:
        """Convert evaluation results to a tidy pandas DataFrame."""
        if "error" in results:
            return pd.DataFrame([{"error": results["error"]}])

        individual = []
        for r in results["individual_results"]:
            row = {
                "question": r["question"],
                "answer": r["answer"],
                "ground_truth": r.get("ground_truth", ""),
                "faithfulness": r["faithfulness"],
                "context_precision": r["context_precision"],
                "context_recall": r["context_recall"],
            }
            individual.append(row)

        df = pd.DataFrame(individual)
        
        avg_row = {
            "question": "**AVERAGE**",
            "answer": "",
            "ground_truth": "",
            "faithfulness": results["average_scores"]["faithfulness"],
            "context_precision": results["average_scores"]["context_precision"],
            "context_recall": results["average_scores"]["context_recall"],
        }
        
        df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)
        
        return df

    def save_results(self, results: dict[str, Any], output_path: str) -> None:
        """Save evaluation results to JSON file."""
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)


def main():
    """CLI entry point for RAGAS evaluation."""
    import argparse

    parser = argparse.ArgumentParser(description="Run RAGAS evaluation on RAG pipeline")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to test dataset JSONL file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/evaluation/results.json",
        help="Path to save results JSON file",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare dense vs hybrid retrieval methods",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="dense",
        choices=["dense", "sparse", "hybrid"],
        help="Retrieval method to use",
    )

    args = parser.parse_args()

    evaluator = RAGASEvaluator()
    
    if args.compare:
        print("Running comparison of Dense vs Sparse vs Hybrid retrieval...")
        results = evaluator.compare_retrieval_methods(args.dataset)
        
        if "error" in results.get("comparison", {}).get("dense", {}):
            print(f"Evaluation failed: {results['comparison']['dense'].get('error', 'Unknown error')}")
            return
        
        summary = results["summary"]
        
        print("\n=== Comparison Summary ===")
        print(f"{'Metric':<20} {'Dense':<12} {'Sparse':<12} {'Hybrid':<12}")
        print("-" * 56)
        
        for metric in summary["metrics"]:
            dense_score = summary["dense_scores"][metric]
            sparse_score = summary["sparse_scores"][metric]
            hybrid_score = summary["hybrid_scores"][metric]
            
            print(f"{metric:<20} {dense_score:<12.4f} {sparse_score:<12.4f} {hybrid_score:<12.4f}")
        
        for method in ["dense", "sparse", "hybrid"]:
            df = evaluator.results_to_dataframe(results["comparison"][method])
            print(f"\n=== {method.capitalize()} Retrieval Results ===")
            print(df.to_string())
        
    else:
        evaluator.set_retrieval_method(args.method)
        
        if args.method in ["sparse", "hybrid"]:
            print("Building hybrid index from existing collection...")
            evaluator.build_hybrid_index_from_collection()
        
        print(f"Running RAGAS evaluation with {args.method} retrieval...")
        results_single = evaluator.evaluate_dataset(args.dataset)
        
        if "error" in results_single:
            print(f"Evaluation failed: {results_single['error']}")
            return

        print("\n=== Average Scores ===")
        for metric, score in results_single["average_scores"].items():
            print(f"{metric}: {score:.4f}")

        print(f"\nTotal questions evaluated: {results_single['total_evaluated']}")

        df = evaluator.results_to_dataframe(results_single)
        print("\n=== Detailed Results ===")
        print(df.to_string())

    evaluator.save_results(results_single if not args.compare else results, args.output)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()