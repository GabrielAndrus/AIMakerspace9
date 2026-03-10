import json
import os
import warnings
from typing import Any

import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, context_precision, context_recall
from ragas.run_config import RunConfig

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import settings
from src.retrieval.embeddings import embed_query, get_embedding_model
from src.retrieval.qdrant_client import QdrantRetriever
from src.utils.langfuse_client import langfuse_trace


class RAGASEvaluator:
    """Evaluates RAG pipeline using RAGAS metrics."""

    def __init__(self, collection_name: str = None, vector_size: int = 2560):
        self.retriever = QdrantRetriever(collection_name=collection_name, vector_size=vector_size)
        self.hybrid_retriever = None
        self.retrieval_method = "dense"
        self.comparison_results = {}
        self.llm_available = False
        self._setup_ragas_llm()

    def _setup_ragas_llm(self) -> None:
        """Configure RAGAS to use local LLM server instead of OpenAI."""
        try:
            from langchain_openai import ChatOpenAI
            from ragas.llms import LangchainLLMWrapper
            
            self.llm = ChatOpenAI(
                model=settings.LLM_MODEL_NAME,
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
        """Generate response using retrieved context.

        Raises RuntimeError if the LLM is unavailable, since a canned fallback
        response would produce meaningless RAGAS scores.
        """
        if not context:
            raise ValueError("No relevant context retrieved for the question. "
                           "Ensure the knowledge base is indexed.")

        context_text = "\n".join([f"- {c[:200]}..." if len(c) > 200 else f"- {c}" for c in context])

        if not self.llm_available:
            raise RuntimeError(
                f"LLM server not available at {settings.LLM_INFERENCE_URL}. "
                "Cannot generate responses for RAGAS evaluation without a working LLM."
            )

        prompt = f"""Based on the following context, answer the question concisely.

Context:
{context_text}

Question: {question}

Answer:"""

        response = self.llm.invoke(prompt)
        return response.content

    def _extract_metric_value(self, result, metric_name: str) -> float:
        """Extract a single numeric value from RAGAS EvaluationResult, handling various formats.

        Args:
            result: Either a dict, EvaluationResult object, or pandas DataFrame
            metric_name: Name of the metric to extract (e.g., 'faithfulness')

        Returns:
            float: The extracted score value

        Raises:
            ValueError: If the metric cannot be extracted from the result.
        """
        value = None

        if hasattr(result, 'scores'):
            value = result.scores[0].get(metric_name)
        elif hasattr(result, 'to_pandas'):
            df = result.to_pandas()
            if metric_name in df.columns:
                value = df[metric_name].iloc[0]
        elif isinstance(result, dict):
            value = result.get(metric_name)

        if value is None:
            raise ValueError(
                f"Could not extract metric '{metric_name}' from RAGAS result. "
                f"Result type: {type(result).__name__}. "
                f"Available: {list(result.keys()) if isinstance(result, dict) else 'unknown'}"
            )

        if isinstance(value, (int, float)):
            return float(value)

        if hasattr(value, 'item'):
            return float(value.item())

        return float(value)

    def evaluate_single(
        self, question: str, ground_truth: str = None
    ) -> dict[str, Any]:
        """Evaluate a single question-answer pair.

        Raises an error if the LLM is unavailable or RAGAS evaluation fails,
        rather than silently returning fake 0.5 scores.
        """

        with langfuse_trace("ragas_single_evaluation", metadata={"retrieval_method": self.retrieval_method}, input={"question": question, "ground_truth": ground_truth}) as span:
            contexts = self.retrieve_context(question)
            answer = self.generate_response(question, contexts)

            if not self.llm_available or not self.ragas_llm:
                raise RuntimeError(
                    f"LLM server not available at {settings.LLM_INFERENCE_URL}. "
                    "RAGAS evaluation requires a working LLM to score faithfulness, "
                    "context precision, and context recall. Please ensure the LLM "
                    "inference server is running."
                )

            data_dict = {
                "question": [question],
                "answer": [answer],
                "contexts": [contexts],
            }

            if ground_truth:
                data_dict["ground_truth"] = [ground_truth]

            dataset = Dataset.from_dict(data_dict)

            metrics = [faithfulness, context_precision, context_recall]
            for metric in metrics:
                metric.llm = self.ragas_llm

            run_config = RunConfig(
                timeout=300,
                max_retries=3,
                max_wait=30
            )

            result = evaluate(
                dataset=dataset,
                metrics=metrics,
                run_config=run_config
            )

            faithfulness_score = self._extract_metric_value(result, "faithfulness")
            context_precision_score = self._extract_metric_value(result, "context_precision")
            context_recall_score = self._extract_metric_value(result, "context_recall")

            scores = {
                "faithfulness": faithfulness_score,
                "context_precision": context_precision_score,
                "context_recall": context_recall_score,
            }

            # Update span with output scores and report scores to Langfuse
            if span is not None:
                span.update(output={"answer": answer, "scores": scores, "num_contexts": len(contexts)})
                for metric_name, metric_value in scores.items():
                    span.score(name=metric_name, value=metric_value, data_type="NUMERIC")

            return {
                "question": question,
                "answer": answer,
                "ground_truth": ground_truth,
                "contexts": contexts,
                **scores,
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

        questions = [item["question"] for item in test_data]

        with langfuse_trace(
            "ragas_dataset_evaluation",
            metadata={"dataset_path": dataset_path, "retrieval_method": self.retrieval_method},
            input={"questions": questions, "retrieval_method": self.retrieval_method, "total_questions": len(test_data)},
        ) as span:
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

            per_question_scores = [
                {
                    "question": r["question"],
                    "faithfulness": r["faithfulness"],
                    "context_precision": r["context_precision"],
                    "context_recall": r["context_recall"],
                }
                for r in results
            ]

            # Set output on the trace with both per-question and average scores
            if span is not None:
                span.update(output={
                    "average_scores": avg_scores,
                    "per_question_scores": per_question_scores,
                    "total_evaluated": len(results),
                })
                # Report average scores to the Langfuse trace
                for metric_name, metric_value in avg_scores.items():
                    span.score_trace(
                        name=f"ragas_{metric_name}",
                        value=metric_value,
                        data_type="NUMERIC",
                        comment=f"Average {metric_name} over {len(results)} questions ({self.retrieval_method} retrieval)",
                    )

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