import json
from typing import Any
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.retrieval import embed_query, QdrantRetriever
from src.agent import DatasetProfiler
import pandas as pd


def evaluate_rag(question: str, ground_truth: str) -> dict[str, float]:
    """Evaluate RAG performance for a single question."""

    retriever = QdrantRetriever()
    query_embedding = embed_query(question)
    results = retriever.search(query_embedding, limit=5)

    contexts = [r["payload"]["content"] for r in results]

    dataset = Dataset.from_dict(
        {
            "question": [question],
            "answer": [ground_truth],
            "contexts": [contexts],
            "ground_truth": [[ground_truth]],
        }
    )

    score = evaluate(
        dataset=dataset, metrics=[faithfulness, context_precision, context_recall, answer_relevancy]
    )

    return {
        "faithfulness": score["faithfulness"],
        "context_precision": score["context_precision"],
        "context_recall": score["context_recall"],
        "answer_relevancy": score["answer_relevancy"],
    }


def run_evaluation(eval_dataset_path: str = "data/eval_dataset.jsonl") -> dict[str, Any]:
    """Run evaluation on the full synthetic dataset."""

    eval_data = []
    with open(eval_dataset_path, "r") as f:
        for line in f:
            eval_data.append(json.loads(line.strip()))

    print(f"Running RAGAS evaluation on {len(eval_data)} questions...")

    all_results = []
    for item in eval_data:
        try:
            result = evaluate_rag(item["question"], item["ground_truth"])
            result["question"] = item["question"]
            all_results.append(result)
            print(f"  - {item['question'][:50]}... -> Faithfulness: {result['faithfulness']:.3f}")
        except Exception as e:
            print(f"  - Error evaluating: {e}")

    if not all_results:
        return {"error": "No evaluations completed"}

    avg_scores = {
        "faithfulness": sum(r["faithfulness"] for r in all_results) / len(all_results),
        "context_precision": sum(r["context_precision"] for r in all_results) / len(all_results),
        "context_recall": sum(r["context_recall"] for r in all_results) / len(all_results),
        "answer_relevancy": sum(r["answer_relevancy"] for r in all_results) / len(all_results),
    }

    return {
        "individual_results": all_results,
        "average_scores": avg_scores,
        "total_evaluated": len(all_results),
    }


if __name__ == "__main__":
    results = run_evaluation()
    print("\n=== RAGAS Evaluation Results ===")
    for metric, score in results.get("average_scores", {}).items():
        print(f"{metric}: {score:.3f}")
