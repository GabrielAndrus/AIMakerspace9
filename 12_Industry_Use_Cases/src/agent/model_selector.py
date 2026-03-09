import os
import json
from typing import Any
from openai import OpenAI
from ..retrieval import embed_query, QdrantRetriever


class ModelSelector:
    """RAG-based model selection agent."""

    def __init__(self):
        self.inference_url = os.getenv("LLM_INFERENCE_URL", "http://192.168.1.185:8080/v1")
        self.inference_key = os.getenv("LLM_INFERENCE_KEY", "not-needed")
        self.client = OpenAI(base_url=self.inference_url, api_key=self.inference_key)
        self.retriever = QdrantRetriever()

    def select_model(self, query: str, profile: dict, top_k: int = 5) -> dict[str, Any]:
        """Select a model based on the dataset profile."""

        print(f"Retrieving relevant context for: {query}")
        query_embedding = embed_query(query)
        retrieved = self.retriever.search(query_embedding, limit=top_k)

        contexts = [r["payload"]["content"] for r in retrieved]
        sources = [r["payload"] for r in retrieved]

        prompt = self._build_prompt(query, profile, contexts)

        print("Generating model recommendation...")
        response = self.client.chat.completions.create(
            model="minimax-m2.5-mlx@8bit",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert ML engineer. Recommend models based on dataset characteristics and scikit-learn best practices.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=500,
        )

        recommendation = response.choices[0].message.content

        return {
            "recommendation": recommendation,
            "retrieved_contexts": contexts,
            "sources": sources,
            "query": query,
        }

    def _build_prompt(self, query: str, profile: dict, contexts: list[str]) -> str:
        """Build the prompt for model selection."""

        context_text = "\n\n".join([f"Context {i + 1}: {ctx}" for i, ctx in enumerate(contexts)])

        prompt = f"""Given the following dataset profile and relevant documentation contexts, recommend the best ML model(s) to use.

Dataset Profile:
{json.dumps(profile, indent=2)}

Query: {query}

Relevant Documentation:
{context_text}

Based on the above, provide:
1. Recommended model(s) with justification
2. Key hyperparameters to consider
3. Any preprocessing recommendations

Provide your recommendation in a clear, structured format."""

        return prompt
