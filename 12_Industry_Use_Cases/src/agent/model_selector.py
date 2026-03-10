import json
from typing import Any
from openai import OpenAI

from src.config import settings
from ..retrieval import embed_query, QdrantRetriever
from src.utils.langfuse_client import langfuse_trace, trace_llm_call


class ModelSelector:
    """RAG-based model selection agent."""

    def __init__(self):
        self.client = OpenAI(base_url=settings.LLM_INFERENCE_URL, api_key=settings.LLM_INFERENCE_KEY)
        self.retriever = QdrantRetriever()

    def select_model(self, query: str, profile: dict, top_k: int = 5) -> dict[str, Any]:
        """Select a model based on the dataset profile."""

        print(f"Retrieving relevant context for: {query}")
        
        with langfuse_trace("model_selection", input={"query": query, "profile": profile}) as trace:
            query_embedding = embed_query(query)
            retrieved = self.retriever.search(query_embedding, limit=top_k)

            contexts = [r["payload"]["content"] for r in retrieved]
            sources = [r["payload"] for r in retrieved]

            prompt = self._build_prompt(query, profile, contexts)

            print("Generating model recommendation...")
            response = self.client.chat.completions.create(
                model=settings.LLM_MODEL_NAME,
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
            trace_llm_call(trace, "generate_recommendation", prompt, recommendation, settings.LLM_MODEL_NAME)

            if trace:
                trace.update(output={"recommendation": recommendation})

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
