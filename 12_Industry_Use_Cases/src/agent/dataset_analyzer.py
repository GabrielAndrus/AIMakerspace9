"""Dataset analyzer agent using DeepAgents pattern.

This agent analyzes uploaded datasets and recommends appropriate training methods
(SFT, DPO, or GRPO) based on data structure and content, enhanced with RAG
retrieval from the knowledge base.
"""

import json
from pathlib import Path
from typing import Any, TypedDict

from src.config import settings
from src.utils.langfuse_client import langfuse_trace, trace_llm_call, trace_retrieval


class DatasetAnalysis(TypedDict):
    file_path: str
    file_type: str
    sample_data: list[dict] | str
    detected_format: str
    recommended_method: str
    reasoning: str
    issues: list[str]
    suggestions: list[str]


class DatasetAnalyzer:
    """Analyze datasets and recommend training methods using LLM + RAG."""

    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.LLM_MODEL_NAME
        self.model = self._create_model()
        self.retriever = None
        self._embed_query = None
        self._setup_retriever()

    def _create_model(self):
        from langchain.chat_models import init_chat_model
        return init_chat_model(
            settings.LLM_MODEL_NAME,
            model_provider="openai",
            base_url=settings.LLM_INFERENCE_URL,
            api_key=settings.LLM_INFERENCE_KEY,
            temperature=0.1,
        )

    def _setup_retriever(self):
        try:
            from src.retrieval import embed_query, QdrantRetriever
            self.retriever = QdrantRetriever()
            self._embed_query = embed_query
        except Exception:
            self.retriever = None

    def _retrieve_context(self, detected_format: str, file_type: str, trace=None) -> str:
        """Retrieve relevant training method documentation from Qdrant."""
        if self.retriever is None:
            return ""

        format_to_query = {
            "dpo": "DPO Direct Preference Optimization dataset format requirements chosen rejected",
            "grpo": "GRPO Group Relative Policy Optimization reward function ground_truth dataset format",
            "sft_conversational": "SFT Supervised Fine-Tuning conversational messages dataset format",
            "sft_continued_pretraining": "SFT continued pre-training text dataset format language modeling",
            "sft_qa": "SFT question answer pairs dataset format instruction fine-tuning",
            "sft_instruction": "SFT instruction prompt completion dataset format",
            "unknown": "LLM training method selection dataset format detection SFT DPO GRPO",
        }

        query = format_to_query.get(detected_format, format_to_query["unknown"])
        query += f" {file_type} file format"

        try:
            query_embedding = self._embed_query(query)
            results = self.retriever.search(query_embedding, limit=5)

            if trace:
                trace_retrieval(
                    trace, "rag_dataset_analysis",
                    query, [{"score": r.get("score"), "title": r["payload"].get("title")} for r in results[:5]]
                )

            contexts = [r["payload"]["content"] for r in results if r.get("payload", {}).get("content")]
            if contexts:
                return "\n\n".join([f"[{i+1}] {ctx}" for i, ctx in enumerate(contexts)])
        except Exception as e:
            print(f"[DatasetAnalyzer] RAG retrieval failed: {e}")

        return ""

    def analyze(self, file_path: str) -> DatasetAnalysis:
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        file_type = self._detect_file_type(path)
        sample_data = self._extract_sample(path, file_type)
        detected_format = self._detect_format(sample_data, file_type)

        return self._run_agent_analysis(file_path, file_type, sample_data, detected_format)

    def _detect_file_type(self, path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix in [".txt", ".text"]:
            return "text"
        elif suffix == ".pdf":
            return "pdf"
        elif suffix in [".json", ".jsonl"]:
            return "jsonl"
        elif suffix == ".csv":
            return "csv"
        else:
            return "unknown"

    def _extract_sample(self, path: Path, file_type: str) -> list[dict] | str:
        if file_type == "jsonl":
            return self._extract_jsonl_sample(path)
        elif file_type in ["text", "txt"]:
            return self._extract_text_sample(path)
        elif file_type == "csv":
            return self._extract_csv_sample(path)
        else:
            with open(path, "r", encoding="utf-8") as f:
                lines = []
                for i, line in enumerate(f):
                    if i >= 10:
                        break
                    lines.append(line.rstrip())
                return "\n".join(lines)

    def _extract_jsonl_sample(self, path: Path) -> list[dict]:
        samples = []
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 10:
                    break
                try:
                    sample = json.loads(line.strip())
                    samples.append(sample)
                except json.JSONDecodeError:
                    continue
        return samples

    def _extract_text_sample(self, path: Path) -> str:
        with open(path, "r", encoding="utf-8") as f:
            lines = []
            for i, line in enumerate(f):
                if i >= 10:
                    break
                lines.append(line.rstrip())
            return "\n".join(lines)

    def _extract_csv_sample(self, path: Path) -> list[dict]:
        import pandas as pd
        df = pd.read_csv(path)
        return df.head(10).to_dict("records")

    def _detect_format(self, sample_data: list[dict] | str, file_type: str) -> str:
        if isinstance(sample_data, list) and len(sample_data) > 0:
            first = sample_data[0]

            if "prompt" in first and "chosen" in first and "rejected" in first:
                return "dpo"
            elif "messages" in first:
                if any(m.get("role") == "assistant" for m in first["messages"]):
                    return "sft_conversational"
                else:
                    return "grpo"
            elif "prompt" in first and ("ground_truth" in first or "pattern" in first):
                return "grpo"
            elif "text" in first:
                return "sft_continued_pretraining"
            elif "title" in first and "content" in first:
                return "sft_continued_pretraining"

        if isinstance(sample_data, str):
            if "Q:" in sample_data and "A:" in sample_data:
                return "sft_qa"

        return "unknown"

    def _run_agent_analysis(
        self,
        file_path: str,
        file_type: str,
        sample_data: list[dict] | str,
        detected_format: str
    ) -> DatasetAnalysis:

        sample_str = json.dumps(sample_data, indent=2) if isinstance(sample_data, list) else sample_data
        if len(sample_str) > 3000:
            sample_str = sample_str[:3000] + "\n... [truncated]"

        rag_context = self._retrieve_context(detected_format, file_type)

        prompt = f"""Analyze this dataset for LLM fine-tuning and recommend the best training method.

FILE: {file_path}
TYPE: {file_type}
DETECTED FORMAT: {detected_format}

SAMPLE DATA (first 10 items/lines):
{sample_str}

RELEVANT KNOWLEDGE BASE CONTEXT:
{rag_context if rag_context else "No additional context retrieved."}

Based on the data structure and the knowledge base information, determine:
1. Most suitable training method (SFT, DPO, or GRPO)
2. Whether the data is properly formatted for that method
3. Any issues or missing fields based on expected formats
4. Suggestions for improvement

Respond in JSON format:
{{
    "recommended_method": "SFT" | "DPO" | "GRPO",
    "reasoning": "explanation of why this method is recommended",
    "issues": ["list", "of", "problems"],
    "suggestions": ["list", "of", "improvements"]
}}"""

        with langfuse_trace("dataset_analysis", input={"file_path": file_path, "file_type": file_type, "detected_format": detected_format, "rag_context_used": bool(rag_context)}) as trace:
            response = self.model.invoke(prompt)

            content = response.content if hasattr(response, 'content') else str(response)
            trace_llm_call(trace, "analyze_dataset", prompt, content, settings.LLM_MODEL_NAME)
            if trace:
                trace.update(output={"analysis": content})

        json_match = content
        if "```json" in content:
            json_match = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            json_match = content.split("```")[1].split("```")[0].strip()

        agent_result = json.loads(json_match)

        return DatasetAnalysis(
            file_path=file_path,
            file_type=file_type,
            sample_data=sample_data[:5] if isinstance(sample_data, list) else sample_data[:500],
            detected_format=detected_format,
            recommended_method=agent_result.get("recommended_method", "SFT"),
            reasoning=agent_result.get("reasoning", ""),
            issues=agent_result.get("issues", []),
            suggestions=agent_result.get("suggestions", []),
        )

    def generate_report(self, analysis: DatasetAnalysis) -> str:
        report_lines = [
            "<h3>Dataset Analysis Report</h3>",
            f"<p><b>File:</b> {Path(analysis['file_path']).name}</p>",
            f"<p><b>Type:</b> {analysis['file_type']}</p>",
            f"<p><b>Detected Format:</b> {analysis['detected_format']}</p>",
            "<hr>",
            f"<h4>Recommendation: <span style='color:#10b981'>{analysis['recommended_method']}</span></h4>",
            f"<p>{analysis['reasoning']}</p>",
        ]

        if analysis["issues"]:
            report_lines.append("<h4 style='color:#ef4444'>Issues Found:</h4><ul>")
            for issue in analysis["issues"]:
                report_lines.append(f"<li>{issue}</li>")
            report_lines.append("</ul>")

        if analysis["suggestions"]:
            report_lines.append("<h4 style='color:#f59e0b'>Suggestions:</h4><ul>")
            for suggestion in analysis["suggestions"]:
                report_lines.append(f"<li>{suggestion}</li>")
            report_lines.append("</ul>")

        return "\n".join(report_lines)


def analyze_dataset(file_path: str) -> DatasetAnalysis:
    analyzer = DatasetAnalyzer()
    return analyzer.analyze(file_path)


def get_training_recommendation(file_path: str) -> tuple[str, str]:
    analyzer = DatasetAnalyzer()
    analysis = analyzer.analyze(file_path)
    report_html = analyzer.generate_report(analysis)
    return analysis["recommended_method"], report_html
