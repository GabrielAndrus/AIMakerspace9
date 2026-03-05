"""Dataset analyzer agent using DeepAgents pattern.

This agent analyzes uploaded datasets and recommends appropriate training methods
(SFT, DPO, or GRPO) based on data structure and content.
"""

import json
from pathlib import Path
from typing import Any, TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langfuse import Langfuse, get_client
from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler

from src.config import settings


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
    """Analyze datasets and recommend training methods using DeepAgents pattern."""

    def __init__(self, model_name: str = "openai/gpt-oss-120b"):
        self.model_name = model_name
        self.model = None
        self.llm_available = False
        self.langfuse_handler = None
        self._setup_model()
        self._setup_langfuse()

    def _setup_model(self):
        try:
            base_url = settings.LLM_INFERENCE_URL
            api_key = settings.LLM_INFERENCE_KEY
            
            self.model = init_chat_model(
                self.model_name,
                model_provider="openai",
                config={
                    "base_url": base_url,
                    "api_key": api_key,
                    "temperature": 0.1,
                },
            )
            self.llm_available = True
        except Exception:
            self.llm_available = False

    def _setup_langfuse(self):
        if settings.LANGFUSE_PUBLIC_KEY and settings.LANGFUSE_SECRET_KEY:
            Langfuse(
                public_key=settings.LANGFUSE_PUBLIC_KEY,
                secret_key=settings.LANGFUSE_SECRET_KEY,
                host=settings.LANGFUSE_HOST,
            )
            self.langfuse_handler = LangfuseCallbackHandler()

    def analyze(self, file_path: str) -> DatasetAnalysis:
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        file_type = self._detect_file_type(path)
        sample_data = self._extract_sample(path, file_type)
        detected_format = self._detect_format(sample_data, file_type)
        
        agent_analysis = self._run_agent_analysis(file_path, file_type, sample_data, detected_format)
        
        if self.langfuse_handler:
            try:
                get_client().flush()
            except Exception:
                pass
        
        return agent_analysis

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
        
        if not self.llm_available:
            return self._rule_based_analysis(file_path, file_type, sample_data, detected_format)
        
        prompt = f"""Analyze this dataset for LLM fine-tuning and recommend the best training method.

FILE: {file_path}
TYPE: {file_type}
DETECTED FORMAT: {detected_format}

SAMPLE DATA (first 10 items/lines):
{sample_str}

Based on the data structure, determine:
1. Most suitable training method (SFT, DPO, or GRPO)
2. Whether the data is properly formatted for that method
3. Any issues or missing fields
4. Suggestions for improvement

Respond in JSON format:
{{
    "recommended_method": "SFT" | "DPO" | "GRPO",
    "reasoning": "explanation of why this method is recommended",
    "issues": ["list", "of", "problems"],
    "suggestions": ["list", "of", "improvements"]
}}"""

        try:
            callbacks = [self.langfuse_handler] if self.langfuse_handler else None
            response = self.model.invoke(prompt, config={"callbacks": callbacks} if callbacks else {})
            
            content = response.content if hasattr(response, 'content') else str(response)
            
            json_match = content
            if "```json" in content:
                json_match = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                json_match = content.split("```")[1].split("```")[0].strip()
            
            agent_result = json.loads(json_match)
        except Exception:
            return self._rule_based_analysis(file_path, file_type, sample_data, detected_format)

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

    def _rule_based_analysis(
        self,
        file_path: str,
        file_type: str,
        sample_data: list[dict] | str,
        detected_format: str
    ) -> DatasetAnalysis:
        
        issues = []
        suggestions = []
        
        if detected_format == "dpo":
            recommended_method = "DPO"
            reasoning = "Dataset contains prompt/chosen/rejected fields, which is the standard format for Direct Preference Optimization (DPO). This method trains models to prefer better responses over worse ones."
            
            if isinstance(sample_data, list) and len(sample_data) > 0:
                first = sample_data[0]
                if not all(k in first for k in ["prompt", "chosen", "rejected"]):
                    issues.append("Missing required fields (prompt, chosen, rejected)")
        
        elif detected_format == "grpo":
            recommended_method = "GRPO"
            reasoning = "Dataset contains ground_truth or pattern validation fields, suitable for Group Relative Policy Optimization (GRPO). This method uses reward signals based on correctness or pattern matching."
            
            if isinstance(sample_data, list) and len(sample_data) > 0:
                first = sample_data[0]
                if "ground_truth" not in first and "pattern" not in first:
                    issues.append("Missing ground_truth or pattern field for validation")
        
        elif detected_format in ["sft_conversational", "sft_continued_pretraining", "sft_qa"]:
            recommended_method = "SFT"
            
            if detected_format == "sft_conversational":
                reasoning = "Dataset contains conversational messages with assistant responses, ideal for Supervised Fine-Tuning (SFT). This teaches the model to follow conversation patterns."
            elif detected_format == "sft_continued_pretraining":
                reasoning = "Dataset contains text/title/content pairs, suitable for continued pre-training or SFT. This helps the model learn domain-specific knowledge."
            else:
                reasoning = "Dataset uses Q:/A: format, a simple structure for question-answer pairs. Recommended for Supervised Fine-Tuning (SFT) to teach the model factual knowledge."
            
            if isinstance(sample_data, list) and len(sample_data) > 0:
                first = sample_data[0]
                if detected_format == "sft_conversational" and not any(m.get("role") == "assistant" for m in first.get("messages", [])):
                    issues.append("Conversational data missing assistant responses")
        
        else:
            recommended_method = "SFT"
            reasoning = "Format could not be automatically detected. Defaulting to SFT (Supervised Fine-Tuning) as the most versatile training method."
            suggestions.append("Verify dataset structure matches expected format (messages, prompt/chosen/rejected, or Q:/A:)")
            suggestions.append("Consider converting to a standard format if this doesn't match")
        
        if not issues:
            suggestions.append("Dataset structure looks good for the recommended method")
        
        return DatasetAnalysis(
            file_path=file_path,
            file_type=file_type,
            sample_data=sample_data[:5] if isinstance(sample_data, list) else sample_data[:500],
            detected_format=detected_format,
            recommended_method=recommended_method,
            reasoning=reasoning,
            issues=issues,
            suggestions=suggestions,
        )

    def _fallback_recommendation(self, detected_format: str) -> str:
        format_to_method = {
            "dpo": "DPO",
            "grpo": "GRPO",
            "sft_conversational": "SFT",
            "sft_continued_pretraining": "SFT",
            "sft_qa": "SFT",
        }
        return format_to_method.get(detected_format, "SFT")

    def generate_report(self, analysis: DatasetAnalysis) -> str:
        report_lines = [
            f"<h3>Dataset Analysis Report</h3>",
            f"<p><b>File:</b> {Path(analysis['file_path']).name}</p>",
            f"<p><b>Type:</b> {analysis['file_type']}</p>",
            f"<p><b>Detected Format:</b> {analysis['detected_format']}</p>",
            f"<hr>",
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
    analysis = analyze_dataset(file_path)
    
    report_lines = [
        f"<h3>Dataset Analysis Report</h3>",
        f"<p><b>File:</b> {Path(analysis['file_path']).name}</p>",
        f"<p><b>Type:</b> {analysis['file_type']}</p>",
        f"<p><b>Detected Format:</b> {analysis['detected_format']}</p>",
        f"<hr>",
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
    
    return analysis["recommended_method"], "\n".join(report_lines)