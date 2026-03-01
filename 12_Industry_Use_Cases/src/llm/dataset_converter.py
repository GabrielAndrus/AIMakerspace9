"""Dataset conversion utilities for LLM fine-tuning."""

import re
from pathlib import Path

from pypdf import PdfReader


def convert_txt_to_sft(filepath: str, format_type: str = "line_by_line") -> list[dict]:
    """
    Convert a text file to SFT training format.

    Args:
        filepath: Path to the text file
        format_type: Conversion format, either 'line_by_line' or 'qa_pairs'

    Returns:
        List of training examples in SFT format

    Raises:
        ValueError: If format_type is not supported
    """
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    if format_type == "line_by_line":
        return _convert_txt_line_by_line(content)
    elif format_type == "qa_pairs":
        return _convert_txt_qa_pairs(content)
    else:
        raise ValueError(
            f"Unsupported format_type: {format_type}. "
            "Supported types are 'line_by_line' and 'qa_pairs'."
        )


def _convert_txt_line_by_line(content: str) -> list[dict]:
    """Convert text content where each non-empty line becomes a training example."""
    examples = []
    for line in content.split("\n"):
        stripped = line.strip()
        if stripped:
            examples.append({"text": stripped})
    return examples


def _convert_txt_qa_pairs(content: str) -> list[dict]:
    """Convert text content with Q: A: blocks into conversational format."""
    examples = []
    pattern = r"Q:\s*(.*?)\s*A:\s*(.*?)(?=Q:|$)"
    matches = re.findall(pattern, content, re.DOTALL)

    for question, answer in matches:
        question = question.strip()
        answer = answer.strip()
        if question and answer:
            examples.append(
                {
                    "messages": [
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": answer},
                    ]
                }
            )
    return examples


def convert_pdf_to_sft(filepath: str, chunk_size: int = 512) -> list[dict]:
    """
    Convert a PDF file to SFT training format with overlapping chunks.

    Args:
        filepath: Path to the PDF file
        chunk_size: Maximum characters per chunk

    Returns:
        List of training examples with text chunks
    """
    reader = PdfReader(filepath)
    full_text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            full_text += page_text + "\n"

    return _create_overlapping_chunks(full_text, chunk_size)


def _create_overlapping_chunks(text: str, chunk_size: int) -> list[dict]:
    """Split text into overlapping chunks."""
    examples = []
    stride = chunk_size // 2

    if stride == 0:
        stride = 1

    text = text.strip()
    if not text:
        return examples

    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()

        if chunk:
            examples.append({"text": chunk})

        start += stride
        if end >= len(text):
            break

    return examples


def convert_to_sft_format(filepath: str) -> list[dict]:
    """
    Auto-detect file type and convert to SFT format.

    Args:
        filepath: Path to the input file (.txt or .pdf)

    Returns:
        List of training examples in SFT format

    Raises:
        ValueError: If file type is not supported
        FileNotFoundError: If the file does not exist
    """
    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    suffix = path.suffix.lower()

    if suffix == ".txt":
        return convert_txt_to_sft(filepath)
    elif suffix == ".pdf":
        return convert_pdf_to_sft(filepath)
    else:
        raise ValueError(f"Unsupported file type: {suffix}. Supported types are .txt and .pdf.")


def convert_to_dpo_format(filepath: str) -> list[dict]:
    """
    Convert a QA format TXT file to DPO training format with preference pairs.

    Args:
        filepath: Path to the input TXT file with QA pairs

    Returns:
        List of training examples in DPO format with 'prompt', 'chosen', and 'rejected' fields

    Raises:
        ValueError: If file type is not supported
        FileNotFoundError: If the file does not exist
    """
    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    suffix = path.suffix.lower()
    if suffix != ".txt":
        raise ValueError(f"Unsupported file type for DPO: {suffix}. Only .txt files are supported.")

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    examples = []
    pattern = r"Q:\s*(.*?)\s*A:\s*(.*?)(?=Q:|$)"
    matches = re.findall(pattern, content, re.DOTALL)

    for question, answer in matches:
        question = question.strip()
        answer = answer.strip()
        if question and answer:
            rejected = _generate_rejected_response(answer)
            examples.append(
                {
                    "prompt": [{"role": "user", "content": question}],
                    "chosen": [{"role": "assistant", "content": answer}],
                    "rejected": [{"role": "assistant", "content": rejected}],
                }
            )
    return examples


def _generate_rejected_response(good_answer: str) -> str:
    """Generate a deliberately worse response for DPO training."""
    words = good_answer.split()
    if len(words) > 10:
        truncated = " ".join(words[:5])
        return f"{truncated}... [incomplete response]"
    elif len(words) > 3:
        return words[0]
    else:
        return "I don't know."


def convert_to_grpo_format(filepath: str) -> list[dict]:
    """
    Convert a QA format TXT file to GRPO training format with prompts only.

    Args:
        filepath: Path to the input TXT file with QA pairs

    Returns:
        List of training examples in GRPO format with 'prompt' field only

    Raises:
        ValueError: If file type is not supported
        FileNotFoundError: If the file does not exist
    """
    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    suffix = path.suffix.lower()
    if suffix != ".txt":
        raise ValueError(
            f"Unsupported file type for GRPO: {suffix}. Only .txt files are supported."
        )

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    examples = []
    pattern = r"Q:\s*(.*?)\s*A:\s*(.*?)(?=Q:|$)"
    matches = re.findall(pattern, content, re.DOTALL)

    for question, _ in matches:
        question = question.strip()
        if question:
            examples.append({"prompt": [{"role": "user", "content": question}]})
    return examples
