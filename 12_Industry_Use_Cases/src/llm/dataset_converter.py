"""Dataset conversion utilities for LLM fine-tuning."""

import json
import re
import warnings
from pathlib import Path

try:
    import fitz
except ImportError:
    fitz = None


# ============================================================================
# DPO Validation Error Messages
# ============================================================================

DPO_VALIDATION_ERRORS = {
    "missing_rejected": """
<b>Your DPO dataset is missing 'rejected' responses.</b>

DPO (Direct Preference Optimization) requires preference pairs that reflect
<b>actual human judgments</b> about which response is better. You cannot generate
rejected responses automatically - they must come from real preference data.

<h3>REQUIRED FORMAT (JSONL preferred):</h3>
<code>{"prompt": [{"role": "user", "content": "What is 2+2?"}], "chosen": [{"role": "assistant", "content": "The answer is 4."}], "rejected": [{"role": "assistant", "content": "I don't know."}]}</code>

<h3>HOW TO CREATE DPO DATA:</h3>
<ol>
  <li><b>Human annotation:</b> Have humans rate/compare model outputs</li>
  <li><b>LLM-as-judge:</b> Use a stronger model to evaluate responses</li>
  <li><b>Existing datasets:</b> Download from HuggingFace (e.g., Anthropic HH-RLHF, OpenAssistant)</li>
  <li><b>Custom collection:</b> Collect prompts + 2+ responses with human rankings</li>
</ol>

<h3>REFERENCE:</h3>
<a href="https://huggingface.co/docs/trl/dpo_trainer#dataset-format" target="_blank">HuggingFace TRL DPO Documentation</a>
""",
    "missing_chosen": """
<b>Your DPO dataset is missing 'chosen' responses.</b>

Each example needs both a 'chosen' (preferred) and 'rejected' (disfavored)
response. The chosen response should be the one a human would prefer.

<h3>REQUIRED FORMAT:</h3>
<code>{"prompt": [...], "chosen": [...], "rejected": [...]}</code>
""",
    "missing_prompt": """
<b>Your DPO dataset is missing 'prompt' field.</b>

Each example must include a prompt that both the chosen and rejected responses
answer.

<h3>REQUIRED FORMAT:</h3>
<code>{"prompt": [...], "chosen": [...], "rejected": [...]}</code>
""",
    "invalid_format": """
<b>Your DPO file is not in the expected format.</b>

<b>Expected:</b> JSONL (JSON Lines) with one JSON object per line
Each object must have: 'prompt', 'chosen', 'rejected'

<h3>Example (one line):</h3>
<code>{"prompt": [{"role": "user", "content": "Hello!"}], "chosen": [{"role": "assistant", "content": "Hi there!"}], "rejected": [{"role": "assistant", "content": "yo"}]}</code>

<h3>Alternative (deprecated):</h3>
TXT format with Q:/A: blocks is still supported but will be deprecated in future versions.
""",
    "empty_file": """
<b>Your DPO file is empty or contains no valid examples.</b>

DPO training requires at least several hundred preference pairs for
meaningful results. Consider:
<ul>
  <li>Using an existing DPO dataset from HuggingFace</li>
  <li>Collecting more preference data</li>
  <li>Checking if your file format is correct</li>
</ul>

<h3>SOURCE FOR DPO DATASETS:</h3>
<a href="https://huggingface.co/datasets?sort=downloads&search=dpo" target="_blank">HuggingFace DPO Datasets</a>
""",
    "identical_chosen_rejected": """
<b>Invalid data: 'chosen' and 'rejected' responses are identical.</b>

DPO requires that the chosen response is actually better than the rejected.
If they are identical, there's no preference signal for the model to learn from.

<h3>Common causes:</h3>
<ul>
  <li>Copied the same response to both fields by mistake</li>
  <li>Dataset generation bug that didn't differentiate responses</li>
  <li>TXT format conversion (which creates synthetic rejected data)</li>
</ul>

<h3>How to fix:</h3>
Ensure each example has two <b>different</b> responses where the 'chosen' one is
actually preferred by humans.
""",
    "unsupported_format": """
<b>Unsupported file format for DPO.</b>

The DPO converter supports:
<ul>
  <li><b>.jsonl</b> (recommended): JSON Lines format with proper DPO structure</li>
  <li><b>.txt</b> (deprecated): Q:/A: format - will create synthetic rejected data</li>
</ul>

Please use JSONL format for real DPO training with actual preference pairs.
""",
}


# ============================================================================
# SFT Conversion Functions
# ============================================================================


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
    if fitz is None:
        raise ImportError("PyMuPDF (fitz) is required for PDF processing. Install with: pip install pymupdf")

    doc = fitz.open(filepath)
    full_text = ""

    for page in doc:
        page_text = page.get_text()
        if page_text:
            full_text += page_text + "\n"

    doc.close()

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


def _convert_jsonl_to_sft(filepath: str) -> list[dict]:
    """Convert JSONL file to SFT training format.
    
    Handles various JSONL formats:
    - {"text": "..."} - direct text
    - {"messages": [...]} - conversational format  
    - {"title": "...", "content": "..."} - knowledge base format (converted to text)
    
    Args:
        filepath: Path to the JSONL file
        
    Returns:
        List of training examples in SFT format
    """
    examples = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                example = json.loads(line)
                
                # Handle different formats
                if "text" in example:
                    examples.append({"text": example["text"]})
                elif "messages" in example:
                    examples.append({"messages": example["messages"]})
                elif "title" in example and "content" in example:
                    text = f"{example['title']}\n\n{example['content']}"
                    examples.append({"text": text})
                else:
                    warnings.warn(f"Line {line_num}: Unknown format, skipping")
            except json.JSONDecodeError as e:
                warnings.warn(f"Line {line_num}: Invalid JSON - {e}")
    return examples


def convert_to_sft_format(filepath: str) -> list[dict]:
    """
    Auto-detect file type and convert to SFT format.

    Args:
        filepath: Path to the input file (.txt, .pdf, or .jsonl)

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
    elif suffix == ".jsonl":
        return _convert_jsonl_to_sft(filepath)
    else:
        raise ValueError(f"Unsupported file type: {suffix}. Supported types are .txt, .pdf, and .jsonl.")


# ============================================================================
# DPO Conversion Functions
# ============================================================================


def validate_dpo_format(examples: list[dict]) -> tuple[bool, str]:
    """Validate DPO dataset format.

    Args:
        examples: List of potential DPO examples

    Returns:
        Tuple of (is_valid, error_message)

    Error messages are verbose and actionable in HTML format.
    """
    if not examples:
        return False, DPO_VALIDATION_ERRORS["empty_file"]

    for i, example in enumerate(examples):
        # Check required fields
        if "prompt" not in example:
            return (
                False,
                f"<b>Example {i}:</b> Missing 'prompt' field.<br><br>{DPO_VALIDATION_ERRORS['missing_prompt']}",
            )

        if "chosen" not in example:
            return (
                False,
                f"<b>Example {i}:</b> Missing 'chosen' field.<br><br>{DPO_VALIDATION_ERRORS['missing_chosen']}",
            )

        if "rejected" not in example:
            return (
                False,
                f"<b>Example {i}:</b> Missing 'rejected' field.<br><br>{DPO_VALIDATION_ERRORS['missing_rejected']}",
            )

        # Validate that chosen != rejected (common mistake)
        if example.get("chosen") == example.get("rejected"):
            return (
                False,
                f"<b>Example {i}:</b> 'chosen' and 'rejected' are identical.<br><br>{DPO_VALIDATION_ERRORS['identical_chosen_rejected']}",
            )

    return True, ""


def convert_to_dpo_format(filepath: str) -> list[dict]:
    """Convert a file to DPO training format with validation.

    Args:
        filepath: Path to the input TXT/JSONL file

    Returns:
        List of training examples in DPO format

    Raises:
        ValueError: If validation fails with detailed error message
        FileNotFoundError: If the file does not exist
    """
    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    suffix = path.suffix.lower()

    # Support both TXT (legacy) and JSONL (preferred)
    if suffix == ".jsonl":
        examples = _parse_jsonl_file(filepath)
    elif suffix == ".txt":
        # Legacy format - warn that this is limited
        warnings.warn(
            "TXT format for DPO is deprecated. Please use JSONL format "
            "for proper preference data. See documentation for details.",
            DeprecationWarning,
            stacklevel=2,
        )
        examples = _convert_txt_to_dpo_legacy(filepath)
    else:
        raise ValueError(DPO_VALIDATION_ERRORS["unsupported_format"])

    # Validate the dataset
    is_valid, error_message = validate_dpo_format(examples)
    if not is_valid:
        raise ValueError(error_message)

    return examples


def _parse_jsonl_file(filepath: str) -> list[dict]:
    """Parse a JSONL file into a list of dicts.

    Each line in the file should be a valid JSON object representing one DPO example.

    Args:
        filepath: Path to the JSONL file

    Returns:
        List of parsed examples

    Raises:
        ValueError: If JSON parsing fails with detailed error message
    """
    examples = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                example = json.loads(line)
                examples.append(example)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"<b>Invalid JSON on line {line_num}:</b> {e}<br><br>"
                    "Each line must be a valid JSON object with 'prompt', 'chosen', and 'rejected' fields.<br><br>"
                    f"{DPO_VALIDATION_ERRORS['invalid_format']}"
                )
    return examples


def _convert_txt_to_dpo_legacy(filepath: str) -> list[dict]:
    """Legacy TXT to DPO conversion - DEPRECATED.

    This creates synthetic rejected responses which are NOT suitable
    for real DPO training. Included for backward compatibility only.

    WARNING: This function will create synthetic/fake rejected responses.
    For meaningful DPO training, use JSONL format with real preference pairs.

    Args:
        filepath: Path to the TXT file with Q:/A: format

    Returns:
        List of DPO examples with synthetic rejected responses
    """
    warnings.warn(
        "Converting TXT to DPO format creates SYNTHETIC rejected responses. "
        "This is NOT suitable for meaningful DPO training. "
        "Please use JSONL format with actual preference pairs from human annotation or LLM-as-judge.",
        UserWarning,
        stacklevel=2,
    )

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    examples = []
    pattern = r"Q:\s*(.*?)\s*A:\s*(.*?)(?=Q:|$)"
    matches = re.findall(pattern, content, re.DOTALL)

    for question, answer in matches:
        question = question.strip()
        answer = answer.strip()

        if not (question and answer):
            continue

        # Generate synthetic rejected - but warn!
        words = answer.split()
        if len(words) > 10:
            rejected = " ".join(words[:3]) + "... [incomplete]"
        else:
            rejected = "I don't know."

        examples.append(
            {
                "prompt": [{"role": "user", "content": question}],
                "chosen": [{"role": "assistant", "content": answer}],
                "rejected": [{"role": "assistant", "content": rejected}],
            }
        )

    return examples


# ============================================================================
# GRPO Conversion Functions
# ============================================================================


def convert_to_grpo_format(filepath: str) -> list[dict]:
    """
    Convert a file to GRPO training format.

    Supports JSONL (preferred) and TXT formats.
    JSONL should include 'prompt' field with conversation history
    and optionally 'ground_truth' or 'pattern' for reward functions.

    Args:
        filepath: Path to the input file (.jsonl or .txt)

    Returns:
        List of training examples in GRPO format

    Raises:
        ValueError: If file type is not supported or format invalid
        FileNotFoundError: If the file does not exist
    """
    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    suffix = path.suffix.lower()

    if suffix == ".jsonl":
        return _parse_grpo_jsonl_file(filepath)
    elif suffix == ".txt":
        warnings.warn(
            "TXT format for GRPO does not include ground truth. "
            "Use JSONL format with 'ground_truth' field for math reward.",
            UserWarning,
            stacklevel=2,
        )
        return _convert_txt_to_grpo_legacy(filepath)
    else:
        raise ValueError(
            f"Unsupported file type for GRPO: {suffix}. Use .jsonl or .txt files."
        )


def _parse_grpo_jsonl_file(filepath: str) -> list[dict]:
    """Parse a JSONL file into GRPO format.

    Expected fields:
    - prompt: Required. List of messages [{"role": "user", "content": "..."}]
    - ground_truth: Optional. For math reward template
    - pattern: Optional. For format_check reward template

    Args:
        filepath: Path to the JSONL file

    Returns:
        List of GRPO examples
    """
    examples = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                example = json.loads(line)
                if "prompt" not in example:
                    warnings.warn(
                        f"Line {line_num}: Missing 'prompt' field, skipping"
                    )
                    continue
                examples.append(example)
            except json.JSONDecodeError as e:
                warnings.warn(f"Line {line_num}: Invalid JSON - {e}")
    return examples


def _convert_txt_to_grpo_legacy(filepath: str) -> list[dict]:
    """Legacy TXT to GRPO conversion.

    Args:
        filepath: Path to the TXT file with Q:/A: format

    Returns:
        List of GRPO examples (without ground truth)
    """
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
