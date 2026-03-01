"""Tests for LLM fine-tuning pipeline components."""

from pathlib import Path
from typing import Generator

import pytest

from src.llm.dataset_converter import (
    convert_to_dpo_format,
    convert_to_grpo_format,
    convert_to_sft_format,
    convert_txt_to_sft,
)


@pytest.fixture
def temp_txt_line_by_line(tmp_path: Path) -> Generator[str, None, None]:
    content = "First training line.\nSecond training line.\nThird training line.\n"
    txt_path = tmp_path / "test_lines.txt"
    txt_path.write_text(content)
    yield str(txt_path)


@pytest.fixture
def temp_txt_qa_pairs(tmp_path: Path) -> Generator[str, None, None]:
    content = """Q: What is machine learning?
A: Machine learning is a subset of artificial intelligence that enables systems to learn from data.

Q: What is deep learning?
A: Deep learning is a type of machine learning using neural networks with multiple layers.

Q: What is NLP?
A: Natural Language Processing is a field focused on interactions between computers and human language.
"""
    txt_path = tmp_path / "test_qa.txt"
    txt_path.write_text(content)
    yield str(txt_path)


@pytest.fixture
def temp_pdf(tmp_path: Path) -> Generator[str, None, None]:
    try:
        from pypdf import PdfWriter

        pdf_path = tmp_path / "test.pdf"
        writer = PdfWriter()
        writer.add_blank_page(width=200, height=200)
        with open(pdf_path, "wb") as f:
            writer.write(f)
        yield str(pdf_path)
    except ImportError:
        pytest.skip("pypdf not available")


def test_convert_txt_line_by_line(temp_txt_line_by_line: str) -> None:
    result = convert_txt_to_sft(temp_txt_line_by_line, format_type="line_by_line")
    assert len(result) == 3
    assert result[0] == {"text": "First training line."}
    assert result[1] == {"text": "Second training line."}
    assert result[2] == {"text": "Third training line."}


def test_convert_txt_qa_pairs(temp_txt_qa_pairs: str) -> None:
    result = convert_txt_to_sft(temp_txt_qa_pairs, format_type="qa_pairs")
    assert len(result) == 3
    assert "messages" in result[0]
    assert len(result[0]["messages"]) == 2
    assert result[0]["messages"][0]["role"] == "user"
    assert result[0]["messages"][1]["role"] == "assistant"


def test_convert_to_sft_format_txt(temp_txt_line_by_line: str) -> None:
    result = convert_to_sft_format(temp_txt_line_by_line)
    assert len(result) == 3
    assert "text" in result[0]


def test_convert_to_sft_format_missing_file(tmp_path: Path) -> None:
    missing = str(tmp_path / "nonexistent.txt")
    with pytest.raises(FileNotFoundError):
        convert_to_sft_format(missing)


def test_convert_to_sft_format_unsupported_type(tmp_path: Path) -> None:
    unsupported = tmp_path / "test.json"
    unsupported.write_text('{"key": "value"}')
    with pytest.raises(ValueError, match="Unsupported file type"):
        convert_to_sft_format(str(unsupported))


def test_convert_to_dpo_format(temp_txt_qa_pairs: str) -> None:
    result = convert_to_dpo_format(temp_txt_qa_pairs)
    assert len(result) == 3
    for item in result:
        assert "prompt" in item
        assert "chosen" in item
        assert "rejected" in item
        assert item["prompt"][0]["role"] == "user"
        assert item["chosen"][0]["role"] == "assistant"
        assert item["rejected"][0]["role"] == "assistant"


def test_convert_to_dpo_format_missing_file(tmp_path: Path) -> None:
    missing = str(tmp_path / "nonexistent.txt")
    with pytest.raises(FileNotFoundError):
        convert_to_dpo_format(missing)


def test_convert_to_dpo_format_unsupported_type(tmp_path: Path) -> None:
    unsupported = tmp_path / "test.pdf"
    unsupported.write_bytes(b"%PDF-1.4 fake pdf")
    with pytest.raises(ValueError, match="Unsupported file type for DPO"):
        convert_to_dpo_format(str(unsupported))


def test_convert_to_grpo_format(temp_txt_qa_pairs: str) -> None:
    result = convert_to_grpo_format(temp_txt_qa_pairs)
    assert len(result) == 3
    for item in result:
        assert "prompt" in item
        assert len(item["prompt"]) == 1
        assert item["prompt"][0]["role"] == "user"


def test_convert_to_grpo_format_missing_file(tmp_path: Path) -> None:
    missing = str(tmp_path / "nonexistent.txt")
    with pytest.raises(FileNotFoundError):
        convert_to_grpo_format(missing)


def test_convert_txt_unsupported_format(temp_txt_line_by_line: str) -> None:
    with pytest.raises(ValueError, match="Unsupported format_type"):
        convert_txt_to_sft(temp_txt_line_by_line, format_type="invalid_format")


def test_convert_to_grpo_format_unsupported_type(tmp_path: Path) -> None:
    unsupported = tmp_path / "test.json"
    unsupported.write_text('{"key": "value"}')
    with pytest.raises(ValueError, match="Unsupported file type for GRPO"):
        convert_to_grpo_format(str(unsupported))


def test_empty_txt_file(tmp_path: Path) -> None:
    empty = tmp_path / "empty.txt"
    empty.write_text("")
    result = convert_txt_to_sft(str(empty), format_type="line_by_line")
    assert len(result) == 0


def test_txt_with_blank_lines(tmp_path: Path) -> None:
    content = "Line one.\n\n\nLine two.\n   \nLine three.\n"
    txt_path = tmp_path / "blanks.txt"
    txt_path.write_text(content)
    result = convert_txt_to_sft(str(txt_path), format_type="line_by_line")
    assert len(result) == 3
