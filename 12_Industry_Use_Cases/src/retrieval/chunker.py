import json
from pathlib import Path
from typing import Any
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(
    knowledge_base_dir: str = "data/knowledge_base", chunk_size: int = 500, chunk_overlap: int = 100
) -> list[dict[str, Any]]:
    """Load and chunk documents from knowledge base JSONL files."""

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n\n", "\n", ". ", " ", ""]
    )

    all_chunks = []
    kb_path = Path(knowledge_base_dir)

    if not kb_path.exists():
        raise FileNotFoundError(f"Knowledge base not found at {knowledge_base_dir}")

    for jsonl_file in kb_path.glob("*.jsonl"):
        with open(jsonl_file, "r") as f:
            for line_num, line in enumerate(f):
                doc = json.loads(line.strip())
                text = doc.get("content", "")
                title = doc.get("title", "")
                source = doc.get("source", "")

                if text:
                    chunks = text_splitter.split_text(text)
                    for chunk_idx, chunk_text in enumerate(chunks):
                        all_chunks.append(
                            {
                                "id": f"{jsonl_file.stem}_{line_num}_{chunk_idx}",
                                "title": title,
                                "content": chunk_text,
                                "source": source,
                                "original_doc": doc,
                            }
                        )

    return all_chunks
