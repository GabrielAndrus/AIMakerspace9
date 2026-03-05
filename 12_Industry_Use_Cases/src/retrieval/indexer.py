import os
from pathlib import Path
from .chunker import chunk_documents
from .embeddings import embed_documents, get_embedding_model
from .qdrant_client import QdrantRetriever


def build_knowledge_base(
    knowledge_base_dir: str = "data/knowledge_base", recreate_collection: bool = False
) -> dict[str, int]:
    """Build the knowledge base by indexing all documents into Qdrant."""

    print(f"Loading and chunking documents from {knowledge_base_dir}...")
    chunks = chunk_documents(knowledge_base_dir)
    print(f"Created {len(chunks)} chunks")

    if not chunks:
        print("No chunks to index")
        return {"total": 0}

    print("Generating embeddings...")
    texts = [chunk["content"] for chunk in chunks]
    model = get_embedding_model()
    embeddings = embed_documents(texts)
    vector_size = len(embeddings[0]) if embeddings else 1536

    print("Connecting to Qdrant...")
    retriever = QdrantRetriever(vector_size=vector_size)

    print("Creating collection...")
    retriever.create_collection(recreate=recreate_collection)

    print("Indexing documents...")
    ids = [chunk["id"] for chunk in chunks]
    payloads = [
        {"title": chunk["title"], "content": chunk["content"], "source": chunk["source"]}
        for chunk in chunks
    ]

    retriever.add_documents(ids, embeddings, payloads)

    print(f"Successfully indexed {len(chunks)} documents")

    return {"total": len(chunks)}


if __name__ == "__main__":
    build_knowledge_base(recreate_collection=True)
