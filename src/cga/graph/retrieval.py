"""Independent per-source retrieval against the ADA / NICE indices."""

from langchain_core.documents import Document

from cga.ingestion.index import load_index


def retrieve(source: str, question: str, *, k: int = 4) -> list[Document]:
    index = load_index(source)
    return index.similarity_search(question, k=k)
