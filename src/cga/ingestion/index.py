"""Builds/loads the per-source vector indices (ADA / NICE).

Indexed separately per source since the disagreement-detection flow needs to
retrieve from each source independently before comparing.
"""

import shutil
from functools import cache
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from cga.config import settings
from cga.ingestion.chunking import chunk_documents
from cga.ingestion.embeddings import get_embeddings
from cga.ingestion.loaders import ada, nice


def _persist_dir(source: str) -> Path:
    return Path(settings.vector_store_dir) / source


def build_index(source: str, documents: list[Document]) -> Chroma:
    persist_dir = _persist_dir(source)
    if persist_dir.exists():
        shutil.rmtree(persist_dir)

    return Chroma.from_documents(
        documents=documents,
        embedding=get_embeddings(),
        collection_name=source,
        persist_directory=str(persist_dir),
    )


@cache
def load_index(source: str) -> Chroma:
    persist_dir = _persist_dir(source)
    if not persist_dir.exists():
        raise FileNotFoundError(
            f"No index found at {persist_dir}. Run `uv run python -m cga.ingestion.index` first."
        )

    return Chroma(
        collection_name=source,
        persist_directory=str(persist_dir),
        embedding_function=get_embeddings(),
    )


def build_all() -> dict[str, Chroma]:
    indices = {}
    for source, loader in (("ada", ada.load), ("nice", nice.load)):
        chunks = chunk_documents(loader())
        indices[source] = build_index(source, chunks)
        print(f"{source}: {len(chunks)} chunks indexed at {_persist_dir(source)}")
    return indices


if __name__ == "__main__":
    build_all()
