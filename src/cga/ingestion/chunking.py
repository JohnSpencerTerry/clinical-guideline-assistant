"""Content-aware chunking: split by size, not by hardcoded per-source rules.

Short Documents (NICE's already-atomic per-recommendation text) pass through
unchanged; long Documents (ADA's whole-section text) get recursively split,
with each chunk's metadata carrying the parent's metadata plus chunk_index/
chunk_count so citations can still point back to the parent section.
"""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(
    documents: list[Document], chunk_size: int = 1000, chunk_overlap: int = 150
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    result: list[Document] = []

    for doc in documents:
        if len(doc.page_content) <= chunk_size:
            result.append(doc)
            continue

        pieces = splitter.split_text(doc.page_content)
        for i, piece in enumerate(pieces):
            result.append(
                Document(
                    page_content=piece,
                    metadata={**doc.metadata, "chunk_index": i, "chunk_count": len(pieces)},
                )
            )

    return result
