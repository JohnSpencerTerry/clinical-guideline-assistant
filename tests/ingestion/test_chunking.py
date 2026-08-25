from langchain_core.documents import Document

from cga.ingestion.chunking import chunk_documents


def test_short_document_passes_through_unchanged():
    doc = Document(page_content="Short text.", metadata={"source": "NICE", "recommendation_number": "1.1"})

    result = chunk_documents([doc], chunk_size=1000, chunk_overlap=150)

    assert len(result) == 1
    assert result[0] is doc
    assert "chunk_index" not in result[0].metadata


def test_long_document_splits_and_carries_parent_metadata():
    long_text = "Sentence about diabetes care. " * 100  # well over chunk_size
    doc = Document(page_content=long_text, metadata={"source": "ADA", "section_number": 3})

    result = chunk_documents([doc], chunk_size=200, chunk_overlap=20)

    assert len(result) > 1
    for i, chunk in enumerate(result):
        assert chunk.metadata["source"] == "ADA"
        assert chunk.metadata["section_number"] == 3
        assert chunk.metadata["chunk_index"] == i
        assert chunk.metadata["chunk_count"] == len(result)
        assert len(chunk.page_content) <= 200


def test_mixed_batch_only_splits_long_documents():
    short_doc = Document(page_content="Short.", metadata={"source": "NICE"})
    long_doc = Document(page_content="Long content. " * 100, metadata={"source": "ADA"})

    result = chunk_documents([short_doc, long_doc], chunk_size=200, chunk_overlap=20)

    assert result[0] is short_doc
    assert len(result) > 2
