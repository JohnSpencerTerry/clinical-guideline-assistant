import pytest

from cga.ingestion.loaders import nice
from cga.ingestion.loaders.base import SourceFilesMissing

SAMPLE_TEXT = (
    "1.9 Insulin\n"
    "Some intro prose that is not numbered.\n"
    "1.9.1 Offer insulin therapy to adults with type 2 diabetes.\n"
    "Continued text for the same recommendation.\n"
    "1.9.2 Consider a different insulin regimen if needed.\n"
    "\f"
    "1.9.3 Review the person's insulin regimen annually.\n"
)


def test_split_recommendations_produces_one_document_per_number():
    docs = nice.split_recommendations(SAMPLE_TEXT)
    numbers = [d.metadata["recommendation_number"] for d in docs]
    assert numbers == ["1.9", "1.9.1", "1.9.2", "1.9.3"]


def test_split_recommendations_derives_section_from_parent_number():
    docs = nice.split_recommendations(SAMPLE_TEXT)
    by_number = {d.metadata["recommendation_number"]: d for d in docs}
    assert by_number["1.9"].metadata["section"] == "1"
    assert by_number["1.9.3"].metadata["section"] == "1.9"


def test_split_recommendations_tracks_page_across_page_breaks():
    docs = nice.split_recommendations(SAMPLE_TEXT)
    by_number = {d.metadata["recommendation_number"]: d for d in docs}
    assert by_number["1.9.2"].metadata["page"] == 0
    assert by_number["1.9.3"].metadata["page"] == 1


def test_split_recommendations_joins_continuation_lines():
    docs = nice.split_recommendations(SAMPLE_TEXT)
    by_number = {d.metadata["recommendation_number"]: d for d in docs}
    assert "Continued text for the same recommendation." in by_number["1.9.1"].page_content


def test_split_recommendations_skips_table_of_contents_lines():
    toc_text = (
        "1.9 Insulin ................................................................ 14\n"
        "1.9.1 Offer insulin therapy to adults with type 2 diabetes.\n"
    )
    docs = nice.split_recommendations(toc_text)
    numbers = [d.metadata["recommendation_number"] for d in docs]
    assert numbers == ["1.9.1"]


def test_load_raises_when_file_missing(tmp_path):
    with pytest.raises(SourceFilesMissing, match="ng28.pdf"):
        nice.load(raw_dir=tmp_path)
