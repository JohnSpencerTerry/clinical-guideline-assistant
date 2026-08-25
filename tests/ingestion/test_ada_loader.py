import pytest

from cga.ingestion.loaders import ada
from cga.ingestion.loaders.base import SourceFilesMissing


def test_parse_section_builds_metadata():
    doc = ada.parse_section(
        "  Some extracted section text.  ",
        section_number=2,
        title="Diagnosis and Classification of Diabetes",
        url="https://doi.org/10.2337/dc26-s002",
    )
    assert doc.page_content == "Some extracted section text."
    assert doc.metadata == {
        "source": "ADA",
        "section_number": 2,
        "title": "Diagnosis and Classification of Diabetes",
        "url": "https://doi.org/10.2337/dc26-s002",
    }


def test_load_raises_when_files_missing(tmp_path):
    with pytest.raises(SourceFilesMissing, match="section_01.pdf"):
        ada.load(raw_dir=tmp_path)
