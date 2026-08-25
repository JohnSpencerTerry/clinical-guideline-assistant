"""Loader for NICE NG28 (Type 2 diabetes in adults: management), split by numbered recommendation.

NICE uses hierarchical dotted numbering (e.g. "1.9", "1.9.3"). Rather than
trying to detect prose heading lines, `section` is derived directly from the
numbering itself: a recommendation's section is its parent number (e.g.
"1.9.3"'s section is "1.9").
"""

import re
from pathlib import Path

from langchain_core.documents import Document

from cga.ingestion.loaders.base import RAW_DATA_DIR, SourceFilesMissing, extract_text_from_pdf

NICE_PDF_URL = "https://www.nice.org.uk/guidance/ng28/resources/type-2-diabetes-in-adults-management-pdf-1837338615493"

_RECOMMENDATION_RE = re.compile(r"^(\d+\.\d+(?:\.\d+)*)\s+(.*)")
_TOC_LINE_RE = re.compile(r"\.{2,}\s*\d+\s*$")  # dot-leader + trailing page number, e.g. "...... 7"
_PAGE_BREAK = "\f"


def split_recommendations(text: str) -> list[Document]:
    entries: list[dict] = []
    current: dict | None = None

    for page_index, page_text in enumerate(text.split(_PAGE_BREAK)):
        for line in page_text.split("\n"):
            stripped = line.strip()
            if not stripped or _TOC_LINE_RE.search(stripped):
                continue
            match = _RECOMMENDATION_RE.match(stripped)
            if match:
                if current is not None:
                    entries.append(current)
                current = {"number": match.group(1), "page": page_index, "lines": [stripped]}
            elif current is not None:
                current["lines"].append(stripped)
    if current is not None:
        entries.append(current)

    documents = []
    for entry in entries:
        parts = entry["number"].split(".")
        section = ".".join(parts[:-1]) if len(parts) > 1 else None
        documents.append(
            Document(
                page_content=" ".join(entry["lines"]).strip(),
                metadata={
                    "source": "NICE",
                    "recommendation_number": entry["number"],
                    "section": section,
                    "page": entry["page"],
                },
            )
        )
    return documents


def load(raw_dir: Path = RAW_DATA_DIR / "nice") -> list[Document]:
    pdf_path = raw_dir / "ng28.pdf"
    if not pdf_path.exists():
        raise SourceFilesMissing(
            f"Missing NICE NG28 PDF at {pdf_path}. Download from {NICE_PDF_URL} and save it there."
        )
    text = extract_text_from_pdf(pdf_path)
    return split_recommendations(text)
