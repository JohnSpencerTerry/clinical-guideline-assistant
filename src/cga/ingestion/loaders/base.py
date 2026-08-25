"""Shared helpers for source loaders: raw-file locations, PDF text extraction."""

from pathlib import Path

from pypdf import PdfReader

RAW_DATA_DIR = Path("data/raw")

PAGE_BREAK = "\f"


class SourceFilesMissing(Exception):
    """Raised when a loader's expected raw file(s) aren't present on disk."""


def extract_text_from_pdf(path: Path) -> str:
    reader = PdfReader(path)
    return PAGE_BREAK.join(page.extract_text() or "" for page in reader.pages)
