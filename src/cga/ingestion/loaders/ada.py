"""Loader for the ADA Standards of Care in Diabetes—2026 (17 individual section PDFs).

Each section is its own journal article, downloaded separately from its DOI
page — not one combined PDF for the whole guideline.
"""

from pathlib import Path

from langchain_core.documents import Document

from cga.ingestion.loaders.base import (
    RAW_DATA_DIR,
    SourceFilesMissing,
    extract_text_from_pdf,
)

ADA_SECTIONS: list[tuple[int, str, str]] = [
    (1, "Improving Care and Promoting Health in Populations", "https://doi.org/10.2337/dc26-s001"),
    (2, "Diagnosis and Classification of Diabetes", "https://doi.org/10.2337/dc26-s002"),
    (3, "Prevention or Delay of Diabetes and Associated Comorbidities", "https://doi.org/10.2337/dc26-s003"),
    (4, "Comprehensive Medical Evaluation and Assessment of Comorbidities", "https://doi.org/10.2337/dc26-s004"),
    (5, "Facilitating Positive Health Behaviors and Well-being to Improve Health Outcomes", "https://doi.org/10.2337/dc26-s005"),
    (6, "Glycemic Goals and Hypoglycemia", "https://doi.org/10.2337/dc26-s006"),
    (7, "Diabetes Technology", "https://doi.org/10.2337/dc26-s007"),
    (8, "Obesity and Weight Management for the Prevention and Treatment of Type 2 Diabetes", "https://doi.org/10.2337/dc26-s008"),
    (9, "Pharmacologic Approaches to Glycemic Treatment", "https://doi.org/10.2337/dc26-s009"),
    (10, "Cardiovascular Disease and Risk Management", "https://doi.org/10.2337/dc26-s010"),
    (11, "Chronic Kidney Disease and Risk Management", "https://doi.org/10.2337/dc26-s011"),
    (12, "Retinopathy, Neuropathy, and Foot Care", "https://doi.org/10.2337/dc26-s012"),
    (13, "Older Adults", "https://doi.org/10.2337/dc26-S013"),
    (14, "Children and Adolescents", "https://doi.org/10.2337/dc26-s014"),
    (15, "Management of Diabetes in Pregnancy", "https://doi.org/10.2337/dc26-s015"),
    (16, "Diabetes Care in the Hospital", "https://doi.org/10.2337/dc26-s016"),
    (17, "Diabetes Advocacy", "https://doi.org/10.2337/dc26-s017"),
]


def parse_section(text: str, *, section_number: int, title: str, url: str) -> Document:
    return Document(
        page_content=text.strip(),
        metadata={
            "source": "ADA",
            "section_number": section_number,
            "title": title,
            "url": url,
        },
    )


def load(raw_dir: Path = RAW_DATA_DIR / "ada") -> list[Document]:
    missing = [
        (number, title, url)
        for number, title, url in ADA_SECTIONS
        if not (raw_dir / f"section_{number:02d}.pdf").exists()
    ]
    if missing:
        lines = "\n".join(f"  section_{n:02d}.pdf  <-  {url}  ({title})" for n, title, url in missing)
        raise SourceFilesMissing(
            f"Missing ADA section PDF(s) in {raw_dir}/. Download and save each as shown:\n{lines}"
        )

    documents = []
    for number, title, url in ADA_SECTIONS:
        text = extract_text_from_pdf(raw_dir / f"section_{number:02d}.pdf")
        documents.append(parse_section(text, section_number=number, title=title, url=url))
    return documents
