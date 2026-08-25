"""Structured claim extraction per source.

Schema: recommendation, applicable population, evidence grade, and an
optional stated_rationale field populated only when the source text itself
gives a reason (never inferred).
"""

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field


class Claim(BaseModel):
    source: str
    recommendation: str = Field(description="The guideline's recommendation, in the model's own words.")
    population: str = Field(description="Who this recommendation applies to.")
    evidence_grade: str | None = Field(default=None, description="Evidence grade/level, if the source states one.")
    stated_rationale: str | None = Field(
        default=None,
        description="The reason given IN THE SOURCE TEXT for this recommendation, if any. Never inferred.",
    )
    citation: str = Field(description="Section/recommendation identifier to cite back to.")


_EXTRACTION_PROMPT = """You extract a single structured claim from clinical guideline text, answering the question below.

Question: {question}

Source: {source}
Retrieved passages (each tagged with its citation identifier):
{passages}

Extract ONE claim that answers the question, grounded only in the passages above.
- `stated_rationale` must be populated ONLY if the passages themselves state a reason. If no reason is given in the text, leave it null. Never infer a rationale that isn't written in the source.
- `citation` must be the identifier of the passage the recommendation is drawn from.
If the passages don't address the question at all, still return your best-effort claim noting that in `recommendation`, with population "not addressed" and citation of the closest passage.
"""


def _format_passages(docs: list[Document]) -> str:
    lines = []
    for doc in docs:
        citation = doc.metadata.get("citation") or doc.metadata.get("recommendation_number") or doc.metadata.get(
            "section_number"
        )
        lines.append(f"[{citation}] {doc.page_content}")
    return "\n\n".join(lines)


def extract_claim(llm: BaseChatModel, *, question: str, source: str, docs: list[Document]) -> Claim:
    structured_llm = llm.with_structured_output(Claim)
    prompt = _EXTRACTION_PROMPT.format(question=question, source=source, passages=_format_passages(docs))
    claim = structured_llm.invoke(prompt)
    claim.source = source
    return claim
