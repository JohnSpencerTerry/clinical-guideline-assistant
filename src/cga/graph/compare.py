"""compare_claims node: classify same / scope difference / genuine conflict / silent.

Classification only — not resolution or hedging.
"""

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field

from cga.graph.extraction import Claim
from cga.graph.state import ComparisonResult

_NOT_ADDRESSED = {"not addressed", "n/a", "none", ""}


class ComparisonClassification(BaseModel):
    classification: ComparisonResult = Field(description="One of: same, scope_difference, conflict, silent.")


_PROMPT = """Compare these two guideline claims answering the same question and classify their relationship.

ADA claim: {ada}
NICE claim: {nice}

Classify as exactly one of:
- "same": both sources recommend the same thing for the same population.
- "scope_difference": they differ only because of population/scope/phrasing, not a real conflict.
- "conflict": the sources genuinely recommend different things for the same scenario.
- "silent": one source doesn't address this at all (population is "not addressed").

Classify only — do not explain or resolve the difference.
"""


def compare_claims(llm: BaseChatModel, *, ada_claim: Claim, nice_claim: Claim) -> ComparisonResult:
    if ada_claim.population.strip().lower() in _NOT_ADDRESSED or nice_claim.population.strip().lower() in _NOT_ADDRESSED:
        return "silent"

    structured_llm = llm.with_structured_output(ComparisonClassification)
    prompt = _PROMPT.format(
        ada=ada_claim.model_dump_json(),
        nice=nice_claim.model_dump_json(),
    )
    result = structured_llm.invoke(prompt)
    return result.classification
