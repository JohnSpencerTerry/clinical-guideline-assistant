"""Branch-specific synthesis nodes: same / scope difference / conflict / silent."""

from langchain_core.language_models.chat_models import BaseChatModel

from cga.graph.extraction import Claim
from cga.graph.state import ComparisonResult

_ROLE = (
    "You explain what published clinical guidelines (ADA, NICE) say about Type 2 Diabetes. "
    "You do not provide individualized medical advice for a specific person's situation."
)

_BRANCH_INSTRUCTIONS: dict[ComparisonResult, str] = {
    "same": "Both sources agree. Give one clean, unified answer citing both.",
    "scope_difference": (
        "The sources differ because of population/scope, not a real conflict. Explain that difference "
        "explicitly rather than flattening it into one answer."
    ),
    "conflict": (
        "The sources genuinely disagree. Present both positions without picking a winner. "
        "Explain WHY they differ ONLY if a claim's stated_rationale field is non-null — quote or "
        "paraphrase that stated rationale. If both stated_rationale fields are null, explicitly say "
        "the sources don't state a reason for the difference. NEVER invent a rationale."
    ),
    "silent": "At least one source doesn't address this. Say so plainly rather than filling the gap.",
}

_PROMPT = """{role}

Question: {question}

ADA claim: {ada}
NICE claim: {nice}

Relationship between the claims: {comparison}
{instructions}

Write the answer for the user now. End with a "Sources:" line listing the citation identifiers used.
"""


def synthesize(
    llm: BaseChatModel,
    *,
    question: str,
    ada_claim: Claim,
    nice_claim: Claim,
    comparison: ComparisonResult,
) -> tuple[str, list[str]]:
    prompt = _PROMPT.format(
        role=_ROLE,
        question=question,
        ada=ada_claim.model_dump_json(),
        nice=nice_claim.model_dump_json(),
        comparison=comparison,
        instructions=_BRANCH_INSTRUCTIONS[comparison],
    )
    response = llm.invoke(prompt)
    citations = [ada_claim.citation, nice_claim.citation]
    return response.content, citations
