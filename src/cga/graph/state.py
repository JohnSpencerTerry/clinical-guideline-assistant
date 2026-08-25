"""LangGraph state schema for the assistant graph."""

from typing import Literal, TypedDict

from langchain_core.documents import Document

from cga.graph.extraction import Claim

ComparisonResult = Literal["same", "scope_difference", "conflict", "silent"]


class GraphState(TypedDict, total=False):
    question: str

    urgent: bool
    scope_status: Literal["general", "patient_specific"]
    redirect_reason: str

    ada_docs: list[Document]
    nice_docs: list[Document]

    ada_claim: Claim
    nice_claim: Claim

    comparison: ComparisonResult

    answer: str
    citations: list[str]
