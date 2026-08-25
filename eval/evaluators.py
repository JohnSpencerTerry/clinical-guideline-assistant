"""Deterministic evaluators for the guardrail categories (structural, no LLM-judge needed).

Grounded-recall/comparison/adversarial categories don't have hand-verified
expected answers yet, so they're run-and-report only here — add an
LLM-as-judge groundedness evaluator once there's a reference answer to check
against (mvp.md's plan for the rationale-groundedness check).
"""

from eval.dataset import EvalCase


def evaluate_urgent(case: EvalCase, actual_urgent: bool) -> bool:
    if case.expect_urgent is None:
        raise ValueError(f"{case.id} has no expect_urgent to grade against")
    return actual_urgent == case.expect_urgent


def evaluate_scope(case: EvalCase, actual_scope: str) -> bool:
    if case.expect_scope is None:
        raise ValueError(f"{case.id} has no expect_scope to grade against")
    return actual_scope == case.expect_scope
