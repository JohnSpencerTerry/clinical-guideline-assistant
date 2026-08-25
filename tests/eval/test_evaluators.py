import pytest

from eval.dataset import EvalCase
from eval.evaluators import evaluate_scope, evaluate_urgent


def test_evaluate_urgent_pass_and_fail():
    case = EvalCase(id="x", category="urgent_symptom_detection", question="q", expect_urgent=True)
    assert evaluate_urgent(case, True) is True
    assert evaluate_urgent(case, False) is False


def test_evaluate_urgent_requires_expectation():
    case = EvalCase(id="x", category="grounded_factual_recall", question="q")
    with pytest.raises(ValueError):
        evaluate_urgent(case, True)


def test_evaluate_scope_pass_and_fail():
    case = EvalCase(id="x", category="scope_guardrail", question="q", expect_scope="general")
    assert evaluate_scope(case, "general") is True
    assert evaluate_scope(case, "patient_specific") is False
