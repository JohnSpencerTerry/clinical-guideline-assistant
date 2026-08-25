from cga.graph.guardrails.scope_classifier import classify_scope
from tests.graph.fakes import FakeChatModel


def test_classify_scope_general():
    llm = FakeChatModel(text="general")
    assert classify_scope(llm, "What do guidelines say about metformin first-line?") == "general"


def test_classify_scope_patient_specific():
    llm = FakeChatModel(text="patient_specific")
    assert classify_scope(llm, "Should I stop taking my metformin?") == "patient_specific"
