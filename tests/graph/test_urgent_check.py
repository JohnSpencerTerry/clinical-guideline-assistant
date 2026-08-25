from cga.graph.guardrails.urgent_check import is_urgent, keyword_check, llm_check
from tests.graph.fakes import FakeChatModel


def test_keyword_check_hits_on_chest_pain():
    assert keyword_check("I've had chest pain for an hour") is True


def test_keyword_check_hits_on_paraphrased_breathing_difficulty():
    assert keyword_check("I haven't been able to catch my breath since this morning") is True


def test_keyword_check_no_hit_on_general_question():
    assert keyword_check("What's the first-line treatment for type 2 diabetes?") is False


def test_llm_check_uses_fallback_classifier():
    llm = FakeChatModel(text="yes")
    assert llm_check(llm, "some paraphrased emergency") is True

    llm = FakeChatModel(text="no")
    assert llm_check(llm, "a normal question") is False


def test_is_urgent_short_circuits_keyword_before_llm():
    llm = FakeChatModel(text="no")  # would say no, but keyword should still win
    assert is_urgent(llm, "I have chest pain") is True
