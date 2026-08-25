from cga.graph.extraction import Claim
from cga.graph.synthesize import synthesize
from tests.graph.fakes import FakeChatModel


def _claim(**overrides):
    defaults = dict(
        source="ADA",
        recommendation="Metformin first-line.",
        population="Adults with type 2 diabetes",
        evidence_grade="A",
        stated_rationale=None,
        citation="s009",
    )
    return Claim(**{**defaults, **overrides})


def test_synthesize_returns_answer_and_citations():
    ada = _claim()
    nice = _claim(source="NICE", citation="1.9.1")
    llm = FakeChatModel(text="Both guidelines agree metformin is first-line.")

    answer, citations = synthesize(
        llm, question="What's first-line?", ada_claim=ada, nice_claim=nice, comparison="same"
    )

    assert "metformin" in answer.lower()
    assert citations == ["s009", "1.9.1"]
