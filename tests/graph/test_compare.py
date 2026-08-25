from cga.graph.compare import ComparisonClassification, compare_claims
from cga.graph.extraction import Claim
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


def test_compare_claims_silent_when_population_not_addressed():
    ada = _claim(population="not addressed")
    nice = _claim(source="NICE", citation="1.9.1")
    llm = FakeChatModel(structured=ComparisonClassification(classification="conflict"))  # should be ignored

    result = compare_claims(llm, ada_claim=ada, nice_claim=nice)

    assert result == "silent"


def test_compare_claims_delegates_to_llm_when_both_addressed():
    ada = _claim()
    nice = _claim(source="NICE", citation="1.9.1")
    llm = FakeChatModel(structured=ComparisonClassification(classification="same"))

    result = compare_claims(llm, ada_claim=ada, nice_claim=nice)

    assert result == "same"
