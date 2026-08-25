from langchain_core.documents import Document

from cga.graph.extraction import Claim, extract_claim
from tests.graph.fakes import FakeChatModel


def test_extract_claim_stamps_source():
    claim_stub = Claim(
        source="wrong",  # extract_claim should overwrite this with the passed source
        recommendation="Metformin is first-line.",
        population="Adults with type 2 diabetes",
        evidence_grade="A",
        stated_rationale=None,
        citation="1.9.1",
    )
    llm = FakeChatModel(structured=claim_stub)
    docs = [Document(page_content="text", metadata={"recommendation_number": "1.9.1"})]

    claim = extract_claim(llm, question="What's first-line?", source="NICE", docs=docs)

    assert claim.source == "NICE"
    assert claim.citation == "1.9.1"
