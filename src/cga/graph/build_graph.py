"""Wires guardrail, retrieval, extraction, comparison, and synthesis nodes into the LangGraph graph.

See the LangGraph flow diagram in mvp.md. Retrieval/extraction run sequentially
per source rather than fanned out in parallel (LangGraph's Send API could
parallelize this later) — same functional independence, simpler first version.
"""

from langgraph.graph import END, StateGraph

from cga.graph.compare import compare_claims
from cga.graph.extraction import extract_claim
from cga.graph.guardrails.scope_classifier import classify_scope
from cga.graph.guardrails.urgent_check import keyword_check, llm_check
from cga.graph.llm import get_chat_model
from cga.graph.retrieval import retrieve
from cga.graph.state import GraphState
from cga.graph.synthesize import synthesize
from cga.memory.checkpointer import get_checkpointer

EMERGENCY_MESSAGE = (
    "This sounds like it could be a medical emergency. Please call 911 (or your local emergency number) "
    "or go to the nearest emergency room now. I can't safely answer guideline questions in place of "
    "immediate care."
)

SCOPE_MESSAGE = (
    "I can explain what published Type 2 Diabetes guidelines say in general, but I can't give "
    "individualized medical advice for a specific person's situation. Feel free to ask me the general "
    "version of your question instead (e.g. what guidelines recommend for a given population/scenario), "
    "and bring your personal situation to your care team."
)


def build_graph(*, llm=None, checkpointer=None):
    llm = llm or get_chat_model()

    def node_urgent_keyword(state: GraphState) -> dict:
        return {"urgent": keyword_check(state["question"])}

    def node_urgent_llm(state: GraphState) -> dict:
        if state.get("urgent"):
            return {}
        return {"urgent": llm_check(llm, state["question"])}

    def node_emergency_redirect(state: GraphState) -> dict:
        return {"answer": EMERGENCY_MESSAGE, "redirect_reason": "urgent"}

    def node_scope_classifier(state: GraphState) -> dict:
        return {"scope_status": classify_scope(llm, state["question"])}

    def node_scope_redirect(state: GraphState) -> dict:
        return {"answer": SCOPE_MESSAGE, "redirect_reason": "patient_specific"}

    def node_retrieve_ada(state: GraphState) -> dict:
        return {"ada_docs": retrieve("ada", state["question"])}

    def node_retrieve_nice(state: GraphState) -> dict:
        return {"nice_docs": retrieve("nice", state["question"])}

    def node_extract_ada(state: GraphState) -> dict:
        claim = extract_claim(llm, question=state["question"], source="ADA", docs=state["ada_docs"])
        return {"ada_claim": claim}

    def node_extract_nice(state: GraphState) -> dict:
        claim = extract_claim(llm, question=state["question"], source="NICE", docs=state["nice_docs"])
        return {"nice_claim": claim}

    def node_compare(state: GraphState) -> dict:
        comparison = compare_claims(llm, ada_claim=state["ada_claim"], nice_claim=state["nice_claim"])
        return {"comparison": comparison}

    def node_synthesize(state: GraphState) -> dict:
        answer, citations = synthesize(
            llm,
            question=state["question"],
            ada_claim=state["ada_claim"],
            nice_claim=state["nice_claim"],
            comparison=state["comparison"],
        )
        return {"answer": answer, "citations": citations}

    graph = StateGraph(GraphState)
    graph.add_node("urgent_check_keyword", node_urgent_keyword)
    graph.add_node("urgent_check_llm", node_urgent_llm)
    graph.add_node("emergency_redirect", node_emergency_redirect)
    graph.add_node("scope_classifier", node_scope_classifier)
    graph.add_node("scope_redirect", node_scope_redirect)
    graph.add_node("retrieve_ada", node_retrieve_ada)
    graph.add_node("retrieve_nice", node_retrieve_nice)
    graph.add_node("extract_ada", node_extract_ada)
    graph.add_node("extract_nice", node_extract_nice)
    graph.add_node("compare_claims", node_compare)
    graph.add_node("synthesize", node_synthesize)

    graph.set_entry_point("urgent_check_keyword")
    graph.add_conditional_edges(
        "urgent_check_keyword",
        lambda s: "hit" if s["urgent"] else "no_hit",
        {"hit": "emergency_redirect", "no_hit": "urgent_check_llm"},
    )
    graph.add_conditional_edges(
        "urgent_check_llm",
        lambda s: "hit" if s["urgent"] else "no_hit",
        {"hit": "emergency_redirect", "no_hit": "scope_classifier"},
    )
    graph.add_conditional_edges(
        "scope_classifier",
        lambda s: s["scope_status"],
        {"patient_specific": "scope_redirect", "general": "retrieve_ada"},
    )
    graph.add_edge("retrieve_ada", "extract_ada")
    graph.add_edge("extract_ada", "retrieve_nice")
    graph.add_edge("retrieve_nice", "extract_nice")
    graph.add_edge("extract_nice", "compare_claims")
    graph.add_edge("compare_claims", "synthesize")
    graph.add_edge("synthesize", END)
    graph.add_edge("emergency_redirect", END)
    graph.add_edge("scope_redirect", END)

    return graph.compile(checkpointer=checkpointer or get_checkpointer())


def ask(question: str, *, thread_id: str = "default") -> GraphState:
    """Convenience one-shot call: builds a fresh graph (and checkpointer) per invocation.

    `thread_id` only threads state within a single call's checkpointer, not
    across separate `ask()` calls — for real multi-turn memory, build the
    graph once (`build_graph()`) and invoke the same compiled app repeatedly,
    as `app/streamlit_app.py` does via `st.cache_resource`.
    """
    app = build_graph()
    return app.invoke({"question": question}, config={"configurable": {"thread_id": thread_id}})
