from cga.graph.build_graph import build_graph
from tests.graph.fakes import FakeChatModel


def test_build_graph_compiles():
    app = build_graph(llm=FakeChatModel(text="ok"))
    node_names = set(app.get_graph().nodes)
    assert "urgent_check_keyword" in node_names
    assert "synthesize" in node_names
    assert "emergency_redirect" in node_names
    assert "scope_redirect" in node_names
