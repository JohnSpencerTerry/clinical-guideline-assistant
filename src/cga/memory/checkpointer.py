"""LangGraph checkpointer for conversation state.

Starts with an in-memory checkpointer (per-process only, resets on restart) —
swap for a persistent backend (e.g. SqliteSaver) later if cross-session
memory is needed.
"""

from langgraph.checkpoint.memory import MemorySaver


def get_checkpointer() -> MemorySaver:
    return MemorySaver()
