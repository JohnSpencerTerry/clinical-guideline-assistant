"""Streamlit chat UI entrypoint.

Run with: uv run streamlit run app/streamlit_app.py
"""

import uuid

import streamlit as st

from cga.graph.build_graph import build_graph

st.title("Clinical Guideline Assistant")
st.caption("Grounded Q&A over Type 2 Diabetes guidelines (ADA, NICE) — learning project, not medical advice.")


@st.cache_resource
def get_app():
    return build_graph()


if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant" and message.get("redirect_reason"):
            st.warning(message["content"])
        else:
            st.markdown(message["content"])
            if message.get("citations"):
                st.caption("Sources: " + ", ".join(message["citations"]))
            if message.get("comparison") in ("scope_difference", "conflict"):
                st.badge("Guidelines differ here", icon="⚠️")

question = st.chat_input("Ask about Type 2 Diabetes guidelines...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Checking guidelines..."):
            app = get_app()
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            result = app.invoke({"question": question}, config=config)

        redirect_reason = result.get("redirect_reason")
        answer = result.get("answer", "")
        citations = result.get("citations")
        comparison = result.get("comparison")

        if redirect_reason:
            st.warning(answer)
        else:
            st.markdown(answer)
            if citations:
                st.caption("Sources: " + ", ".join(citations))
            if comparison in ("scope_difference", "conflict"):
                st.badge("Guidelines differ here", icon="⚠️")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "citations": citations,
            "comparison": comparison,
            "redirect_reason": redirect_reason,
        }
    )
