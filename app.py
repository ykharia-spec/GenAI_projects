import os
import streamlit as st
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, AIMessage

from build_graph import build_graph
from web_tools import NEWS_CATEGORIES
from graph_plot import plot_langgraph_lr, default_workflow_metadata


# ---------------------------
# Config / Setup
# ---------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")


graph = build_graph()


# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="NewsGenie (Lean)", layout="wide", initial_sidebar_state="expanded")

with st.sidebar:
    st.subheader("Controls")
    category = st.selectbox("News category", NEWS_CATEGORIES, index=0)

    with st.expander("Show LangGraph workflow (L→R)", expanded=True):
        nodes, edges, edge_labels = default_workflow_metadata()

        plot_slot = st.empty()

        # If state_out exists this run, highlight it; otherwise show plain graph
        path = st.session_state.get("last_trace", [])
        fig, _ = plot_langgraph_lr(nodes, edges, edge_labels=edge_labels, path=path)
        plot_slot.pyplot(fig, clear_figure=True)

    st.caption("Tip: Ask 'latest news on <topic>' for best results.")
    st.markdown("---")
    st.subheader("Status")
    st.write("LLM key:", "✅" if OPENAI_API_KEY else "❌ missing OPENAI_API_KEY")
    st.write("News key:", "✅" if NEWS_API_KEY else "⚠️ missing NEWS_API_KEY (web fallback still works)")
    if st.button("Clear chat"):
        st.session_state.pop("messages", None)
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render history
for m in st.session_state.messages:
    role = "user" if isinstance(m, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.write(m.content)

user_input = st.chat_input("Ask anything, or request latest news…")

if user_input:
    # Add user msg
    st.session_state.messages.append(HumanMessage(content=user_input))
    with st.chat_message("user"):
        st.write(user_input)

    # Invoke LangGraph
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            state_in = {
                "messages": st.session_state.messages,
                "user_query": user_input,
                "category": category,
            }
            try:
                state_out = graph.invoke(state_in)

                print("Intent:", state_out.get("intent"))
                print("Final answer:", state_out.get("answer"))
                print("Refined query:", state_out.get("user_query"))
                print("Category:", state_out.get("category"))
                print("Error:", state_out.get("error"))

                state_out.setdefault("_trace", []).append("END")
                st.session_state["last_trace"] = state_out.get("_trace", [])
                print(state_out["_trace"])

                # Render/update the workflow plot in the sidebar using the *updated* trace
                nodes, edges, edge_labels = default_workflow_metadata()
                path = st.session_state.get("last_trace", [])
                fig, _ = plot_langgraph_lr(nodes, edges, edge_labels=edge_labels, path=path)
                plot_slot.pyplot(fig, clear_figure=True)

                answer = state_out.get("answer", "No answer produced.")
            except Exception as e:
                answer = f"⚠️ Error: {e}"

        st.write(answer)
        st.session_state.messages.append(AIMessage(content=answer))
