import streamlit as st
from langgraph.graph import StateGraph, END

from agents import node_router_agent, node_general_agent, node_news_agent, node_clarify


# ---------------------------
# Graph wiring (minimal)
# ---------------------------
def route_from_intent(state: dict) -> str:
    intent = state.get("intent", "ambiguous")
    if intent == "news":
        return "news"
    if intent == "general":
        return "general"
    return "ambiguous"


@st.cache_resource
def build_graph():
    g = StateGraph(dict)
    g.add_node("router", node_router_agent)
    g.add_node("general", node_general_agent)
    g.add_node("news", node_news_agent)
    g.add_node("clarify", node_clarify)

    g.set_entry_point("router")
    g.add_conditional_edges(
        "router",
        route_from_intent,
        {"news": "news", "general": "general", "ambiguous": "clarify"},
    )

    g.add_edge("news", END)
    g.add_edge("general", END)
    g.add_edge("clarify", END)
    return g.compile()