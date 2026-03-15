import os
import json
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from web_tools import fetch_news, web_search

# ---------------------------
# Config / Setup
# ---------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

llm_router = ChatOpenAI(
    model=OPENAI_MODEL,
    temperature=0.0,
    api_key=OPENAI_API_KEY,
    model_kwargs={"response_format": {"type": "json_object"}},
)
llm_general = ChatOpenAI(model=OPENAI_MODEL, temperature=0.3, api_key=OPENAI_API_KEY)
llm_news = ChatOpenAI(model=OPENAI_MODEL, temperature=0.2, api_key=OPENAI_API_KEY)


# ---------------------------
# Agents (LangGraph nodes)
# ---------------------------
ROUTER_SYSTEM = """You are an intent classifier for a news assistant.

Classify the user's message into one of the following intents:
- "news": user asks for latest news, headlines, updates or topic news
- "general": general questions, explanations, brainstorming, non-news requests
- "ambiguous": could be either or unclear; ask a short clarification

Examples:
User: "latest headlines on tesla" -> {"intent":"news","reason":"explicit latest headlines","search_query":"tesla latest headlines"}
User: "explain tesla's business model" -> {"intent":"general","reason":"explanation request","search_query":"tesla business model"}
User: "tell me about tesla" -> {"intent":"ambiguous","reason":"could be general or news","search_query":"tesla"}

Return ONLY valid JSON:
{"intent":"news|general|ambiguous","reason":"<brief>","search_query":"<refined query>"}
"""

GENERAL_SYSTEM = """You are helpful agent who is adept at answering generic and non-news queries from user.
Answer the user's question clearly and concisely.
If the user seems to want news but asked generally or if the tools failed, suggest alternatives or ask clarifying questions".
"""

NEWS_SYSTEM = """You are a very informed news agent (NewsGenie).
You will receive:
- category
- user query
- news articles (maybe empty)
- optional web results
Write:
1) 2-4 sentence overview
2) bullet list of top items: Title — Source — Date — Link
3) "What to watch" (1-2 bullets)
If no useful results, say so and ask a clarifying question.
Keep it crisp and avoid hallucinating facts.
"""


def mark_trace(state: dict, node_name: str) -> None:
    state.setdefault("_trace", []).append(node_name)


def node_router_agent(state: dict) -> dict:
    mark_trace(state, "router")
    user_query = state.get("user_query", "")
    category = state.get("category", "general")

    lowered = user_query.lower()
    if lowered.startswith("news:"):
        state["intent"] = "news"
        state["user_query"] = user_query.split(":", 1)[1].strip() or user_query
        state["error"] = None
        return state

    if lowered.startswith("general:"):
        state["intent"] = "general"
        state["user_query"] = user_query.split(":", 1)[1].strip() or user_query
        state["error"] = None
        return state

    NEWS_HINTS = ("news", "headline", "headlines", "latest", "updates", "breaking")    # some key words to interpret the user query intends to get 'news'
    if any(h in lowered for h in NEWS_HINTS):
        state["intent"] = "news"
        state["error"] = None
        return state

    msgs = [
        SystemMessage(content=ROUTER_SYSTEM),    # System Prompt to the LLM, also defines that the output be a strict json format
        HumanMessage(content=f"Category: {category}\nUser: {user_query}"),    # Human query
    ]
    resp = llm_router.invoke(msgs).content    # Invoke the LLM with the System and Human messages/prompts

    try:
        data = json.loads(resp)
    except Exception:
        data = {"intent": "ambiguous", "search_query": user_query, "reason": "Could not parse JSON."}

    intent = data.get("intent", "ambiguous")
    if intent not in ("news", "general", "ambiguous"):
        intent = "ambiguous"

    state["intent"] = intent
    state["user_query"] = (data.get("search_query") or user_query).strip()
    state["error"] = None
    return state


def node_general_agent(state: dict) -> dict:
    mark_trace(state, "general")
    user_query = state.get("user_query", "")
    messages = state.get("messages", [])

    prompt = [SystemMessage(content=GENERAL_SYSTEM)]
    # keep a little context
    prompt.extend(messages[-10:])
    prompt.append(HumanMessage(content=user_query))

    ans = llm_general.invoke(prompt).content
    state["answer"] = ans
    return state


def node_news_agent(state: dict) -> dict:
    mark_trace(state, "news")
    category = state.get("category", "general")
    user_query = state.get("user_query", "")

    news_items, web_results = [], []
    error = None

    # 1) Try News API
    try:
        news_items = fetch_news(category, user_query)    # fetch news from newsapi.org based on user_query
    except Exception as e:
        error = str(e)
        news_items = []

    # 2) If no news found (or API failed), do web search fallback
    if not news_items:
        try:
            web_results = web_search(user_query, max_results=5)    # No news found at newsapi.org. So, use web search.
        except Exception as e:
            error = error or str(e)
            web_results = []

    state["news_items"] = news_items
    state["web_results"] = web_results
    state["error"] = error

    # 3) Summarize with NewsAgent
    context = {
        "category": category,
        "query": user_query,
        "news_items": news_items,
        "web_results": web_results,
        "tool_error": error,
    }

    prompt = [
        SystemMessage(content=NEWS_SYSTEM),    # System prompt to let LLM know it is news agent and what it should output
        HumanMessage(content=f"Context JSON:\n{json.dumps(context, ensure_ascii=False, indent=2)}\n\nWrite the response."),    # Provide a nice formatted json of the context to LLM
    ]
    ans = llm_news.invoke(prompt).content
    state["answer"] = ans
    return state


def node_clarify(state: dict) -> dict:
    mark_trace(state, "clarify")
    q = state.get("user_query", "")
    state["answer"] = (
        f"Quick clarification: do you want **latest news or headlines** about **“{q}”** "
        "or a **general explanation**?\n\n"
        "Reply with:\n- `news: <topic>` or\n- `general: <question>`"
    )
    return state
