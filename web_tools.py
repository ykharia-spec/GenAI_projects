
import os
import requests
from dotenv import load_dotenv
from duckduckgo_search import DDGS

# ---------------------------
# Config / Setup
# ---------------------------
load_dotenv()

NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# Seven categories supported by newsapi.org
NEWS_CATEGORIES = ["general", "business", "technology", "sports", "entertainment", "health", "science"]


# ---------------------------
# Tools (simple functions)
# ---------------------------
def fetch_news(category: str, query: str, page_size: int = 6):
    """
    NewsAPI.org top-headlines by category + optional query.
    Returns a normalized list of dicts.
    """
    if not NEWS_API_KEY:
        raise RuntimeError("Missing NEWS_API_KEY (set it in .env).")

    category = category if category in NEWS_CATEGORIES else "general"
    q = (query or "").strip()

    if q:
        url = "https://newsapi.org/v2/everything"
        params = {
            "apiKey": NEWS_API_KEY,
            "q": q,
            "pageSize": page_size,
            "language": "en",
            "sortBy": "publishedAt",
        }
    else:
        url = "https://newsapi.org/v2/top-headlines"
        params = {
            "apiKey": NEWS_API_KEY,
            "category": category,
            "pageSize": page_size,
            "language": "en",
            "country": "us",
        }

    r = requests.get(url, params=params, timeout=12)
    if r.status_code != 200:
        raise RuntimeError(f"News API error {r.status_code}: {r.text}")

    articles = r.json().get("articles", []) or []
    normalized = []
    for a in articles:
        normalized.append({
            "title": a.get("title"),
            "source": (a.get("source") or {}).get("name"),
            "publishedAt": a.get("publishedAt"),
            "url": a.get("url"),
            "description": a.get("description"),
        })
    return normalized


def web_search(query: str, max_results: int = 5):
    """Keyless web search fallback via DuckDuckGo."""
    if not query or not query.strip():
        return []
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query.strip(), max_results=max_results):
            results.append({
                "title": r.get("title"),
                "url": r.get("href"),
                "snippet": r.get("body"),
            })
    return results
