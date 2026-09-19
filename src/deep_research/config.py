"""Runtime wiring: turns environment variables into concrete provider
instances. Keeping this in one place means `api.py`, `eval/run_eval.py`
and any future CLI all agree on the same provider-selection rules.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

from .llm import GroqLLM, LLMClient
from .search import DuckDuckGoSearch, SearchClient, TavilySearch

load_dotenv()


def build_llm() -> LLMClient:
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    if provider == "groq":
        return GroqLLM()
    raise ValueError(f"Unknown LLM_PROVIDER: {provider!r}")


def build_search() -> SearchClient:
    provider = os.getenv("SEARCH_PROVIDER", "tavily").lower()
    if provider == "tavily" and os.getenv("TAVILY_API_KEY"):
        return TavilySearch()
    if provider == "duckduckgo":
        return DuckDuckGoSearch()
    # Fall back gracefully: no Tavily key configured -> use the no-key provider
    # instead of hard-failing, since that's usually what a first-time clone wants.
    return DuckDuckGoSearch()
