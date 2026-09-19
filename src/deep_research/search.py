"""Search provider abstraction.

Same pattern as `llm.py`: agents depend on `SearchClient`, not on a vendor.
`TavilySearch` is the recommended provider (a free-tier key is enough to
run this whole project). `DuckDuckGoSearch` is a no-API-key fallback so
the project still runs for anyone who clones it without signing up for
anything. `MockSearch` is the deterministic, offline fixture used in
tests and CI.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Optional

import httpx

from .schemas import SearchResult


class SearchClient(ABC):
    name: str = "unknown"

    @abstractmethod
    def search(self, query: str, max_results: int = 4) -> list[SearchResult]:
        ...


class TavilySearch(SearchClient):
    name = "tavily"

    def __init__(self, api_key: Optional[str] = None, timeout: float = 15.0):
        self._key = api_key or os.getenv("TAVILY_API_KEY")
        if not self._key:
            raise RuntimeError(
                "TAVILY_API_KEY is not set. Get a free key at https://tavily.com "
                "and put it in your .env file (see .env.example)."
            )
        self._timeout = timeout

    def search(self, query: str, max_results: int = 4) -> list[SearchResult]:
        resp = httpx.post(
            "https://api.tavily.com/search",
            json={
                "api_key": self._key,
                "query": query,
                "max_results": max_results,
                "search_depth": "basic",
            },
            timeout=self._timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        results = []
        for item in data.get("results", [])[:max_results]:
            results.append(
                SearchResult(
                    url=item.get("url", ""),
                    title=item.get("title", ""),
                    snippet=item.get("content", "")[:500],
                )
            )
        return results


class DuckDuckGoSearch(SearchClient):
    """No-key fallback using DuckDuckGo's HTML lite endpoint.

    Scraping HTML is inherently more brittle than a proper search API, so
    this is offered as a zero-setup fallback, not the recommended provider
    for anything beyond local experimentation.
    """

    name = "duckduckgo"

    def __init__(self, timeout: float = 15.0):
        self._timeout = timeout

    def search(self, query: str, max_results: int = 4) -> list[SearchResult]:
        import re
        from html import unescape

        resp = httpx.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query},
            headers={"User-Agent": "Mozilla/5.0 (deep-research-agent)"},
            timeout=self._timeout,
        )
        resp.raise_for_status()
        html = resp.text
        # Minimal, dependency-free extraction of DDG lite result blocks.
        pattern = re.compile(
            r'<a rel="nofollow" class="result__a" href="(?P<url>[^"]+)">(?P<title>.*?)</a>.*?'
            r'<a class="result__snippet"[^>]*>(?P<snippet>.*?)</a>',
            re.DOTALL,
        )
        results = []
        for m in pattern.finditer(html):
            title = unescape(re.sub("<.*?>", "", m.group("title"))).strip()
            snippet = unescape(re.sub("<.*?>", "", m.group("snippet"))).strip()
            results.append(SearchResult(url=m.group("url"), title=title, snippet=snippet[:500]))
            if len(results) >= max_results:
                break
        return results


class MockSearch(SearchClient):
    """Deterministic fixture provider for tests and `eval/run_eval.py --mock`.

    Built from a dict mapping a substring of the query to a fixed list of
    `SearchResult`s, so a benchmark question always returns the same
    "web" every run.
    """

    name = "mock"

    def __init__(self, fixtures: Optional[dict[str, list[SearchResult]]] = None):
        self._fixtures = fixtures or {}

    def add(self, query_substring: str, results: list[SearchResult]) -> "MockSearch":
        self._fixtures[query_substring] = results
        return self

    def search(self, query: str, max_results: int = 4) -> list[SearchResult]:
        for key, results in self._fixtures.items():
            if key.lower() in query.lower():
                return results[:max_results]
        return []
