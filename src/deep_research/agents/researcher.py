"""Researcher agent: for one subquestion, search the web and extract
atomic, quoted, citable claims from the top results.

The agent is explicitly told to skip sources that don't address the
subquestion rather than force a claim out of an irrelevant page — an
empty result for a subquestion is a legitimate, honest outcome, and the
writer agent surfaces that in the report's limitations section instead of
the pipeline silently inventing something.
"""

from __future__ import annotations

from ..llm import LLMClient
from ..schemas import Evidence, SearchResult, SubQuestion
from ..search import SearchClient

SYSTEM_PROMPT = """You are the researcher agent in a research assistant. You \
are given ONE subquestion and a numbered list of web search results (title, \
url, snippet) for it. Extract at most 3 atomic, specific claims that help \
answer the subquestion, each grounded in exactly one of the given results. \
Every claim MUST include a short supporting quote copied from that result's \
snippet -- never invent a quote, and never cite a result whose snippet does \
not actually support the claim. If none of the results are useful, return an \
empty claims list rather than forcing an answer."""

USER_TEMPLATE = """Subquestion: {subquestion}

Search results:
{results_block}

Respond as JSON: {{"claims": [{{"claim": "...", "result_index": <int>, "quote": "..."}}]}}
`result_index` must be the number shown before the result you are citing."""


def _format_results(results: list[SearchResult]) -> str:
    lines = []
    for i, r in enumerate(results):
        lines.append(f"[{i}] {r.title}\nURL: {r.url}\nSnippet: {r.snippet}")
    return "\n\n".join(lines) if lines else "(no results)"


def research(
    llm: LLMClient,
    search: SearchClient,
    subquestion: SubQuestion,
    max_sources: int = 4,
) -> tuple[list[Evidence], list[SearchResult]]:
    results = search.search(subquestion.text, max_results=max_sources)
    if not results:
        return [], []

    user = USER_TEMPLATE.format(subquestion=subquestion.text, results_block=_format_results(results))
    data, _resp = llm.complete_json(SYSTEM_PROMPT, user)
    raw_claims = data.get("claims", [])

    evidence: list[Evidence] = []
    for item in raw_claims:
        idx = item.get("result_index")
        if not isinstance(idx, int) or not (0 <= idx < len(results)):
            continue  # structurally invalid citation -> drop, never guess
        source = results[idx]
        claim_text = (item.get("claim") or "").strip()
        quote = (item.get("quote") or "").strip()
        if not claim_text:
            continue
        evidence.append(
            Evidence(
                subquestion_id=subquestion.id,
                claim=claim_text,
                url=source.url,
                source_title=source.title,
                quote=quote,
            )
        )
    return evidence, results
