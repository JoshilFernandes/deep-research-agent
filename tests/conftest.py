"""Shared fixtures for the offline test suite.

The scripted mock LLM below plays the role of a real model across all four
agent turns for a fixed two-subquestion run: it knows how to answer the
planner's decomposition prompt, the researcher's extraction prompt (for
each subquestion, including one where the two sources deliberately
disagree with each other), the verifier's cross-check prompt, and the
writer's section/summary prompts. This is what lets `test_graph_end_to_end`
run the *real* graph and agent code, with zero network calls.
"""

from __future__ import annotations

import json

import pytest

from deep_research.llm import MockLLM
from deep_research.schemas import SearchResult
from deep_research.search import MockSearch

QUESTION = "What was the population of Berlin in 2024?"


@pytest.fixture
def mock_search() -> MockSearch:
    search = MockSearch()
    search.add(
        "official population",
        [
            SearchResult(
                url="https://www.berlin.de/statistics/population-2024",
                title="Berlin Statistics Office: Population 2024",
                snippet="The official registered population of Berlin was 3.85 million at the end of 2024.",
            ),
            SearchResult(
                url="https://example-blog.com/berlin-2024",
                title="A random blog about Berlin",
                snippet="Berlin is a great city with lots of parks and a growing tech scene.",
            ),
        ],
    )
    search.add(
        "population growth",
        [
            SearchResult(
                url="https://www.berlin.de/statistics/population-2024",
                title="Berlin Statistics Office: Population 2024",
                snippet="Population grew by about 0.9% compared to 2023, driven mainly by migration.",
            ),
            SearchResult(
                url="https://competing-outlet.example/berlin-pop",
                title="Competing Outlet: Berlin shrinking",
                snippet="Berlin's population actually fell by 2% in 2024 according to unofficial estimates.",
            ),
        ],
    )
    return search


@pytest.fixture
def mock_llm() -> MockLLM:
    llm = MockLLM()

    def planner_handler(system: str, user: str) -> str:
        return json.dumps(
            {
                "subquestions": [
                    {"text": "What was Berlin's official population 2024?", "rationale": "core fact"},
                    {"text": "How did Berlin's population growth trend in 2024?", "rationale": "trend context"},
                ]
            }
        )

    def researcher_handler(system: str, user: str) -> str:
        if "official population" in user:
            return json.dumps(
                {
                    "claims": [
                        {
                            "claim": "Berlin's official registered population was 3.85 million at end of 2024.",
                            "result_index": 0,
                            "quote": "The official registered population of Berlin was 3.85 million at the end of 2024.",
                        }
                    ]
                }
            )
        if "population growth" in user:
            return json.dumps(
                {
                    "claims": [
                        {
                            "claim": "Berlin's population grew about 0.9% in 2024 vs 2023.",
                            "result_index": 0,
                            "quote": "Population grew by about 0.9% compared to 2023, driven mainly by migration.",
                        },
                        {
                            "claim": "Berlin's population fell 2% in 2024.",
                            "result_index": 1,
                            "quote": "Berlin's population actually fell by 2% in 2024 according to unofficial estimates.",
                        },
                    ]
                }
            )
        return json.dumps({"claims": []})

    def verifier_handler(system: str, user: str) -> str:
        if "growth trend" in user:
            # Two claims about the same subquestion disagree -> contradicted.
            return json.dumps(
                {
                    "verdicts": [
                        {"claim_index": 0, "verdict": "contradicted", "notes": "Conflicts with claim 1 (opposite direction)."},
                        {"claim_index": 1, "verdict": "contradicted", "notes": "Conflicts with claim 0 (opposite direction)."},
                    ]
                }
            )
        return json.dumps(
            {"verdicts": [{"claim_index": 0, "verdict": "supported", "notes": "Single authoritative source."}]}
        )

    def writer_section_handler(system: str, user: str) -> str:
        if "official population" in user:
            return json.dumps({"content": "Berlin's official registered population was 3.85 million at the end of 2024."})
        return json.dumps({"content": "No sufficiently verified information was found for this subquestion."})

    def writer_summary_handler(system: str, user: str) -> str:
        return json.dumps(
            {
                "summary": "Berlin's official population was 3.85 million at the end of 2024; growth-trend sources disagreed and could not be verified."
            }
        )

    llm.add("Produce at most", planner_handler)
    llm.add("Search results:", researcher_handler)
    llm.add("Cross-check the claims", verifier_handler)
    llm.add("Supported claims:", writer_section_handler)
    llm.add("Sections:", writer_summary_handler)
    return llm
