"""Generic, question-agnostic offline providers used by `run_eval.py --mock`.

Unlike `tests/conftest.py` (which scripts exact responses for one fixed
question so the graph's *logic* can be asserted precisely), these mocks
answer any benchmark question generically. They exist so the eval harness
-- and therefore CI -- can exercise the full pipeline across the whole
benchmark set on every push, with zero network calls and zero API cost,
checking the pipeline's *structural invariants* (every run succeeds,
every claim carries a citation ID or is honestly marked unverified, trace
data is populated) rather than factual correctness, which needs a live
model and is out of scope for a deterministic CI gate.
"""

from __future__ import annotations

import json
import re

from deep_research.llm import MockLLM
from deep_research.schemas import SearchResult
from deep_research.search import SearchClient


class GenericMockSearch(SearchClient):
    name = "generic-mock"

    def search(self, query: str, max_results: int = 4) -> list[SearchResult]:
        results = [
            SearchResult(
                url=f"https://sourcea.example/{abs(hash(query)) % 9999}",
                title=f"Primary source on: {query[:60]}",
                snippet=f"According to available data, {query.rstrip('?')} is well documented in the literature.",
            ),
            SearchResult(
                url=f"https://sourceb.example/{abs(hash(query + 'b')) % 9999}",
                title=f"Secondary perspective on: {query[:60]}",
                snippet=f"A second, independent source corroborates the main findings about {query.rstrip('?')}.",
            ),
        ]
        return results[:max_results]


def _extract_question(user: str) -> str:
    m = re.search(r"Research question:\s*(.+)", user)
    return m.group(1).strip() if m else "the topic"


def _extract_subquestion(user: str) -> str:
    m = re.search(r"Subquestion:\s*(.+)", user)
    return m.group(1).strip() if m else "the subquestion"


def _extract_results(user: str) -> list[tuple[int, str, str]]:
    """Returns [(index, title, snippet), ...] parsed from the researcher/verifier prompt block."""
    pattern = re.compile(r"\[(\d+)\]\s*(.+?)\nURL: .+?\nSnippet: (.+?)(?=\n\n|\Z)", re.DOTALL)
    return [(int(i), title.strip(), snippet.strip()) for i, title, snippet in pattern.findall(user)]


def _extract_claims(user: str) -> list[tuple[int, str]]:
    """Returns [(index, claim), ...] parsed from the verifier prompt's claims block."""
    pattern = re.compile(r'\[(\d+)\] Claim: "(.+?)"')
    return [(int(i), c) for i, c in pattern.findall(user)]


def planner_handler(system: str, user: str) -> str:
    q = _extract_question(user)
    return json.dumps(
        {
            "subquestions": [
                {"text": f"What are the key facts and current state of: {q}", "rationale": "grounding"},
                {"text": f"What evidence or expert sources exist about: {q}", "rationale": "sourcing"},
            ]
        }
    )


def researcher_handler(system: str, user: str) -> str:
    results = _extract_results(user)
    if not results:
        return json.dumps({"claims": []})
    idx, title, snippet = results[0]
    return json.dumps(
        {
            "claims": [
                {
                    "claim": f"{title} indicates: {snippet[:100]}",
                    "result_index": idx,
                    "quote": snippet,
                }
            ]
        }
    )


def verifier_handler(system: str, user: str) -> str:
    claims = _extract_claims(user)
    verdicts = [
        {"claim_index": i, "verdict": "supported", "notes": "Consistent with the cited source."}
        for i, _ in claims
    ]
    return json.dumps({"verdicts": verdicts})


def writer_section_handler(system: str, user: str) -> str:
    sq = _extract_subquestion(user)
    return json.dumps({"content": f"Based on the verified sources, {sq.rstrip('?')} is supported by the cited evidence."})


def writer_summary_handler(system: str, user: str) -> str:
    q = _extract_question(user)
    return json.dumps({"summary": f"Based on verified sources, here is a grounded answer to: {q}"})


def build_generic_mock_llm() -> MockLLM:
    llm = MockLLM()
    llm.add("Produce at most", planner_handler)
    llm.add("Search results:", researcher_handler)
    llm.add("Cross-check the claims", verifier_handler)
    llm.add("Supported claims:", writer_section_handler)
    llm.add("Sections:", writer_summary_handler)
    return llm
