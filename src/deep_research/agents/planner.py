"""Planner agent: turns one broad research question into a small set of
independently-searchable subquestions.

Decomposition is what lets the graph fan out research in parallel (see
`graph.py`), and it also bounds the blast radius of a bad LLM turn: if the
planner mis-parses, only planning fails, not the whole run.
"""

from __future__ import annotations

from ..llm import LLMClient
from ..schemas import SubQuestion

SYSTEM_PROMPT = """You are the planning agent in a research assistant. Given a \
user's research question, break it into a small number of concrete, \
independently-searchable subquestions that together would let someone \
write a well-sourced answer to the original question. Avoid overlap \
between subquestions. Prefer subquestions that are specific enough to \
search well (include entities, dates, or comparisons where relevant)."""

USER_TEMPLATE = """Research question: {question}

Produce at most {max_subquestions} subquestions as a JSON object of the form:
{{"subquestions": [{{"text": "...", "rationale": "why this matters to the answer"}}]}}"""


def plan(llm: LLMClient, question: str, max_subquestions: int = 4) -> list[SubQuestion]:
    user = USER_TEMPLATE.format(question=question, max_subquestions=max_subquestions)
    data, _resp = llm.complete_json(SYSTEM_PROMPT, user)
    raw = data.get("subquestions", [])
    if not raw:
        raise ValueError("Planner returned zero subquestions")
    subquestions = [
        SubQuestion(text=item["text"].strip(), rationale=item.get("rationale", "").strip())
        for item in raw[:max_subquestions]
        if item.get("text", "").strip()
    ]
    if not subquestions:
        raise ValueError("Planner returned subquestions with no usable text")
    return subquestions
