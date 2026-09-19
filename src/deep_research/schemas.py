"""Typed data contracts shared by every node in the research graph.

Keeping these as Pydantic models (rather than passing raw dicts between
agents) is what makes the pipeline testable: every node has a validated
input and output shape, so a bad LLM completion fails fast at the node
boundary instead of silently corrupting the final report.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


class SearchResult(BaseModel):
    """A single hit returned by a search provider."""

    url: str
    title: str
    snippet: str = ""
    source_domain: str = ""

    def model_post_init(self, __context) -> None:
        if not self.source_domain and self.url:
            try:
                from urllib.parse import urlparse

                object.__setattr__(self, "source_domain", urlparse(self.url).netloc.replace("www.", ""))
            except Exception:
                pass


class SubQuestion(BaseModel):
    """One decomposed piece of the user's original research question."""

    id: str = Field(default_factory=lambda: _new_id("sq"))
    text: str
    rationale: str = ""


class Evidence(BaseModel):
    """An atomic, citable claim a researcher agent extracted from one source.

    `url` MUST be a URL that actually appeared in that subquestion's search
    results — this is enforced structurally (not just trusted from the LLM)
    by the citation-integrity check in `verifier.py` before any semantic
    verification happens.
    """

    id: str = Field(default_factory=lambda: _new_id("ev"))
    subquestion_id: str
    claim: str
    url: str
    source_title: str = ""
    quote: str = ""


class Verdict(str, Enum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    CONTRADICTED = "contradicted"
    INVALID_CITATION = "invalid_citation"


class VerifiedClaim(BaseModel):
    """The outcome of running an Evidence item through the verifier."""

    evidence_id: str
    subquestion_id: str
    claim: str
    citation_urls: list[str]
    verdict: Verdict
    notes: str = ""


class Citation(BaseModel):
    id: str
    url: str
    title: str = ""


class ReportSection(BaseModel):
    heading: str
    content: str
    citation_ids: list[str] = Field(default_factory=list)


class ResearchReport(BaseModel):
    run_id: str
    question: str
    summary: str
    sections: list[ReportSection]
    citations: list[Citation]
    limitations: list[str] = Field(default_factory=list)
    generated_at: str = Field(default_factory=_now)


class ResearchRequest(BaseModel):
    question: str
    max_subquestions: int = 4
    max_sources_per_subquestion: int = 4

    @field_validator("question")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("question must not be empty")
        return v.strip()

    @field_validator("max_subquestions")
    @classmethod
    def _bound_subq(cls, v: int) -> int:
        return max(1, min(v, 6))

    @field_validator("max_sources_per_subquestion")
    @classmethod
    def _bound_sources(cls, v: int) -> int:
        return max(1, min(v, 8))


class TraceStep(BaseModel):
    node: str
    label: str
    started_at: str
    ended_at: Optional[str] = None
    latency_ms: Optional[float] = None
    tokens_in: int = 0
    tokens_out: int = 0
    detail: str = ""


class RunTrace(BaseModel):
    run_id: str
    model: str
    steps: list[TraceStep] = Field(default_factory=list)

    @property
    def total_latency_ms(self) -> float:
        return sum(s.latency_ms or 0.0 for s in self.steps)

    @property
    def total_tokens(self) -> int:
        return sum(s.tokens_in + s.tokens_out for s in self.steps)
