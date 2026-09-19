"""Writer agent: turns verified claims into the final cited report.

Two rules keep this agent honest:
- It only ever writes prose from claims the verifier marked `supported`.
  Anything `unsupported`, `contradicted`, or `invalid_citation` is excluded
  from the narrative and instead listed, verbatim, in a "Limitations"
  section built with plain code (no LLM call) -- so that section can never
  be softened or hidden by a generation quirk.
- Every sentence in a section is expected to map to at least one citation
  id; the citations list is assembled from the same claims the section
  prose was built from, not re-derived by asking the model to remember URLs.
"""

from __future__ import annotations

from collections import defaultdict

from ..llm import LLMClient
from ..schemas import (
    Citation,
    Evidence,
    ReportSection,
    ResearchReport,
    SubQuestion,
    Verdict,
    VerifiedClaim,
)

SECTION_SYSTEM_PROMPT = """You are the writing agent in a research assistant. \
Given a subquestion and a list of verified, supported claims (each already \
citation-checked -- you do not need to re-verify them), write a concise, \
well-organized paragraph or two answering the subquestion using ONLY the \
given claims. Do not add outside knowledge or speculation. Refer to claims \
naturally; do not fabricate numbers."""

SECTION_USER_TEMPLATE = """Subquestion: {subquestion}

Supported claims:
{claims_block}

Respond as JSON: {{"content": "..."}}"""

SUMMARY_SYSTEM_PROMPT = """You are the writing agent in a research assistant. \
Given the original research question and the section summaries already \
written, write a 2-4 sentence executive summary that directly answers the \
original question, staying strictly within what the sections state."""

SUMMARY_USER_TEMPLATE = """Original question: {question}

Sections:
{sections_block}

Respond as JSON: {{"summary": "..."}}"""


def _claims_block(claims: list[VerifiedClaim]) -> str:
    return "\n".join(f'- "{c.claim}" [{c.citation_urls[0] if c.citation_urls else ""}]' for c in claims)


def write_report(
    llm: LLMClient,
    run_id: str,
    question: str,
    subquestions: list[SubQuestion],
    evidence_by_id: dict[str, Evidence],
    verified_claims: list[VerifiedClaim],
) -> ResearchReport:
    by_subquestion: dict[str, list[VerifiedClaim]] = defaultdict(list)
    for vc in verified_claims:
        by_subquestion[vc.subquestion_id].append(vc)

    citations: dict[str, Citation] = {}
    sections: list[ReportSection] = []
    limitations: list[str] = []

    for sq in subquestions:
        claims_for_sq = by_subquestion.get(sq.id, [])
        supported = [c for c in claims_for_sq if c.verdict == Verdict.SUPPORTED]

        if supported:
            user = SECTION_USER_TEMPLATE.format(
                subquestion=sq.text, claims_block=_claims_block(supported)
            )
            data, _resp = llm.complete_json(SECTION_SYSTEM_PROMPT, user)
            content = data.get("content", "").strip()
            citation_ids = []
            for c in supported:
                url = c.citation_urls[0] if c.citation_urls else ""
                ev = evidence_by_id.get(c.evidence_id)
                title = ev.source_title if ev else url
                cid = f"c{len(citations) + 1}" if url not in {cc.url for cc in citations.values()} else next(
                    cc.id for cc in citations.values() if cc.url == url
                )
                if url and cid not in citations:
                    citations[cid] = Citation(id=cid, url=url, title=title)
                if cid in citations:
                    citation_ids.append(cid)
            sections.append(ReportSection(heading=sq.text, content=content, citation_ids=citation_ids))
        else:
            sections.append(
                ReportSection(
                    heading=sq.text,
                    content="No sufficiently verified information was found for this subquestion.",
                    citation_ids=[],
                )
            )

        for c in claims_for_sq:
            if c.verdict != Verdict.SUPPORTED:
                limitations.append(f"[{sq.text}] \"{c.claim}\" -- {c.verdict.value}: {c.notes}".strip())

    sections_block = "\n\n".join(f"{s.heading}: {s.content}" for s in sections)
    summary_user = SUMMARY_USER_TEMPLATE.format(question=question, sections_block=sections_block)
    summary_data, _resp = llm.complete_json(SUMMARY_SYSTEM_PROMPT, summary_user)
    summary = summary_data.get("summary", "").strip()

    return ResearchReport(
        run_id=run_id,
        question=question,
        summary=summary,
        sections=sections,
        citations=list(citations.values()),
        limitations=limitations,
    )
