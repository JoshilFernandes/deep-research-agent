"""Verifier agent: the project's core "don't just trust the model" layer.

Verification happens in two passes, deliberately in this order:

1. Deterministic citation-integrity check (no LLM call). Every claim's URL
   must be one of the URLs that were actually returned for that
   subquestion's search. This alone catches an entire class of failure
   (a model citing a source it never saw) with zero ambiguity and zero
   token cost.

2. LLM cross-source check, run once per subquestion over all of that
   subquestion's surviving claims together, so the model can compare them
   against each other. It classifies each claim as supported (consistent
   with at least one source and not contradicted), contradicted (conflicts
   with another claim on the same subquestion), or unsupported (too weak
   or vague to state as fact).

Claims that fail either pass are never silently dropped -- they flow into
the writer agent's "Limitations" section instead, so the final report is
explicit about what it could and couldn't verify.
"""

from __future__ import annotations

from collections import defaultdict

from ..llm import LLMClient
from ..schemas import Evidence, Verdict, VerifiedClaim

SYSTEM_PROMPT = """You are the verification agent in a research assistant. You \
are given a subquestion and a numbered list of claims gathered about it, each \
with its source quote. Cross-check the claims against each other: \
- Mark a claim "contradicted" if another claim in the list conflicts with it \
  (e.g. different numbers, dates, or opposite conclusions for the same fact). \
- Mark a claim "unsupported" if its quote is too vague, off-topic, or weak to \
  actually establish the claim. \
- Otherwise mark it "supported". \
Be conservative: prefer "unsupported" over "supported" when in doubt."""

USER_TEMPLATE = """Subquestion: {subquestion}

Claims:
{claims_block}

Respond as JSON: {{"verdicts": [{{"claim_index": <int>, "verdict": "supported"|"unsupported"|"contradicted", "notes": "..."}}]}}
Include exactly one verdict per claim index shown."""


def _format_claims(items: list[Evidence]) -> str:
    lines = []
    for i, ev in enumerate(items):
        lines.append(f'[{i}] Claim: "{ev.claim}"\n    Source quote: "{ev.quote}" ({ev.url})')
    return "\n".join(lines)


def check_citation_integrity(
    evidence: list[Evidence], valid_urls_by_subquestion: dict[str, set[str]]
) -> tuple[list[Evidence], list[VerifiedClaim]]:
    """Deterministic pass. Returns (evidence that passed, verdicts for evidence that failed)."""
    passed: list[Evidence] = []
    failed: list[VerifiedClaim] = []
    for ev in evidence:
        valid_urls = valid_urls_by_subquestion.get(ev.subquestion_id, set())
        if ev.url in valid_urls:
            passed.append(ev)
        else:
            failed.append(
                VerifiedClaim(
                    evidence_id=ev.id,
                    subquestion_id=ev.subquestion_id,
                    claim=ev.claim,
                    citation_urls=[ev.url],
                    verdict=Verdict.INVALID_CITATION,
                    notes="Cited URL did not appear in this subquestion's search results.",
                )
            )
    return passed, failed


def verify_semantic(llm: LLMClient, subquestion_text: str, evidence: list[Evidence]) -> list[VerifiedClaim]:
    """LLM pass over one subquestion's already citation-valid claims."""
    if not evidence:
        return []
    user = USER_TEMPLATE.format(subquestion=subquestion_text, claims_block=_format_claims(evidence))
    data, _resp = llm.complete_json(SYSTEM_PROMPT, user)
    verdicts_raw = {v.get("claim_index"): v for v in data.get("verdicts", [])}

    results: list[VerifiedClaim] = []
    for i, ev in enumerate(evidence):
        v = verdicts_raw.get(i, {"verdict": "unsupported", "notes": "No verdict returned by verifier."})
        try:
            verdict = Verdict(v.get("verdict", "unsupported"))
        except ValueError:
            verdict = Verdict.UNSUPPORTED
        results.append(
            VerifiedClaim(
                evidence_id=ev.id,
                subquestion_id=ev.subquestion_id,
                claim=ev.claim,
                citation_urls=[ev.url],
                verdict=verdict,
                notes=v.get("notes", ""),
            )
        )
    return results


def verify_all(
    llm: LLMClient,
    evidence: list[Evidence],
    subquestion_text_by_id: dict[str, str],
    valid_urls_by_subquestion: dict[str, set[str]],
) -> list[VerifiedClaim]:
    passed, invalid_verdicts = check_citation_integrity(evidence, valid_urls_by_subquestion)

    grouped: dict[str, list[Evidence]] = defaultdict(list)
    for ev in passed:
        grouped[ev.subquestion_id].append(ev)

    semantic_verdicts: list[VerifiedClaim] = []
    for subquestion_id, items in grouped.items():
        semantic_verdicts.extend(
            verify_semantic(llm, subquestion_text_by_id.get(subquestion_id, ""), items)
        )

    return invalid_verdicts + semantic_verdicts
