"""Unit tests for the verifier's deterministic (non-LLM) citation-integrity
pass -- the check that catches a model citing a URL it was never shown.
"""

from __future__ import annotations

from deep_research.agents.verifier import check_citation_integrity
from deep_research.schemas import Evidence, Verdict


def test_claim_citing_a_url_outside_its_search_results_is_rejected():
    evidence = [
        Evidence(subquestion_id="sq1", claim="Some claim", url="https://not-in-results.example/page"),
    ]
    valid_urls = {"sq1": {"https://real-source.example/page"}}

    passed, failed = check_citation_integrity(evidence, valid_urls)

    assert passed == []
    assert len(failed) == 1
    assert failed[0].verdict == Verdict.INVALID_CITATION


def test_claim_citing_a_url_that_was_actually_returned_passes_through():
    evidence = [
        Evidence(subquestion_id="sq1", claim="Some claim", url="https://real-source.example/page"),
    ]
    valid_urls = {"sq1": {"https://real-source.example/page", "https://other.example/page"}}

    passed, failed = check_citation_integrity(evidence, valid_urls)

    assert failed == []
    assert len(passed) == 1
    assert passed[0].url == "https://real-source.example/page"


def test_subquestion_with_no_recorded_results_rejects_all_its_claims():
    evidence = [Evidence(subquestion_id="sq_missing", claim="x", url="https://anything.example")]
    passed, failed = check_citation_integrity(evidence, valid_urls_by_subquestion={})
    assert passed == []
    assert failed[0].verdict == Verdict.INVALID_CITATION
