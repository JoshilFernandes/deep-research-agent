from __future__ import annotations

import pytest
from pydantic import ValidationError

from deep_research.schemas import ResearchRequest, SearchResult


def test_research_request_rejects_empty_question():
    with pytest.raises(ValidationError):
        ResearchRequest(question="   ")


def test_research_request_clamps_out_of_range_bounds():
    req = ResearchRequest(question="What is quantum computing?", max_subquestions=99, max_sources_per_subquestion=0)
    assert req.max_subquestions == 6  # clamped to the documented max
    assert req.max_sources_per_subquestion == 1  # clamped to the documented min


def test_search_result_derives_domain_from_url_when_not_given():
    r = SearchResult(url="https://www.example.com/some/page", title="t", snippet="s")
    assert r.source_domain == "example.com"
