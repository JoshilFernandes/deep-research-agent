"""API-level test: the FastAPI app streams run events and ends with a
cited report, using the same offline mocks as the graph tests (no network).
"""

from __future__ import annotations

from fastapi.testclient import TestClient

import deep_research.api as api_module


def test_health():
    client = TestClient(api_module.app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_research_stream_ends_with_a_report(monkeypatch, mock_llm, mock_search):
    monkeypatch.setattr(api_module, "build_llm", lambda: mock_llm)
    monkeypatch.setattr(api_module, "build_search", lambda: mock_search)

    client = TestClient(api_module.app)
    with client.stream(
        "POST",
        "/api/research",
        json={"question": "What was the population of Berlin in 2024?", "max_subquestions": 2},
    ) as resp:
        assert resp.status_code == 200
        body = "".join(resp.iter_text())

    assert "run_started" in body
    assert "\"event\": \"done\"" in body or "event: done" in body
    assert "3.85" in body  # the verified population figure made it into the streamed report
