"""End-to-end test of the full plan -> research -> verify -> write pipeline,
run entirely against `MockLLM` / `MockSearch` (see conftest.py).

This is the test that matters most for this project: it proves the graph
wiring, the reducers, the parallel fan-out, and all four agents work
together correctly -- not just in isolation -- without needing a live
Groq/Tavily key. The same fixtures back `eval/run_eval.py --mock`, which
is what runs in CI on every push.
"""

from __future__ import annotations

from deep_research.graph import run_research
from deep_research.schemas import Verdict


def test_full_pipeline_produces_a_cited_report_and_flags_the_contradiction(mock_llm, mock_search):
    report, tracer = run_research(
        mock_llm,
        mock_search,
        question="What was the population of Berlin in 2024?",
        max_subquestions=2,
        max_sources_per_subquestion=4,
        run_id="test_run_1",
    )

    # The report exists and answers the question.
    assert report.run_id == "test_run_1"
    assert "3.85 million" in report.summary or "3.85" in report.summary
    assert len(report.sections) == 2

    # The supported claim made it into a section with a real citation.
    berlin_section = next(s for s in report.sections if "official population" in s.heading)
    assert berlin_section.citation_ids, "supported section should carry at least one citation"
    assert any("berlin.de" in c.url for c in report.citations)

    # The contradiction between the two growth-trend sources was CAUGHT,
    # not silently resolved or hallucinated away.
    assert report.limitations, "contradictory claims must surface as limitations"
    assert any("contradicted" in note.lower() for note in report.limitations)

    # Tracer recorded every node, including the parallel research fan-out.
    node_names = [s.node for s in tracer.trace.steps]
    assert node_names.count("plan") == 1
    assert node_names.count("research") == 2  # one per subquestion, run in parallel
    assert node_names.count("verify") == 1
    assert node_names.count("write") == 1
    assert tracer.trace.total_latency_ms > 0


def test_pipeline_is_deterministic_given_the_same_mocks(mock_llm, mock_search):
    report1, _ = run_research(mock_llm, mock_search, "What was the population of Berlin in 2024?", run_id="a")
    report2, _ = run_research(mock_llm, mock_search, "What was the population of Berlin in 2024?", run_id="b")
    assert report1.summary == report2.summary
    assert [s.content for s in report1.sections] == [s.content for s in report2.sections]
