"""The LangGraph orchestration: plan -> research (parallel fan-out) -> verify -> write.

    START -> plan --Send x N--> research_one --(barrier)--> verify -> write -> END

`plan` decomposes the question into subquestions, then a conditional edge
emits one `Send` per subquestion so `research_one` runs those searches
concurrently instead of one after another -- for a 4-subquestion run this
is roughly a 3-4x wall-clock improvement over a sequential chain, which is
the actual reason to reach for LangGraph's graph model here instead of a
plain for-loop.

Every node is wrapped in the run tracer so the frontend and eval harness
get a step-by-step latency/token breakdown "for free".
"""

from __future__ import annotations

import operator
import uuid
from typing import Annotated, Optional, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from .agents.planner import plan as plan_agent
from .agents.researcher import research as research_agent
from .agents.verifier import verify_all
from .agents.writer import write_report
from .llm import LLMClient
from .schemas import Evidence, ResearchReport, SearchResult, SubQuestion, VerifiedClaim
from .search import SearchClient
from .tracing import RunTracer


def _merge_dicts(a: dict, b: dict) -> dict:
    merged = dict(a)
    merged.update(b)
    return merged


class ResearchState(TypedDict):
    run_id: str
    question: str
    max_subquestions: int
    max_sources_per_subquestion: int
    subquestions: list[SubQuestion]
    evidence: Annotated[list[Evidence], operator.add]
    search_results_by_subquestion: Annotated[dict[str, list[SearchResult]], _merge_dicts]
    verified_claims: list[VerifiedClaim]
    report: Optional[ResearchReport]


class ResearchOneInput(TypedDict):
    run_id: str
    subquestion: SubQuestion
    max_sources_per_subquestion: int


def build_graph(llm: LLMClient, search: SearchClient, tracer: RunTracer):
    def plan_node(state: ResearchState) -> dict:
        with tracer.step("plan", "Decomposing question into subquestions") as rec:
            subquestions = plan_agent(llm, state["question"], state["max_subquestions"])
            rec["detail"] = "; ".join(sq.text for sq in subquestions)
        return {"subquestions": subquestions}

    def route_to_research(state: ResearchState) -> list[Send]:
        return [
            Send(
                "research_one",
                {
                    "run_id": state["run_id"],
                    "subquestion": sq,
                    "max_sources_per_subquestion": state["max_sources_per_subquestion"],
                },
            )
            for sq in state["subquestions"]
        ]

    def research_one_node(input: ResearchOneInput) -> dict:
        sq = input["subquestion"]
        with tracer.step("research", f"Researching: {sq.text}") as rec:
            evidence, results = research_agent(
                llm, search, sq, input["max_sources_per_subquestion"]
            )
            rec["detail"] = f"{len(results)} sources, {len(evidence)} claims extracted"
        return {
            "evidence": evidence,
            "search_results_by_subquestion": {sq.id: results},
        }

    def verify_node(state: ResearchState) -> dict:
        with tracer.step("verify", "Cross-checking claims across sources") as rec:
            subquestion_text_by_id = {sq.id: sq.text for sq in state["subquestions"]}
            valid_urls_by_subquestion = {
                sq_id: {r.url for r in results}
                for sq_id, results in state["search_results_by_subquestion"].items()
            }
            verified = verify_all(
                llm, state["evidence"], subquestion_text_by_id, valid_urls_by_subquestion
            )
            supported = sum(1 for v in verified if v.verdict.value == "supported")
            rec["detail"] = f"{supported}/{len(verified)} claims verified as supported"
        return {"verified_claims": verified}

    def write_node(state: ResearchState) -> dict:
        with tracer.step("write", "Writing final cited report") as rec:
            evidence_by_id = {e.id: e for e in state["evidence"]}
            report = write_report(
                llm,
                state["run_id"],
                state["question"],
                state["subquestions"],
                evidence_by_id,
                state["verified_claims"],
            )
            rec["detail"] = f"{len(report.sections)} sections, {len(report.citations)} citations"
        return {"report": report}

    graph = StateGraph(ResearchState)
    graph.add_node("plan", plan_node)
    graph.add_node("research_one", research_one_node)
    graph.add_node("verify", verify_node)
    graph.add_node("write", write_node)

    graph.add_edge(START, "plan")
    graph.add_conditional_edges("plan", route_to_research, ["research_one"])
    graph.add_edge("research_one", "verify")
    graph.add_edge("verify", "write")
    graph.add_edge("write", END)

    return graph.compile()


def run_research(
    llm: LLMClient,
    search: SearchClient,
    question: str,
    max_subquestions: int = 4,
    max_sources_per_subquestion: int = 4,
    run_id: Optional[str] = None,
) -> tuple[ResearchReport, RunTracer]:
    """Synchronous convenience wrapper used by the eval harness, tests, and the CLI."""
    run_id = run_id or f"run_{uuid.uuid4().hex[:10]}"
    tracer = RunTracer(run_id=run_id, model=llm.model_name)
    graph = build_graph(llm, search, tracer)
    final_state = graph.invoke(
        {
            "run_id": run_id,
            "question": question,
            "max_subquestions": max_subquestions,
            "max_sources_per_subquestion": max_sources_per_subquestion,
            "evidence": [],
            "search_results_by_subquestion": {},
            "verified_claims": [],
            "report": None,
        }
    )
    return final_state["report"], tracer
