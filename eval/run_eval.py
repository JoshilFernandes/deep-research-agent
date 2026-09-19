#!/usr/bin/env python3
"""Evaluation harness: runs the full agent pipeline over `benchmark.jsonl`
and reports structural quality metrics. This is what runs in CI on every
push (`--mock`, offline, deterministic) and what you'd run locally with a
live key (`--live`) before shipping a prompt or graph change.

Usage:
    python eval/run_eval.py --mock                 # CI mode, no network, no API cost
    python eval/run_eval.py --live                 # uses GROQ_API_KEY / TAVILY_API_KEY from .env

Exit code is non-zero if any run fails or the citation-integrity invariant
is violated, so this can gate a CI job.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from deep_research.graph import run_research  # noqa: E402
from deep_research.schemas import ResearchReport  # noqa: E402
from mocks import GenericMockSearch, build_generic_mock_llm  # noqa: E402

BENCHMARK_PATH = Path(__file__).resolve().parent / "benchmark.jsonl"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def load_benchmark() -> list[dict]:
    with open(BENCHMARK_PATH) as f:
        return [json.loads(line) for line in f if line.strip()]


def check_uncited_prose(report: ResearchReport) -> list[str]:
    """Structural invariant: a section with real prose content must carry
    at least one citation, OR be the honest fallback string. A section
    with neither is the one failure mode this whole project exists to
    prevent -- confident-sounding prose with nothing backing it.
    """
    problems = []
    fallback = "No sufficiently verified information was found for this subquestion."
    for section in report.sections:
        if section.content.strip() == fallback:
            continue
        if not section.citation_ids:
            problems.append(f"Section {section.heading!r} has prose but zero citations")
    return problems


def run_benchmark(mode: str) -> dict:
    items = load_benchmark()
    if mode == "mock":
        llm = build_generic_mock_llm()
        search = GenericMockSearch()
    else:
        from deep_research.config import build_llm, build_search

        llm = build_llm()
        search = build_search()

    rows = []
    failures = []

    for item in items:
        t0 = time.perf_counter()
        try:
            report, tracer = run_research(
                llm, search, item["question"], max_subquestions=item.get("min_subquestions", 2)
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000
            uncited = check_uncited_prose(report)
            n_sections_with_citations = sum(1 for s in report.sections if s.citation_ids)
            domains = {c.url.split("/")[2] for c in report.citations if "/" in c.url}
            rows.append(
                {
                    "id": item["id"],
                    "question": item["question"],
                    "ok": not uncited,
                    "sections": len(report.sections),
                    "sections_with_citations": n_sections_with_citations,
                    "citation_coverage": round(n_sections_with_citations / max(1, len(report.sections)), 2),
                    "unique_source_domains": len(domains),
                    "limitations_flagged": len(report.limitations),
                    "latency_ms": round(elapsed_ms, 1),
                    "tokens": tracer.trace.total_tokens,
                    "problems": uncited,
                }
            )
            if uncited:
                failures.append(f"{item['id']}: {uncited}")
        except Exception as e:  # noqa: BLE001 - eval harness must record, not crash, on a bad run
            elapsed_ms = (time.perf_counter() - t0) * 1000
            rows.append({"id": item["id"], "question": item["question"], "ok": False, "error": str(e), "latency_ms": round(elapsed_ms, 1)})
            failures.append(f"{item['id']}: exception: {e}")

    n_ok = sum(1 for r in rows if r.get("ok"))
    summary = {
        "mode": mode,
        "model": getattr(llm, "model_name", "unknown"),
        "total": len(rows),
        "passed": n_ok,
        "pass_rate": round(n_ok / len(rows), 2) if rows else 0.0,
        "avg_latency_ms": round(sum(r.get("latency_ms", 0) for r in rows) / len(rows), 1) if rows else 0.0,
        "avg_citation_coverage": round(
            sum(r.get("citation_coverage", 0) for r in rows if "citation_coverage" in r)
            / max(1, sum(1 for r in rows if "citation_coverage" in r)),
            2,
        ),
        "rows": rows,
        "failures": failures,
    }
    return summary


def write_report(summary: dict) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "latest.md"
    lines = [
        f"# Eval report ({summary['mode']} mode, model={summary['model']})",
        "",
        f"- Pass rate: **{summary['passed']}/{summary['total']}** ({summary['pass_rate'] * 100:.0f}%)",
        f"- Avg latency: {summary['avg_latency_ms']} ms/run",
        f"- Avg citation coverage: {summary['avg_citation_coverage'] * 100:.0f}%",
        "",
        "| id | question | ok | sections | citation coverage | domains | limitations flagged | latency (ms) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in summary["rows"]:
        lines.append(
            f"| {r['id']} | {r['question'][:60]} | {'✅' if r.get('ok') else '❌'} | "
            f"{r.get('sections', '-')} | {r.get('citation_coverage', '-')} | "
            f"{r.get('unique_source_domains', '-')} | {r.get('limitations_flagged', '-')} | {r.get('latency_ms', '-')} |"
        )
    if summary["failures"]:
        lines += ["", "## Failures", ""] + [f"- {f}" for f in summary["failures"]]
    out_path.write_text("\n".join(lines) + "\n")
    (RESULTS_DIR / "latest.json").write_text(json.dumps(summary, indent=2))
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--mock", action="store_true", help="Run offline against deterministic mocks (CI mode).")
    group.add_argument("--live", action="store_true", help="Run against real GROQ_API_KEY / TAVILY_API_KEY.")
    args = parser.parse_args()

    summary = run_benchmark("mock" if args.mock else "live")
    out_path = write_report(summary)

    print(f"\nEval report written to {out_path}")
    print(f"Pass rate: {summary['passed']}/{summary['total']} ({summary['pass_rate'] * 100:.0f}%)")
    print(f"Avg citation coverage: {summary['avg_citation_coverage'] * 100:.0f}%")

    return 0 if summary["passed"] == summary["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
