"""FastAPI app: streams a research run's progress live over SSE, then the
final cited report, and exposes the per-run trace for the observability
panel in the frontend.

Runs are executed with LangGraph's `.stream()` (not `.invoke()`) precisely
so the frontend can show "planning...", "researching subquestion 2 of
4...", "verifying...", "writing..." as they happen instead of a blank
spinner for 20+ seconds, which is what most weekend-project demos ship.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

from .config import build_llm, build_search
from .graph import build_graph
from .schemas import ResearchRequest
from .tracing import RunTracer

app = FastAPI(title="Deep Research Agent", version="0.1.0")

_RUNS: dict[str, dict] = {}  # run_id -> {"report": dict, "trace": dict}  (in-memory demo store)

_FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend"
if _FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_FRONTEND_DIR)), name="static")


@app.get("/")
def index():
    index_file = _FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"service": "deep-research-agent", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    run = _RUNS.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="run not found")
    return run


async def _stream_events(request: ResearchRequest) -> AsyncIterator[dict]:
    run_id = f"run_{uuid.uuid4().hex[:10]}"
    llm = build_llm()
    search = build_search()
    tracer = RunTracer(run_id=run_id, model=llm.model_name)
    graph = build_graph(llm, search, tracer)

    yield {"event": "run_started", "data": json.dumps({"run_id": run_id})}

    initial_state = {
        "run_id": run_id,
        "question": request.question,
        "max_subquestions": request.max_subquestions,
        "max_sources_per_subquestion": request.max_sources_per_subquestion,
        "evidence": [],
        "search_results_by_subquestion": {},
        "verified_claims": [],
        "report": None,
    }

    final_state = None
    seen_steps = 0
    async for state_update in graph.astream(initial_state, stream_mode="values"):
        final_state = state_update
        # Emit any trace steps recorded since the last update.
        while seen_steps < len(tracer.trace.steps):
            step = tracer.trace.steps[seen_steps]
            yield {"event": "step", "data": step.model_dump_json()}
            seen_steps += 1

    report = final_state["report"] if final_state else None
    _RUNS[run_id] = {"report": report.model_dump() if report else None, "trace": tracer.to_dict()}

    yield {
        "event": "done",
        "data": json.dumps(
            {
                "run_id": run_id,
                "report": report.model_dump() if report else None,
                "trace": tracer.to_dict(),
            }
        ),
    }


@app.post("/api/research")
async def research(request: ResearchRequest):
    return EventSourceResponse(_stream_events(request))
