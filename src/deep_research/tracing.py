"""Lightweight run tracer.

Every node in the graph wraps its work in `tracer.step(...)`. This is the
project's observability layer: it costs nothing extra (no external APM
dependency) but gives the frontend and the eval harness a per-node
breakdown of latency and token usage for every run, which is exactly the
kind of visibility a production LLM system needs and a toy demo skips.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from datetime import datetime, timezone

from .schemas import RunTrace, TraceStep


class RunTracer:
    def __init__(self, run_id: str, model: str):
        self.trace = RunTrace(run_id=run_id, model=model)

    @contextmanager
    def step(self, node: str, label: str):
        started = datetime.now(timezone.utc).isoformat()
        t0 = time.perf_counter()
        record = {"tokens_in": 0, "tokens_out": 0, "detail": ""}
        try:
            yield record
        finally:
            latency_ms = (time.perf_counter() - t0) * 1000
            self.trace.steps.append(
                TraceStep(
                    node=node,
                    label=label,
                    started_at=started,
                    ended_at=datetime.now(timezone.utc).isoformat(),
                    latency_ms=round(latency_ms, 1),
                    tokens_in=record.get("tokens_in", 0),
                    tokens_out=record.get("tokens_out", 0),
                    detail=record.get("detail", ""),
                )
            )

    def to_dict(self) -> dict:
        return self.trace.model_dump()
