# Deep Research Agent

A multi-agent research assistant that plans, searches, **verifies**, and writes a cited report for any research question — built on LangGraph, with parallel agent fan-out, a hybrid deterministic + LLM verification layer, and an evaluation harness that runs in CI on every push.

```
you ask a question
        │
        ▼
   plan (LLM)  →  breaks it into 2-4 subquestions
        │
        ▼  (fanned out in parallel via LangGraph's Send API)
   research × N →  search the web, extract quoted, citable claims
        │
        ▼
   verify        →  1) deterministic: is this URL one the search actually returned?
        │            2) LLM: do these claims about the same subquestion contradict each other?
        ▼
   write          →  turns only VERIFIED claims into prose; everything else goes into
                      an honest "Limitations" section instead of being hallucinated away
```

## Why this project exists

Most weekend "AI research agent" demos stop at "LLM calls a search tool and writes an answer." The interesting engineering problem — and the actual bar most AI Engineer roles are hiring for in 2026 — is what happens *after* the model has an answer: can you verify it, can you catch it contradicting itself, and can you be honest in the output about what you couldn't confirm? This project's verifier is a first-class agent, not an afterthought, and it's covered by tests that prove it actually catches a deliberately contradictory pair of sources (`tests/conftest.py`, `tests/test_graph_end_to_end.py`).

## What makes this more than a chatbot wrapper

- **Real parallelism, not a sequential chain.** Subquestions are researched concurrently via LangGraph's `Send` API (see `docs/architecture.md`), not in a `for` loop.
- **Hybrid verification.** A non-LLM, structural citation-integrity check runs before any semantic check — a model citing a URL it was never shown is caught deterministically, with zero ambiguity and zero token cost.
- **Honest failure mode.** Unsupported or contradicted claims never make it into the report's prose. They're surfaced verbatim in a "Limitations" section, built without an LLM call so it can't be quietly softened.
- **Observability for free.** Every node is wrapped in a run tracer (latency, tokens, per-step detail); the frontend streams it live over SSE instead of a blank spinner.
- **An eval suite that's actually a CI gate.** `eval/run_eval.py --mock` runs the full pipeline over 8 benchmark questions offline and fails the build if any section has prose with no citation. This is the same mechanism `.github/workflows/ci.yml` runs on every push.
- **Provider-agnostic by design.** `LLMClient` / `SearchClient` are small abstractions; the real implementations (Groq, Tavily/DuckDuckGo) and the deterministic mocks used by every test implement the same interface, so the whole graph is testable without hitting a live API.

## Quickstart

### Option A — Docker (recommended)

```bash
cp .env.example .env        # add your GROQ_API_KEY (free at console.groq.com/keys)
                             # and optionally TAVILY_API_KEY (free at tavily.com)
docker compose up --build
# open http://localhost:8000
```

### Option B — local Python

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pip install -e .
cp .env.example .env        # add your keys
uvicorn deep_research.api:app --reload
# open http://localhost:8000
```

No Tavily key? The app automatically falls back to a no-key DuckDuckGo provider — nothing here requires signing up for anything beyond a free Groq key.

## Running the tests and eval suite

```bash
pytest -v                        # 10 tests, fully offline, ~0.3s
python eval/run_eval.py --mock   # 8-question structural eval, fully offline
python eval/run_eval.py --live   # same benchmark against your real GROQ_API_KEY / TAVILY_API_KEY
```

`eval/results/latest.md` is regenerated with a pass-rate table after every run.

## Project layout

```
src/deep_research/
  schemas.py       # Pydantic contracts every node validates against
  llm.py            # LLMClient abstraction: GroqLLM (real) / MockLLM (offline, scripted)
  search.py         # SearchClient abstraction: Tavily / DuckDuckGo / MockSearch
  tracing.py        # per-run, per-node latency + token tracer
  graph.py           # the LangGraph StateGraph: plan -> research (parallel) -> verify -> write
  agents/            # one module per agent: planner, researcher, verifier, writer
  api.py             # FastAPI app, streams run progress over SSE
frontend/index.html   # single-file UI: live progress, cited report, limitations, trace table
eval/
  benchmark.jsonl     # 8 open-ended research questions
  run_eval.py         # eval harness / CI gate
  mocks.py            # generic offline LLM + search mocks used by --mock
tests/                 # 10 tests: schemas, verifier unit tests, full pipeline, API
.github/workflows/ci.yml
```

## Tech stack

Python · LangGraph · FastAPI · Pydantic v2 · Server-Sent Events · Groq (Llama 3.3 70B) · Tavily · Docker · GitHub Actions

## Roadmap

- [ ] Swap the in-memory run store for Redis/Postgres for multi-instance deployment
- [ ] Add a second LLM provider (OpenAI/Anthropic) to `llm.py` to demonstrate the abstraction is real, not aspirational
- [ ] Persist eval trend history across runs instead of only `latest.md`

## License

MIT — see [LICENSE](LICENSE).
