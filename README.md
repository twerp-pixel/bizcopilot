# BizCopilot — Agentic Business Analyst Copilot for Retail Decision Support

## The problem
A retail/CPG category manager needs to diagnose sales or inventory anomalies
every week (e.g. "why did SKU-1004 drop 30% in the Midwest?"), but doesn't
have time to query dashboards or read every analyst note and competitor
update. BizCopilot answers the question directly, in plain English, with
every number traced back to a real source — a SQL query result or a
retrieved document — rather than a plausible-sounding hallucination.

## Architecture

```
User question
     │
     ▼
[ Planner ]  (LLM) — decomposes the question into a short tool-call plan
     │
     ▼
[ Executor ] (code) — runs each planned step:
     ├── sql_tool       → structured sales/inventory/promotions data (SQLite)
     ├── retrieval_tool → hybrid BM25 + embedding search over analyst notes
     └── stats_tool     → % change / z-score anomaly checks
     │
     ▼
[ Synthesizer ] (LLM) — writes the memo, citing sources for every claim
     │
     ▼
[ Verifier ] (code) — extracts every number in the memo, checks it actually
     appears in the tool results. If not grounded, re-runs the synthesizer
     once with the failing numbers flagged.
     │
     ▼
Final memo: Executive Summary → Diagnosis → Evidence → Recommended Action → Caveats
```

Hybrid retrieval fuses BM25 (keyword) and sentence-embedding (semantic)
rankings via Reciprocal Rank Fusion — the same retrieval technique used in
my other project, RepoLens, reused here on a business-document corpus
instead of a codebase.

## Why this design
- **Grounding is enforced in code, not just prompted for.** The verifier
  step doesn't trust the LLM to self-report accuracy — it mechanically
  checks every number in the output against the actual tool results.
- **The agent is hand-rolled, not framework-hidden.** Planner/executor/
  synthesizer/verifier are plain Python functions so the control flow is
  fully inspectable — useful both for debugging and for explaining the
  design in an interview.
- **The dataset has known ground truth.** Three anomalies are deliberately
  injected into the synthetic data with a matching "explanation" document,
  so retrieval and end-to-end quality can be measured against a real answer
  key instead of eyeballing outputs.

## Evaluation

Run `python3 -m eval.eval_harness --retrieval-only` for retrieval-only
metrics (no API key needed), or drop `--retrieval-only` for the full
end-to-end run (requires `ANTHROPIC_API_KEY`, makes real API calls).

| Metric | Result |
|---|---|
| Retrieval Recall@3 | *fill in after running eval* |
| Retrieval MRR | *fill in after running eval* |
| End-to-end groundedness rate | *fill in after running eval* |
| Avg. latency per question | *fill in after running eval* |

*(Numbers intentionally left blank here — fill them in with your actual
run output once you have an API key set up. Don't put in a number you
didn't measure.)*

## Tech stack
Python · FastAPI · Anthropic Claude API · sentence-transformers
(all-MiniLM-L6-v2) · rank-bm25 · SQLite · vanilla HTML/JS frontend

## Repo structure
```
bizcopilot/
  data/           synthetic data generator, SQLite DB, analyst-note corpus
  tools/          sql_tool, retrieval_tool, stats_tool
  agent/          orchestrator (planner/executor/synth/verify), llm_client
  eval/           test question set + eval harness
  app/            FastAPI backend + static chat UI
  Dockerfile      for deployment
  run.sh          local dev runner
```

## Running it
See [SETUP.md](SETUP.md) for full local run and deployment instructions.

Quick start:
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then add your ANTHROPIC_API_KEY
./run.sh
# open http://localhost:8000
```
