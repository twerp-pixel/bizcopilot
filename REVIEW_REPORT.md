# BizCopilot Review Report

## Scope

Reviewed the uploaded project ZIP, including the orchestrator, provider client,
SQL/retrieval/statistics tools, FastAPI entry point, evaluation harness, tests,
synthetic data generator/database, setup docs, and configuration. No `.env`,
API keys, virtual environment, `.git`, or generated evaluation results were
included in the returned archive.

## Changes made

- Made `rank_bm25` a lazy import so orchestrator unit tests can import without
  retrieval dependencies installed; retrieval reports an actionable install
  message if the dependency is missing.
- Opened SQLite using read-only URI mode as defense in depth, in addition to
  existing query validation.
- Added synthesis guidance for failed tool steps and mismatched evidence scope.
- Added regression tests for signed percentage values and ordinary two-decimal
  rounding.
- Added planner/timing data to saved end-to-end evaluation details for diagnosis.
- Replaced an environment-specific frozen dependency list (including numerous
  CUDA/NVIDIA packages) with direct application dependencies and compatible
  version ranges.
- Updated provider setup docs and `.env.example` for Anthropic, Gemini, and Groq.
- Documented verifier and synthetic inventory limitations without changing
  evaluation metrics or claiming semantic grounding is solved.

## Checks actually run in this environment

- `python3 -m compileall -q agent tools eval app tests data` — passed.
- `python3 -m unittest discover -s tests -t . -v` — 7 tests passed.
- `python3 tools/sql_tool.py` — passed; schema and sample query returned results.
- `python3 -m eval.eval_harness --retrieval-only` — could not complete because
  `rank-bm25` is not installed in this environment. Installing dependencies
  was blocked by unavailable network/DNS. The missing package is listed in
  `requirements.txt`; rerun locally after installation.
- Full end-to-end evaluation was not run. It requires the configured provider
  SDK/dependencies and a working API key; no API key was requested or exposed.

The execution environment emitted a spreadsheet-runtime warmup traceback during
Python startup, but the compile and unit-test commands themselves returned
success. The unit-test result was `OK` (7 tests).

## Run locally on EndeavourOS/Linux

From the project directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` locally: set `LLM_PROVIDER` to `anthropic`, `gemini`, or `groq`,
then add the matching API key. Do not paste API keys into chat or commit `.env`.

Run checks:

```bash
python3 -m compileall -q agent tools eval app tests data
python3 -m unittest discover -s tests -t . -v
python3 -m eval.eval_harness --retrieval-only
python3 -m eval.eval_harness
```

The last command makes live API calls. Review `eval/results/end_to_end_results.json`
after it finishes. Do not publish metrics until the run is stable and the
memos have been inspected for causal overclaims, scope mismatches, and incorrect
numeric associations.

## Remaining limitations

The numeric verifier only checks whether a number appears in tool output; it
does not semantically bind each number to its claim. The synthesis prompt helps
but does not replace stronger claim-level verification. The synthetic database
also has SKU-1004 on-hand inventory reaching zero from week 5 onward while
recorded sales remain positive. This is preserved and documented rather than
silently changing dataset semantics. No live LLM behavior or retrieval metric
is claimed as verified here.
