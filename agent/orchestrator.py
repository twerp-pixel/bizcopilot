"""
BizCopilot agent orchestrator.

Pipeline: PLAN -> EXECUTE -> SYNTHESIZE -> VERIFY (retry once if ungrounded).

This is intentionally hand-rolled (not hidden behind a framework) so the
control flow is fully legible for an interview walkthrough:

  1. Planner (LLM):   question -> ordered list of tool calls
  2. Executor (code): runs each planned tool call, collects results + provenance
  3. Synthesizer (LLM): tool results -> structured memo, citing sources
  4. Verifier (code): extracts numeric claims from the memo, checks each one
     appears in the executor's tool results (grounding check). If claims
     fail, the synthesizer is called once more with the failures flagged.
"""
import re
import json
import time

from tools import sql_tool, retrieval_tool, stats_tool
from agent.llm_client import call_llm, call_llm_json

PLANNER_SYSTEM = f"""You are the planning module of a retail business-analyst AI agent.
Given a business question, output a JSON plan: a list of steps, each an object with
"tool" (one of "sql", "retrieve", "stats") and "args".

Available tools:
- sql: args {{"query": "<SELECT statement>"}}. Schema:
{sql_tool.get_schema()}
- retrieve: args {{"query": "<natural language search over analyst notes/news>"}}
- stats: args {{"operation": "pct_change"|"zscores", "values": [list of numbers]}} for
  pct_change use args {{"operation":"pct_change","old":X,"new":Y}}

Rules:
- Always include at least one "sql" step to get the actual numbers, and at least one
  "retrieve" step to check for a documented explanation.
- Keep the plan to 2-4 steps. Return ONLY a JSON object: {{"steps": [...]}}. No prose.
"""

SYNTH_SYSTEM = """You are a retail business analyst writing a short memo for a
category manager who is NOT technical. Given the user's question and the results
gathered by tools (SQL rows and retrieved evidence documents), write a memo with
these exact sections:

## Executive Summary
## Diagnosis
## Supporting Evidence
## Recommended Action
## Confidence & Caveats

Rules:
- Every number you state (units, %, revenue, week) MUST come directly from the
  provided tool results. Do not invent or round in a way that changes the number.
- In "Supporting Evidence", cite sources explicitly, e.g. "(source: sales table,
  SKU-1004/Midwest/week 9)" or "(source: note_001)".
- Keep it under 250 words. Plain English, no jargon.
"""

NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?%?")


def _run_step(step):
    tool = step.get("tool")
    args = step.get("args", {})
    if tool == "sql":
        rows = sql_tool.run_sql(args["query"])
        return {"tool": "sql", "query": args["query"], "result": rows}
    elif tool == "retrieve":
        hits = retrieval_tool.retrieve(args["query"], k=3)
        return {"tool": "retrieve", "query": args["query"], "result": hits}
    elif tool == "stats":
        op = args.get("operation")
        if op == "pct_change":
            val = stats_tool.pct_change(args["old"], args["new"])
            return {"tool": "stats", "op": op, "result": val}
        elif op == "zscores":
            val = stats_tool.zscores(args["values"])
            return {"tool": "stats", "op": op, "result": val}
        else:
            return {"tool": "stats", "op": op, "result": None, "error": "unknown operation"}
    else:
        return {"tool": tool, "result": None, "error": "unknown tool"}


def _collect_known_numbers(tool_results):
    """Every number that legitimately appears anywhere in the tool results."""
    known = set()
    for r in tool_results:
        blob = json.dumps(r.get("result"))
        for match in NUMBER_RE.findall(blob):
            known.add(match.rstrip("%"))
    return known


def _verify(memo_text, tool_results):
    """Grounding check: every number in the memo should trace back to tool output.
    Returns (is_grounded, list_of_unverified_numbers)."""
    known = _collect_known_numbers(tool_results)
    claimed = set(m.rstrip("%") for m in NUMBER_RE.findall(memo_text))
    # ignore tiny numbers likely to be section artifacts (e.g. list markers) - keep it simple:
    unverified = [c for c in claimed if c not in known and len(c) > 1]
    return (len(unverified) == 0), unverified


def plan(question: str):
    return call_llm_json(PLANNER_SYSTEM, question, max_tokens=2000)

def execute(plan_obj):
    results = []
    for step in plan_obj.get("steps", []):
        try:
            results.append(_run_step(step))
        except Exception as e:
            results.append({"tool": step.get("tool"), "error": str(e)})
    return results


def synthesize(question, tool_results, retry_note=None):
    user_prompt = f"User question: {question}\n\nTool results (JSON):\n{json.dumps(tool_results, indent=2)}"
    if retry_note:
        user_prompt += f"\n\nNOTE: Your previous draft included unverifiable numbers: {retry_note}. Only use numbers that literally appear in the tool results above."
    return call_llm(SYNTH_SYSTEM, user_prompt, max_tokens=700)


def answer(question: str, allow_retry: bool = True):
    t0 = time.time()
    plan_obj = plan(question)
    tool_results = execute(plan_obj)
    memo = synthesize(question, tool_results)

    grounded, unverified = _verify(memo, tool_results)
    retried = False
    if not grounded and allow_retry:
        memo = synthesize(question, tool_results, retry_note=unverified)
        grounded, unverified = _verify(memo, tool_results)
        retried = True

    return {
        "question": question,
        "plan": plan_obj,
        "tool_results": tool_results,
        "memo": memo,
        "grounded": grounded,
        "unverified_numbers": unverified,
        "retried": retried,
        "latency_sec": round(time.time() - t0, 2),
    }


if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "Why did SKU-1004 sales drop in the Midwest around week 9?"
    result = answer(q)
    print(json.dumps({k: v for k, v in result.items() if k != "memo"}, indent=2))
    print("\n--- MEMO ---\n")
    print(result["memo"])
