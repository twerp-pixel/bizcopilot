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
from decimal import Decimal

PLANNER_SYSTEM = f"""Create a JSON execution plan for retail analysis.

Database schema:
{sql_tool.get_schema()}

Tools:
- sql: {{"query":"..."}}
- retrieve: {{"query":"..."}}

Return valid JSON only:
{{"steps":[{{"tool":"sql","args":{{"query":"..."}}}},{{"tool":"retrieve","args":{{"query":"..."}}}}]}}

Rules:
- Return exactly two steps: one SQL, one retrieve.
- SQL must be read-only.
- For regional trends, aggregate sales across stores by week.
- When combining sales, inventory, and promotions, aggregate each
  table separately in CTEs before joining on week.
- Use LEFT JOINs between aggregated tables.
- Keep the SQL query focused on the weeks relevant to the question.
- Do not add stats steps; Calculate percentage changes using:
  100.0 * (current - previous) / NULLIF(previous, 0).
- Ensure division uses floating-point arithmetic, not integer division.
- When comparing weeks, verify that the calculated percentage agrees
  with the underlying values before returning the result.
- Treat promotions and retrieved notes as possible explanations,
  not proof of causation or proven causes.
- Do not infer that stockouts did not occur from inventory values alone.
- Never report a percentage change that contradicts the underlying values.
- Report zero inventory as a stockout risk, not proof that a stockout occurred.
- Every step in "steps" must be an object with "tool" and "args" fields.
- Include exactly one SQL step and one retrieve step; never put strings
  such as "tool" or ":" directly in the steps array.
- Output complete JSON, with no reasoning or extra text.
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
- For percentage changes, preserve the sign shown in the tool results.
  For example, report -30.91%, if the source value is -30.91.
- Do not convert signed values into unsigned percentages.
- Do not claim that inventory levels prove whether stockouts occurred.
  Describe zero inventory as a stockout risk, not proof of an actual stockout.
- Do not present promotions, supply delays, or other factors as proven causes
  unless the evidence directly establishes causation.
- Do not invent inventory thresholds, targets, or numerical recommendations.
  Recommend a threshold only when the tool results explicitly support it.
- Distinguish observed facts from hypotheses and recommendations.
- If the evidence is insufficient to identify a cause, say so explicitly.
- Base each conclusion on the relevant SKU, region, and week; do not combine unrelated values merely because they appear in the tool results.
- If a source percentage is negative, include its negative sign explicitly.
  Write "-30.91%" rather than "fell by 30.91%" when reporting that value.
- Do not number recommendations; use bullet points instead.
- Do not invent numerical thresholds, targets, or safety-stock levels.
- Recommend numerical thresholds only when explicitly supported by the
  provided tool results.
- Future monitoring recommendations may mention future weeks, but do not
  present them as observed results.
- Distinguish observed facts from possible explanations and hypotheses.
- Temporal alignment alone does not establish causation.
- A tool result containing an "error" means that step failed. Do not treat
  missing results as evidence or invent a substitute result. State the
  limitation when it affects the answer, and base claims only on successful
  tool results.
- When structured data and documents cover different scopes (for example,
  one region versus all regions), do not generalize beyond the narrower
  evidence.

"""


NUMBER_RE = re.compile(r"(?<![\w.])-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?(?![\w])")


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


def _extract_numbers(text):
    """Extract complete numeric values, excluding SKU identifiers."""
    text = text.translate(str.maketrans({
    "\u2010": "-",  # hyphen
    "\u2011": "-",  # non-breaking hyphen
    "\u2012": "-",  # figure dash
    "\u2013": "-",  # en dash
    "\u2014": "-",  # em dash
    }))
    text = re.sub(r"\bSKU-\d+\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"(?m)^\s*(?:[-*]\s*)?\d+[.)]\s+", "", text)
    return {
    format(Decimal(match.rstrip("%").replace(",", "")), "f")
    for match in NUMBER_RE.findall(text)
    }


def _collect_known_numbers(tool_results):
    """Collect numeric values present in the tool results."""
    known = set()

    for result in tool_results:
        blob = json.dumps(result.get("result"), ensure_ascii=False)
        known.update(_extract_numbers(blob))

    return known


def _verify(memo_text, tool_results):
    """Check numeric claims against tool results, allowing display precision."""
    known = _collect_known_numbers(tool_results)
    claimed = _extract_numbers(memo_text)

    known_values = [Decimal(n) for n in known]
    unverified = []

    for number in claimed:

        claim = Decimal(number)

        # Integers must match exactly; decimals allow normal display precision.
        if "." in number:
            tolerance = Decimal("0.005")
        else:
            tolerance = Decimal("0")

        matched = any(
            abs(claim - value) <= tolerance
            for value in known_values
        )

        if not matched:
            unverified.append(number)

    return len(unverified) == 0, sorted(unverified)

def _validate_plan(plan_obj):
    """Reject valid JSON that is not a valid execution plan."""
    if not isinstance(plan_obj, dict):
        raise ValueError("Plan must be a JSON object.")

    steps = plan_obj.get("steps")
    if not isinstance(steps, list) or len(steps) != 2:
        raise ValueError("Plan must contain exactly two steps.")

    tools_seen = []

    for step in steps:
        if not isinstance(step, dict):
            raise ValueError("Every plan step must be an object.")

        tool = step.get("tool")
        args = step.get("args")

        if tool not in ("sql", "retrieve"):
            raise ValueError(f"Unsupported tool: {tool!r}")

        if not isinstance(args, dict):
            raise ValueError(f"Arguments for {tool!r} must be an object.")

        if not isinstance(args.get("query"), str) or not args["query"].strip():
            raise ValueError(f"{tool!r} requires a non-empty query.")

        tools_seen.append(tool)

    if sorted(tools_seen) != ["retrieve", "sql"]:
        raise ValueError("Plan must contain one SQL step and one retrieve step.")

    return plan_obj


def plan(question: str):
    correction = ""

    for attempt in range(2):
        prompt = question
        if correction:
            prompt += (
                "\n\nYour previous plan had an invalid structure. "
                'Return exactly {"steps":[{"tool":"sql","args":{"query":"..."}},'
                '{"tool":"retrieve","args":{"query":"..."}}]}. '
                "Each steps entry must be an object, not a string."
            )

        try:
            plan_obj = call_llm_json(
                PLANNER_SYSTEM,
                prompt,
                max_tokens=1800,
            )
            return _validate_plan(plan_obj)
        except ValueError as exc:
            if attempt == 1:
                raise ValueError(
                    f"Planner failed validation after two attempts: {exc}"
                ) from exc
            correction = str(exc)

    raise RuntimeError("Planner failed unexpectedly.")


def execute(plan_obj):
    results = []

    for step in plan_obj.get("steps", []):
        tool_name = step.get("tool") if isinstance(step, dict) else None

        try:
            if not isinstance(step, dict):
                raise ValueError("Plan step must be a JSON object.")

            results.append(_run_step(step))
        except Exception as exc:
            results.append({
                "tool": tool_name,
                "error": str(exc),
            })

    return results


def synthesize(question, tool_results, retry_note=None):
    user_prompt = f"User question: {question}\n\nTool results (JSON):\n{json.dumps(tool_results, indent=2)}"
    if retry_note:
        user_prompt += f"\n\nNOTE: Your previous draft included unverifiable numbers: {retry_note}. Only use numbers that literally appear in the tool results above."
    return call_llm(SYNTH_SYSTEM, user_prompt, max_tokens=700)


def answer(question: str, allow_retry: bool = True):
    t0 = time.perf_counter()

    t = time.perf_counter()
    plan_obj = plan(question)
    plan_sec = time.perf_counter() - t

    t = time.perf_counter()
    tool_results = execute(plan_obj)
    execute_sec = time.perf_counter() - t

    t = time.perf_counter()
    memo = synthesize(question, tool_results)
    synthesize_sec = time.perf_counter() - t

    t = time.perf_counter()
    grounded, unverified = _verify(memo, tool_results)
    verify_sec = time.perf_counter() - t

    retried = False
    retry_sec = 0.0

    if not grounded and allow_retry:
        t = time.perf_counter()
        memo = synthesize(question, tool_results, retry_note=unverified)
        retry_sec = time.perf_counter() - t

        t = time.perf_counter()
        grounded, unverified = _verify(memo, tool_results)
        verify_sec += time.perf_counter() - t
        retried = True

    return {
        "question": question,
        "plan": plan_obj,
        "tool_results": tool_results,
        "memo": memo,
        "grounded": grounded,
        "unverified_numbers": unverified,
        "retried": retried,
        "latency_sec": round(time.perf_counter() - t0, 2),
        "timings": {
            "plan_sec": round(plan_sec, 2),
            "execute_sec": round(execute_sec, 2),
            "synthesize_sec": round(synthesize_sec, 2),
            "retry_sec": round(retry_sec, 2),
            "verify_sec": round(verify_sec, 4),
        },
    }


if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "Why did SKU-1004 sales drop in the Midwest around week 9?"
    result = answer(q)
    print(json.dumps({k: v for k, v in result.items() if k != "memo"}, indent=2))
    print("\n--- MEMO ---\n")
    print(result["memo"])
