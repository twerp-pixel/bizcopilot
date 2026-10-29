"""
Evaluation harness for BizCopilot.

Two parts:
1. Retrieval-only eval: Recall@K and MRR of the hybrid retriever against the
   known-correct doc_id for each test question (no LLM needed - fast, free,
   run this constantly while iterating on retrieval).
2. End-to-end eval: runs the full agent (requires a configured LLM provider like ANTHROPIC_API_KEY)
   and reports groundedness rate + latency across the test set.

Run:
  python3 -m eval.eval_harness --retrieval-only     # no API key needed
  python3 -m eval.eval_harness                       # full end-to-end (costs API calls)
"""
import json
import os
import sys
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from tools import retrieval_tool


def load_test_set():
    with open(os.path.join(HERE, "test_questions.json")) as f:
        return json.load(f)


def eval_retrieval(k=3):
    test_set = load_test_set()
    hits_at_k = 0
    reciprocal_ranks = []

    for item in test_set:
        results = retrieval_tool.retrieve(item["question"], k=k)
        doc_ids = [r["doc_id"] for r in results]
        if item["expected_doc_id"] in doc_ids:
            hits_at_k += 1
            rank = doc_ids.index(item["expected_doc_id"]) + 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

    recall_at_k = hits_at_k / len(test_set)
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)

    print(f"Retrieval eval over {len(test_set)} questions (k={k}):")
    print(f"  Recall@{k}: {recall_at_k:.2%}")
    print(f"  MRR:        {mrr:.3f}")
    return {"recall_at_k": recall_at_k, "mrr": mrr, "k": k, "n": len(test_set)}


def eval_end_to_end():
    from agent import orchestrator

    test_set = load_test_set()
    results = []
    grounded_count = 0
    total_latency = 0.0

    for item in test_set:
        r = orchestrator.answer(item["question"])
        grounded = r["grounded"]
        grounded_count += int(grounded)
        total_latency += r["latency_sec"]
        results.append({
            "id": item["id"],
            "question": item["question"],
            "grounded": grounded,
            "unverified_numbers": r["unverified_numbers"],
            "retried": r["retried"],
            "latency_sec": r["latency_sec"],
            "memo": r.get("memo"),
            "tool_results": r.get("tool_results"),
            "plan": r.get("plan"),
            "timings": r.get("timings"),
        })
        print(f"[{item['id']}] grounded={grounded} retried={r['retried']} latency={r['latency_sec']}s")

    summary = {
        "n": len(test_set),
        "groundedness_rate": grounded_count / len(test_set),
        "avg_latency_sec": round(total_latency / len(test_set), 2),
        "details": results,
    }

    out_path = os.path.join(HERE, "results", "end_to_end_results.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nGroundedness rate: {summary['groundedness_rate']:.2%}")
    print(f"Avg latency:       {summary['avg_latency_sec']}s")
    print(f"Full results written to {out_path}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()

    retrieval_summary = eval_retrieval(k=args.k)

    if not args.retrieval_only:
        eval_end_to_end()
    else:
        print("\n(Skipped end-to-end eval - remove --retrieval-only to run it, requires API KEY)")
