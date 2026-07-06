from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from agent.agent import run_agent
from eval.judge import judge_answer

GOLDEN_PATH = Path(__file__).parent / "golden_dataset.jsonl"
RESULTS_PATH = Path("logs/eval_results.csv")

PRICE_TABLE_PER_MTOK = {
    "anthropic/claude-3.5-haiku": {"input": 0.8, "output": 4.0},
    "openai/gpt-4o-mini": {"input": 0.15, "output": 0.6},
    "meta-llama/llama-3.1-70b-instruct": {"input": 0.35, "output": 0.4},
    "deepseek/deepseek-v3": {"input": 0.27, "output": 1.1},
}

def load_golden_dataset() -> list[dict]:
    return [json.loads(line) for line in GOLDEN_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]

def estimate_cost(model: str, prompt_tokens: int | None, completion_tokens: int | None) -> float | None:
    prices = PRICE_TABLE_PER_MTOK.get(model)
    if prices is None or prompt_tokens is None or completion_tokens is None:
        return None
    return (prompt_tokens * prices["input"] + completion_tokens * prices["output"]) / 1_000_000

def run_eval_for_model(model: str, dataset: list[dict]) -> list[dict]:
    rows = []
    for item in dataset:
        start = time.perf_counter()
        result = run_agent(item["question"], model=model)
        latency = time.perf_counter() - start

        usage = result.get("usage")
        prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
        completion_tokens = getattr(usage, "completion_tokens", None) if usage else None

        verdict = judge_answer(
            question=item["question"],
            expected_answer=item["expected_answer"],
            agent_answer=result["answer"] or "",
        )

        rows.append(
            {
                "model": model,
                "question": item["question"],
                "agent_answer": result["answer"],
                "judge_score": verdict.score,
                "is_hallucination": verdict.is_hallucination,
                "judge_rationale": verdict.rationale,
                "latency_seconds": round(latency, 3),
                "rounds_used": result["rounds_used"],
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "estimated_cost_usd": estimate_cost(model, prompt_tokens, completion_tokens),
            }
        )
    return rows

def print_summary(rows: list[dict]):
    from collections import defaultdict

    by_model: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)

    print("\n=== Сводка по моделям ===")
    print(f"{'model':<40} {'avg_score':>10} {'hallucination%':>15} {'avg_latency_s':>14} {'avg_cost_usd':>13}")
    for model, model_rows in by_model.items():
        avg_score = sum(r["judge_score"] for r in model_rows) / len(model_rows)
        hallucination_pct = sum(r["is_hallucination"] for r in model_rows) / len(model_rows) * 100
        avg_latency = sum(r["latency_seconds"] for r in model_rows) / len(model_rows)
        costs = [r["estimated_cost_usd"] for r in model_rows if r["estimated_cost_usd"] is not None]
        avg_cost = sum(costs) / len(costs) if costs else None
        print(
            f"{model:<40} {avg_score:>10.2f} {hallucination_pct:>14.1f}% {avg_latency:>14.2f} "
            f"{(f'{avg_cost:.5f}' if avg_cost is not None else 'n/a'):>13}"
        )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--compare",
        type=str,
        default=None,
        help="Список моделей через запятую для сравнения, например: "
        "anthropic/claude-3.5-haiku,openai/gpt-4o-mini",
    )
    args = parser.parse_args()

    dataset = load_golden_dataset()
    models = [m.strip() for m in args.compare.split(",")] if args.compare else [None]

    all_rows: list[dict] = []
    for model in models:
        from common.llm_client import AGENT_MODEL

        resolved_model = model or AGENT_MODEL
        print(f"Прогоняю golden dataset на модели: {resolved_model} ({len(dataset)} примеров)...")
        rows = run_eval_for_model(resolved_model, dataset)
        all_rows.extend(rows)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nПодробные результаты сохранены в {RESULTS_PATH}")
    print_summary(all_rows)

if __name__ == "__main__":
    main()
