from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOG_PATH = Path("logs/events.jsonl")

ERROR_TYPES = ["hallucination", "off_topic", "format_error", "tool_misuse", "retrieval_miss", "none"]

def classify_error(agent_result: dict[str, Any], retrieval_empty: bool = False) -> str:
    answer = (agent_result.get("answer") or "").lower()
    if agent_result.get("rounds_used", 0) >= 5 and "лимит итераций" in answer:
        return "tool_misuse"
    if retrieval_empty and "не знаю" not in answer and "не нашл" not in answer:
        return "hallucination"
    if retrieval_empty:
        return "retrieval_miss"
    return "none"

def log_event(
    question: str,
    agent_result: dict[str, Any],
    latency_seconds: float,
    prompt_version: str = "v1",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    retrieval_calls = [tc for tc in agent_result.get("tool_calls_log", []) if tc["name"] == "search_docs"]
    retrieval_empty = any(not tc["result"].get("results") for tc in retrieval_calls) if retrieval_calls else False

    usage = agent_result.get("usage")
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "answer": agent_result.get("answer"),
        "model": agent_result.get("model"),
        "prompt_version": prompt_version,
        "rounds_used": agent_result.get("rounds_used"),
        "tool_calls": agent_result.get("tool_calls_log"),
        "latency_seconds": round(latency_seconds, 3),
        "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
        "completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
        "error_type": classify_error(agent_result, retrieval_empty=retrieval_empty),
    }
    if extra:
        event.update(extra)

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")

    return event

class Timer:

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self._start

if __name__ == "__main__":

    fake_result = {"answer": "тест", "tool_calls_log": [], "rounds_used": 1, "model": "test/model", "usage": None}
    event = log_event("тестовый вопрос", fake_result, latency_seconds=0.42)
    print(json.dumps(event, ensure_ascii=False, indent=2))
