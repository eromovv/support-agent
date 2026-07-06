from __future__ import annotations

import sys
import time

from agent.agent import run_agent
from observability.logger import log_event

def ask(question: str):
    start = time.perf_counter()
    result = run_agent(question)
    latency = time.perf_counter() - start
    log_event(question=question, agent_result=result, latency_seconds=latency)

    print(f"\n🤖 {result['answer']}\n")
    if result["tool_calls_log"]:
        print("— вызванные инструменты —")
        for tc in result["tool_calls_log"]:
            print(f"  • {tc['name']}({tc['arguments']})")
    print(f"[модель: {result['model']}, раундов: {result['rounds_used']}, латентность: {latency:.2f}с]\n")

def main():
    if len(sys.argv) > 1:
        ask(" ".join(sys.argv[1:]))
        return

    print("Support Agent CLI. Введите вопрос (Ctrl+C для выхода).\n")
    while True:
        try:
            question = input("Вы: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nЗавершение.")
            break
        if not question:
            continue
        ask(question)

if __name__ == "__main__":
    main()
