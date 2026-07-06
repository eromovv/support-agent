from __future__ import annotations

import json

from common.llm_client import chat_completion, JUDGE_MODEL, AGENT_MODEL
from models.schemas import JudgeVerdict

JUDGE_SYSTEM_PROMPT = """\
Ты — независимый судья качества ответов AI-агента поддержки. Тебе даётся вопрос \
пользователя, эталонный (ожидаемый) ответ и реальный ответ агента. Оцени реальный \
ответ по шкале от 1 до 5:
5 — полностью соответствует по смыслу эталонному ответу, без лишних выдумок
4 — в целом верно, но есть небольшие упущения
3 — частично верно, есть заметные пробелы или неточности
2 — в основном неверно или не по теме
1 — полностью неверно или явная галлюцинация

Также определи, содержит ли ответ агента факты, которых нет в эталонном ответе \
и которые выглядят выдуманными (is_hallucination).

Отвечай СТРОГО в формате JSON без каких-либо пояснений вне JSON:
{"score": <int 1-5>, "rationale": "<короткое обоснование>", "is_hallucination": <true/false>}
"""

def judge_answer(question: str, expected_answer: str, agent_answer: str, judge_model: str | None = None) -> JudgeVerdict:
    judge_model = judge_model or JUDGE_MODEL
    if judge_model == AGENT_MODEL:
        raise ValueError(
            "JUDGE_MODEL совпадает с AGENT_MODEL — судья не должен оценивать сам себя. "
            "Смените JUDGE_MODEL в .env."
        )

    user_content = (
        f"Вопрос пользователя: {question}\n\n"
        f"Эталонный ответ: {expected_answer}\n\n"
        f"Ответ агента: {agent_answer}"
    )
    response = chat_completion(
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        model=judge_model,
        temperature=0.0,
        max_tokens=300,
        use_fallback=False,
    )
    raw = response.choices[0].message.content.strip()

    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    data = json.loads(raw)
    return JudgeVerdict(**data)

if __name__ == "__main__":
    verdict = judge_answer(
        question="Что такое faithfulness?",
        expected_answer="Faithfulness — метрика того, насколько ответ подтверждён найденными документами.",
        agent_answer="Faithfulness показывает, насколько ответ основан на реальных документах, а не выдуман.",
    )
    print(verdict.model_dump_json(indent=2))
