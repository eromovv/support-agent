from __future__ import annotations

import os
from typing import Any

import openai
from openai import OpenAI
from dotenv import load_dotenv
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

load_dotenv()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

def _get_client() -> OpenAI:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key or api_key.startswith("sk-or-v1-your-key"):
        raise RuntimeError(
            "OPENROUTER_API_KEY не настроен. Скопируйте .env.example в .env "
            "и укажите свой ключ с https://openrouter.ai/keys"
        )
    return OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
        default_headers={

            "HTTP-Referer": os.environ.get("APP_URL", "http://localhost"),
            "X-Title": "support-agent",
        },
    )

_client: OpenAI | None = None

def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = _get_client()
    return _client

AGENT_MODEL = os.environ.get("AGENT_MODEL", "anthropic/claude-3.5-haiku")
FALLBACK_MODELS = [
    m.strip() for m in os.environ.get("FALLBACK_MODELS", "").split(",") if m.strip()
]
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "openai/gpt-4o-mini")

if JUDGE_MODEL == AGENT_MODEL:

    import warnings

    warnings.warn(
        "JUDGE_MODEL совпадает с AGENT_MODEL. Судья должен быть другой моделью, "
        "иначе оценка eval будет необъективно завышенной.",
        stacklevel=2,
    )

@retry(
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(4),
    retry=retry_if_exception_type(
        (openai.APITimeoutError, openai.APIConnectionError, openai.RateLimitError)
    ),
)
def chat_completion(
    messages: list[dict[str, Any]],
    model: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: Any = None,
    temperature: float = 0.2,
    max_tokens: int = 1024,
    use_fallback: bool = True,
):
    model = model or AGENT_MODEL
    extra_body: dict[str, Any] = {}

    if use_fallback and FALLBACK_MODELS and model == AGENT_MODEL:

        extra_body["models"] = [AGENT_MODEL, *FALLBACK_MODELS]

    kwargs: dict[str, Any] = dict(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if extra_body:
        kwargs["extra_body"] = extra_body
    if tools:
        kwargs["tools"] = tools
    if tool_choice:
        kwargs["tool_choice"] = tool_choice

    return get_client().chat.completions.create(**kwargs)

def estimate_cost_usd(prompt_tokens: int, completion_tokens: int, price_per_mtok: dict) -> float:
    return (
        prompt_tokens * price_per_mtok.get("input", 0)
        + completion_tokens * price_per_mtok.get("output", 0)
    ) / 1_000_000
