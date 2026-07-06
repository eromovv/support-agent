from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

TICKETS_DB_PATH = "tickets.db"

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": (
                "Ищет релевантные фрагменты в базе знаний по вопросу пользователя. "
                "Используй этот инструмент, прежде чем отвечать на вопрос по теме базы знаний."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Поисковый запрос"},
                    "top_k": {
                        "type": "integer",
                        "description": "Сколько фрагментов вернуть",
                        "default": 3,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_ticket",
            "description": (
                "Создаёт тикет поддержки, если вопрос пользователя не может быть решён "
                "через базу знаний и требует внимания человека."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "priority": {
                        "type": "string",
                        "enum": ["low", "normal", "high", "urgent"],
                        "default": "normal",
                    },
                },
                "required": ["title", "description"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "escalate_to_human",
            "description": (
                "Эскалирует диалог на живого оператора, когда вопрос выходит за рамки "
                "компетенции агента или пользователь явно просит человека."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string", "description": "Причина эскалации"},
                },
                "required": ["reason"],
            },
        },
    },
]

def _init_tickets_db():
    conn = sqlite3.connect(TICKETS_DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS tickets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            description TEXT NOT NULL,
            priority TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    return conn

def tool_search_docs(query: str, top_k: int = 3) -> dict:
    from agent.retrieval import hybrid_search

    results = hybrid_search(query, top_k=top_k)
    return {
        "results": [
            {"source": r["source"], "chunk_id": r["chunk_id"], "text": r["text"]} for r in results
        ]
    }

def tool_create_ticket(title: str, description: str, priority: str = "normal") -> dict:
    conn = _init_tickets_db()
    now = datetime.now(timezone.utc).isoformat()
    cursor = conn.execute(
        "INSERT INTO tickets (title, description, priority, created_at) VALUES (?, ?, ?, ?)",
        (title, description, priority, now),
    )
    conn.commit()
    ticket_id = cursor.lastrowid
    conn.close()
    return {"ticket_id": ticket_id, "status": "created"}

def tool_escalate_to_human(reason: str) -> dict:

    Path("logs").mkdir(exist_ok=True)
    with open("logs/escalations.jsonl", "a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {"reason": reason, "timestamp": datetime.now(timezone.utc).isoformat()},
                ensure_ascii=False,
            )
            + "\n"
        )
    return {"status": "escalated", "reason": reason}

TOOL_DISPATCH = {
    "search_docs": tool_search_docs,
    "create_ticket": tool_create_ticket,
    "escalate_to_human": tool_escalate_to_human,
}

def execute_tool(name: str, arguments: dict) -> dict:
    if name not in TOOL_DISPATCH:
        return {"error": f"Неизвестный инструмент: {name}"}
    try:
        return TOOL_DISPATCH[name](**arguments)
    except Exception as exc:
        return {"error": f"Ошибка выполнения инструмента {name}: {exc}"}
