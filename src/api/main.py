from __future__ import annotations

import sqlite3
import time
from collections import defaultdict

from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent.agent import run_agent
from agent.tools import TICKETS_DB_PATH
from observability.logger import log_event

app = FastAPI(title="Support Agent API")

RATE_LIMIT_REQUESTS = 20
RATE_LIMIT_WINDOW_SECONDS = 60
_request_log: dict[str, list[float]] = defaultdict(list)

def _check_rate_limit(client_ip: str):
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW_SECONDS
    _request_log[client_ip] = [t for t in _request_log[client_ip] if t > window_start]
    if len(_request_log[client_ip]) >= RATE_LIMIT_REQUESTS:
        raise HTTPException(status_code=429, detail="Слишком много запросов, попробуйте позже.")
    _request_log[client_ip].append(now)

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    answer: str
    rounds_used: int
    model: str
    latency_seconds: float
    latency_seconds: float

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, request: Request):
    _check_rate_limit(request.client.host if request.client else "unknown")

    start = time.perf_counter()
    result = run_agent(req.message)
    latency = time.perf_counter() - start

    log_event(question=req.message, agent_result=result, latency_seconds=latency)

    return ChatResponse(
        answer=result["answer"],
        rounds_used=result["rounds_used"],
        model=result["model"],
        latency_seconds=latency,
    )

@app.get("/tickets")
def list_tickets():
    try:
        conn = sqlite3.connect(TICKETS_DB_PATH)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM tickets ORDER BY id DESC").fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except sqlite3.OperationalError:
        return []

_STATIC_DIR = Path(__file__).parent / "static"
if _STATIC_DIR.is_dir():
    app.mount("/", StaticFiles(directory=_STATIC_DIR, html=True), name="static")
