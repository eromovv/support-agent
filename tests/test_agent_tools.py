import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent.tools import TOOLS_SCHEMA, execute_tool, tool_create_ticket, tool_escalate_to_human

def test_tools_schema_has_required_fields():
    names = {t["function"]["name"] for t in TOOLS_SCHEMA}
    assert names == {"search_docs", "create_ticket", "escalate_to_human"}
    for tool in TOOLS_SCHEMA:
        assert tool["type"] == "function"
        assert "description" in tool["function"]
        assert "parameters" in tool["function"]
        assert tool["function"]["parameters"]["type"] == "object"

def test_create_ticket_writes_to_sqlite(tmp_path, monkeypatch):
    db_path = tmp_path / "test_tickets.db"
    monkeypatch.setattr("agent.tools.TICKETS_DB_PATH", str(db_path))

    result = tool_create_ticket(title="Тест", description="Описание проблемы", priority="high")
    assert result["status"] == "created"
    assert isinstance(result["ticket_id"], int)

    conn = sqlite3.connect(db_path)
    row = conn.execute("SELECT title, priority FROM tickets WHERE id = ?", (result["ticket_id"],)).fetchone()
    conn.close()
    assert row == ("Тест", "high")

def test_escalate_to_human_logs_event(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = tool_escalate_to_human(reason="Пользователь просит человека")
    assert result["status"] == "escalated"

    log_file = tmp_path / "logs" / "escalations.jsonl"
    assert log_file.exists()
    logged = json.loads(log_file.read_text(encoding="utf-8").splitlines()[0])
    assert logged["reason"] == "Пользователь просит человека"

def test_execute_tool_dispatches_unknown_tool_gracefully():
    result = execute_tool("nonexistent_tool", {})
    assert "error" in result

def test_execute_tool_handles_bad_arguments_gracefully(tmp_path, monkeypatch):
    db_path = tmp_path / "test_tickets.db"
    monkeypatch.setattr("agent.tools.TICKETS_DB_PATH", str(db_path))

    result = execute_tool("create_ticket", {"title": "Только заголовок"})
    assert "error" in result
