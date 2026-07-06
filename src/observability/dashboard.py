from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

LOG_PATH = Path("logs/events.jsonl")

st.set_page_config(page_title="Support Agent Observability", layout="wide")
st.title("📊 Support Agent — Observability")

if not LOG_PATH.exists():
    st.warning(
        "Лог-файл logs/events.jsonl не найден. Задайте агенту пару вопросов через "
        "`python src/agent/agent.py \"...\"` или через API, чтобы появились данные."
    )
    st.stop()

records = [json.loads(line) for line in LOG_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
df = pd.DataFrame(records)
df["timestamp"] = pd.to_datetime(df["timestamp"])

col1, col2, col3, col4 = st.columns(4)
col1.metric("Всего запросов", len(df))
col2.metric("Средняя латентность, с", f"{df['latency_seconds'].mean():.2f}")
col3.metric("Доля ошибок", f"{(df['error_type'] != 'none').mean() * 100:.1f}%")
col4.metric("Модели в использовании", df["model"].nunique())

st.subheader("Латентность по времени")
st.line_chart(df.set_index("timestamp")["latency_seconds"])

st.subheader("Распределение типов ошибок")
error_counts = df["error_type"].value_counts()
st.bar_chart(error_counts)

st.subheader("Последние запросы")
st.dataframe(
    df[["timestamp", "question", "answer", "model", "error_type", "latency_seconds", "rounds_used"]]
    .sort_values("timestamp", ascending=False)
    .head(50),
    use_container_width=True,
)

with st.expander("Полный raw-лог (для дебага trace)"):
    st.json(records[-5:])
