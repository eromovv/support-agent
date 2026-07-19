# Support Agent

RAG-агент технической поддержки. Отвечает на вопросы по загруженной базе знаний
(Markdown/PDF), опираясь на гибридный retrieval (семантика + BM25) и tool-use.
Если релевантных документов нет — агент честно отвечает «не знаю» и может завести
тикет (`search_docs` / `create_ticket` / `escalate_to_human`). Работает через
OpenRouter с автоматическим fallback на резервные модели.

## Стек

- **LLM**: OpenRouter (`instructor` + tool-use, fallback-модели)
- **Векторное хранилище**: Qdrant
- **Эмбеддинги**: `sentence-transformers` (e5)
- **Гибридный поиск**: dense (Qdrant) + `rank_bm25`
- **API**: FastAPI + статический веб-чат
- **Кэш / rate-limit**: Redis
- **Наблюдаемость**: Streamlit-дашборд, JSON-логи
- **Оценка качества**: golden dataset + LLM-judge

## Структура

```
src/
  agent/          # оркестрация агента, retrieval, инструменты
  api/            # FastAPI (/chat, /health, /tickets) + web UI
  ingestion/      # чанкинг, эмбеддинги, запись в Qdrant
  eval/           # прогон golden dataset и LLM-judge
  observability/  # логгер и Streamlit-дашборд
  cli.py          # интерактивный CLI
data/raw_docs/    # исходная база знаний (сюда кладём документы)
```

## Развёртывание

### 1. Инфраструктура

```bash
docker compose up -d      # Qdrant (6333) и Redis (6379)
```

### 2. Окружение Python

```bash
python -m venv .venv
.venv\Scripts\activate            # PowerShell/Windows
# source .venv/bin/activate       # Linux/macOS
pip install -r requirements.txt
```

### 3. Конфигурация

```bash
cp .env.example .env              # затем впишите OPENROUTER_API_KEY
```

Ключевые переменные: `OPENROUTER_API_KEY`, `AGENT_MODEL`, `FALLBACK_MODELS`,
`QDRANT_URL`, `REDIS_URL`, `AGENT_DOMAIN`, `RETRIEVAL_MIN_SCORE` (см. `.env.example`).

### 4. Индексация базы знаний

Положите документы в `data/raw_docs/`, затем:

```bash
cd src
python -m ingestion.embed_and_store    # чанкинг + эмбеддинги + запись в Qdrant
```

## Запуск

Все команды выполняются из каталога `src/`.

```bash
# API + веб-чат  →  http://localhost:8000
uvicorn api.main:app --host 0.0.0.0 --port 8000

# Интерактивный CLI
python cli.py
python cli.py "Как настроить fallback-модели?"

# Дашборд наблюдаемости  →  http://localhost:8501
streamlit run observability/dashboard.py

# Оценка качества
python -m eval.run_eval
python -m eval.run_eval --compare anthropic/claude-3.5-haiku,openai/gpt-4o-mini
```

## API

| Метод | Путь       | Назначение                         |
|-------|------------|------------------------------------|
| POST  | `/chat`    | вопрос агенту, ответ + лог вызовов |
| GET   | `/tickets` | список заведённых тикетов          |
| GET   | `/health`  | health-check                       |
| GET   | `/`        | веб-чат                            |

## Тесты

```bash
pytest
```
