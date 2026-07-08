from __future__ import annotations

import json
import os
from pathlib import Path

from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "support_docs")
PROCESSED_PATH = "data/processed/chunks.json"
RETRIEVAL_MIN_SCORE = float(os.environ.get("RETRIEVAL_MIN_SCORE", "0.15"))

_embedding_model = None
_bm25_index = None
_bm25_records: list[dict] = []

def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        from ingestion.embed_and_store import DEFAULT_MODEL_NAME

        _embedding_model = SentenceTransformer(DEFAULT_MODEL_NAME)
    return _embedding_model

def _tokenize(text: str) -> list[str]:
    return text.lower().split()

def _get_bm25_index():
    global _bm25_index, _bm25_records
    if _bm25_index is None:
        path = Path(PROCESSED_PATH)
        if not path.exists():
            raise RuntimeError(
                f"{PROCESSED_PATH} не найден. Сначала запустите "
                "`python src/ingestion/embed_and_store.py`, чтобы построить корпус."
            )
        _bm25_records = json.loads(path.read_text(encoding="utf-8"))
        tokenized = [_tokenize(r["text"]) for r in _bm25_records]
        _bm25_index = BM25Okapi(tokenized)
    return _bm25_index

QUERY_PREFIX = "query: "

def semantic_search(query: str, top_k: int = 5) -> list[dict]:
    model = _get_embedding_model()
    query_vector = model.encode(QUERY_PREFIX + query, normalize_embeddings=True).tolist()

    client = QdrantClient(url=QDRANT_URL)
    hits = client.query_points(
        collection_name=QDRANT_COLLECTION, query=query_vector, limit=top_k
    ).points
    return [
        {
            "text": h.payload["text"],
            "source": h.payload["source"],
            "chunk_id": h.payload["chunk_id"],
            "score": h.score,
        }
        for h in hits
    ]

def bm25_search(query: str, top_k: int = 5) -> list[dict]:
    index = _get_bm25_index()
    scores = index.get_scores(_tokenize(query))
    ranked = sorted(zip(_bm25_records, scores), key=lambda x: x[1], reverse=True)[:top_k]
    return [
        {"text": r["text"], "source": r["source"], "chunk_id": r["chunk_id"], "score": float(s)}
        for r, s in ranked
        if s > 0
    ]

def _normalize(results: list[dict]) -> list[dict]:
    if not results:
        return results
    max_score = max(r["score"] for r in results) or 1.0
    for r in results:
        r["norm_score"] = r["score"] / max_score
    return results

def hybrid_search(query: str, top_k: int = 5, semantic_weight: float = 0.6) -> list[dict]:
    candidate_pool = max(top_k * 5, 20)
    raw_semantic = semantic_search(query, top_k=candidate_pool)

    top_semantic_score = max((r["score"] for r in raw_semantic), default=0.0)
    if top_semantic_score < RETRIEVAL_MIN_SCORE:
        return []

    semantic_results = _normalize(raw_semantic)
    bm25_results = _normalize(bm25_search(query, top_k=candidate_pool))

    combined: dict[str, dict] = {}
    for r in semantic_results:
        combined[r["chunk_id"]] = {**r, "hybrid_score": semantic_weight * r["norm_score"]}
    for r in bm25_results:
        if r["chunk_id"] in combined:
            combined[r["chunk_id"]]["hybrid_score"] += (1 - semantic_weight) * r["norm_score"]
        else:
            combined[r["chunk_id"]] = {**r, "hybrid_score": (1 - semantic_weight) * r["norm_score"]}

    ranked = sorted(combined.values(), key=lambda r: r["hybrid_score"], reverse=True)
    return ranked[:top_k]

if __name__ == "__main__":
    import sys

    query = sys.argv[1] if len(sys.argv) > 1 else "Что такое prompt injection?"
    for r in hybrid_search(query, top_k=3):
        print(f"[{r['hybrid_score']:.3f}] {r['source']} / {r['chunk_id']}")
        print(r["text"][:200])
        print()
