from __future__ import annotations

import json
import uuid
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from sentence_transformers import SentenceTransformer

from ingestion.chunker import Chunk, chunk_directory

DEFAULT_MODEL_NAME = "intfloat/multilingual-e5-base"
PASSAGE_PREFIX = "passage: "

def get_embedding_model(model_name: str = DEFAULT_MODEL_NAME) -> SentenceTransformer:
    return SentenceTransformer(model_name)

def embed_and_store(
    raw_docs_dir: str = "data/raw_docs",
    processed_path: str = "data/processed/chunks.json",
    qdrant_url: str = "http://localhost:6333",
    collection_name: str = "support_docs",
    strategy: str = "paragraph",
    embedding_model_name: str = DEFAULT_MODEL_NAME,
    recreate_collection: bool = True,
) -> int:
    chunks: list[Chunk] = chunk_directory(raw_docs_dir, strategy=strategy)
    if not chunks:
        raise RuntimeError(f"В {raw_docs_dir} не найдено ни одного .md файла")

    model = get_embedding_model(embedding_model_name)
    vectors = model.encode(
        [PASSAGE_PREFIX + c.text for c in chunks], show_progress_bar=True, normalize_embeddings=True
    )
    vector_size = vectors.shape[1]

    client = QdrantClient(url=qdrant_url)
    if recreate_collection:
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=qmodels.VectorParams(size=vector_size, distance=qmodels.Distance.COSINE),
        )

    points = []
    processed_records = []
    for chunk, vector in zip(chunks, vectors):
        point_id = str(uuid.uuid4())
        points.append(
            qmodels.PointStruct(
                id=point_id,
                vector=vector.tolist(),
                payload={"text": chunk.text, "source": chunk.source, "chunk_id": chunk.chunk_id, "heading": chunk.heading},
            )
        )
        processed_records.append(
            {"id": point_id, "text": chunk.text, "source": chunk.source, "chunk_id": chunk.chunk_id}
        )

    client.upsert(collection_name=collection_name, points=points)

    Path(processed_path).parent.mkdir(parents=True, exist_ok=True)
    Path(processed_path).write_text(json.dumps(processed_records, ensure_ascii=False, indent=2), encoding="utf-8")

    return len(chunks)

if __name__ == "__main__":
    n = embed_and_store()
    print(f"Записано {n} чанков в Qdrant и в data/processed/chunks.json")
