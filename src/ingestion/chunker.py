from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class Chunk:
    text: str
    source: str
    chunk_id: str
    heading: str | None = None
    metadata: dict = field(default_factory=dict)

def _split_into_paragraphs(text: str) -> list[tuple[str | None, str]]:
    lines = text.split("\n")
    paragraphs: list[tuple[str | None, str]] = []
    current_heading: str | None = None
    buffer: list[str] = []

    def flush():
        if buffer:
            joined = "\n".join(buffer).strip()
            if joined:
                paragraphs.append((current_heading, joined))
            buffer.clear()

    for line in lines:
        heading_match = re.match(r"^#{1,6}\s+(.*)", line)
        if heading_match:
            flush()
            current_heading = heading_match.group(1).strip()
            continue
        if line.strip() == "":
            flush()
        else:
            buffer.append(line)
    flush()
    return paragraphs

def chunk_paragraph(text: str, source: str, target_words: int = 350, overlap_ratio: float = 0.15) -> list[Chunk]:
    paragraphs = _split_into_paragraphs(text)
    chunks: list[Chunk] = []
    buffer_words: list[str] = []
    buffer_heading: str | None = None
    idx = 0

    def make_chunk():
        nonlocal idx
        if not buffer_words:
            return
        chunk_text = " ".join(buffer_words).strip()
        chunks.append(
            Chunk(
                text=chunk_text,
                source=source,
                chunk_id=f"{Path(source).stem}-p{idx}",
                heading=buffer_heading,
            )
        )
        idx += 1

    for heading, para in paragraphs:
        words = para.split()
        if buffer_heading is None:
            buffer_heading = heading
        if len(buffer_words) + len(words) > target_words and buffer_words:
            make_chunk()
            overlap_n = int(len(buffer_words) * overlap_ratio)
            buffer_words = buffer_words[-overlap_n:] if overlap_n else []
            buffer_heading = heading
        buffer_words.extend(words)
    make_chunk()
    return chunks

def chunk_fixed(text: str, source: str, chunk_size_words: int = 350, overlap_words: int = 50) -> list[Chunk]:

    clean_text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    words = clean_text.split()
    chunks: list[Chunk] = []
    step = max(chunk_size_words - overlap_words, 1)
    idx = 0
    for start in range(0, len(words), step):
        window = words[start : start + chunk_size_words]
        if not window:
            break
        chunks.append(
            Chunk(
                text=" ".join(window),
                source=source,
                chunk_id=f"{Path(source).stem}-f{idx}",
            )
        )
        idx += 1
        if start + chunk_size_words >= len(words):
            break
    return chunks

def chunk_text(text: str, source: str, strategy: str = "paragraph", **kwargs) -> list[Chunk]:
    if strategy == "paragraph":
        return chunk_paragraph(text, source, **kwargs)
    if strategy == "fixed":
        return chunk_fixed(text, source, **kwargs)
    raise ValueError(f"Неизвестная стратегия чанкинга: {strategy}")

def chunk_directory(directory: str | Path, strategy: str = "paragraph", **kwargs) -> list[Chunk]:
    directory = Path(directory)
    all_chunks: list[Chunk] = []
    for path in sorted(directory.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        all_chunks.extend(chunk_text(text, source=path.name, strategy=strategy, **kwargs))
    return all_chunks

if __name__ == "__main__":
    import sys

    docs_dir = sys.argv[1] if len(sys.argv) > 1 else "data/raw_docs"
    chunks = chunk_directory(docs_dir)
    print(f"Всего чанков: {len(chunks)}")
    for c in chunks[:3]:
        print("---")
        print(f"[{c.source} / {c.chunk_id}] {c.text[:150]}...")
