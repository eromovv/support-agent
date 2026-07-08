import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ingestion.chunker import (
    _clean_pdf_text,
    chunk_fixed,
    chunk_paragraph,
    chunk_text,
    read_document,
)

SAMPLE_MD = """# Заголовок один

Первый абзац с несколькими словами для теста чанкинга разбиения текста тут.

Второй абзац продолжает тему и добавляет ещё немного слов для теста.

## Заголовок два

Третий абзац относится уже к другому разделу документа с другим заголовком.
"""

def test_chunk_paragraph_respects_target_size():
    chunks = chunk_paragraph(SAMPLE_MD, source="test.md", target_words=15, overlap_ratio=0.2)
    assert len(chunks) >= 2
    for c in chunks:
        assert c.text.strip() != ""
        assert c.source == "test.md"

def test_chunk_paragraph_overlap_between_consecutive_chunks():
    chunks = chunk_paragraph(SAMPLE_MD, source="test.md", target_words=10, overlap_ratio=0.3)
    if len(chunks) >= 2:
        first_words = set(chunks[0].text.split())
        second_words = set(chunks[1].text.split())

        assert first_words & second_words or len(chunks[0].text.split()) < 10

def test_chunk_fixed_word_count_and_overlap():
    chunks = chunk_fixed(SAMPLE_MD, source="test.md", chunk_size_words=10, overlap_words=3)
    assert len(chunks) >= 1
    for c in chunks:
        word_count = len(c.text.split())
        assert word_count <= 10

def test_chunk_text_dispatch_raises_on_unknown_strategy():
    try:
        chunk_text(SAMPLE_MD, source="test.md", strategy="does_not_exist")
        assert False, "должна была быть ValueError"
    except ValueError:
        pass

def test_chunk_ids_unique():
    chunks = chunk_paragraph(SAMPLE_MD, source="test.md", target_words=15)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids)), "chunk_id должны быть уникальны"

def test_clean_pdf_text_dehyphenates_and_joins_lines():
    raw = "Возврат средств осуществля-\nется в течение\n14 дней.\n\nВторой абзац политики."
    cleaned = _clean_pdf_text(raw)
    assert "осуществляется" in cleaned
    assert "в течение 14 дней." in cleaned
    assert "\n\n" in cleaned

def test_read_document_rejects_unsupported_format():
    try:
        read_document("some_file.txt")
        assert False, "должна была быть ValueError"
    except ValueError:
        pass
