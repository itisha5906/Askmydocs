"""
tests/test_chunker.py — Unit tests for core/chunker.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.documents import Document
from core.chunker import chunk_documents


def _make_doc(text: str, page: int = 1, source: str = "test.pdf", ocr: bool = False) -> Document:
    return Document(
        page_content=text,
        metadata={"source": source, "page": page, "ocr": ocr, "char_count": len(text), "file_hash": "abc123"},
    )


class TestChunkDocuments:
    def test_returns_list_of_documents(self):
        doc = _make_doc("A" * 900)
        chunks = chunk_documents([doc])
        assert isinstance(chunks, list)
        assert all(isinstance(c, Document) for c in chunks)

    def test_chunks_exceed_one_for_long_text(self):
        """A 2000-char document with chunk_size=800 must produce more than 1 chunk."""
        doc = _make_doc("Word " * 400)  # ~2000 chars
        chunks = chunk_documents([doc], chunk_size=800, chunk_overlap=150)
        assert len(chunks) > 1

    def test_required_metadata_keys_present(self):
        doc = _make_doc("Short text for metadata test " * 10)
        chunks = chunk_documents([doc])
        required = {"source", "page", "chunk_id", "chunk_index", "total_chars", "is_ocr"}
        for chunk in chunks:
            missing = required - chunk.metadata.keys()
            assert not missing, f"Missing metadata keys: {missing}"

    def test_chunk_id_is_unique(self):
        doc = _make_doc("X " * 600)
        chunks = chunk_documents([doc], chunk_size=200, chunk_overlap=50)
        ids = [c.metadata["chunk_id"] for c in chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs must be unique"

    def test_is_ocr_flag_forwarded(self):
        doc = _make_doc("OCR page content " * 20, ocr=True)
        chunks = chunk_documents([doc])
        assert all(c.metadata["is_ocr"] is True for c in chunks)

    def test_empty_document_list_returns_empty(self):
        assert chunk_documents([]) == []

    def test_empty_page_content_skipped(self):
        docs = [_make_doc(""), _make_doc("Valid content " * 20)]
        chunks = chunk_documents(docs)
        # The empty doc should produce no chunks
        assert all(len(c.page_content.strip()) >= 20 for c in chunks)

    def test_source_filename_preserved(self):
        doc = _make_doc("Content " * 50, source="myfile.pdf")
        chunks = chunk_documents([doc])
        assert all(c.metadata["source"] == "myfile.pdf" for c in chunks)

    def test_chunk_size_respected(self):
        """No chunk should exceed chunk_size + overlap significantly."""
        doc = _make_doc("Hello world! " * 200)
        chunks = chunk_documents([doc], chunk_size=300, chunk_overlap=50)
        # Allow a small tolerance (splitter may slightly exceed on boundary words)
        for chunk in chunks:
            assert len(chunk.page_content) <= 400, (
                f"Chunk too large: {len(chunk.page_content)} chars"
            )
