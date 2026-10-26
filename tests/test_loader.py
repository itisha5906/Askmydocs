"""
tests/test_loader.py — Unit tests for core/loader.py

Uses reportlab to generate minimal in-memory PDFs so no external file is needed.
"""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import pytest

# Make the project root importable
sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_text_pdf(text_per_page: list[str], path: Path) -> None:
    """Create a minimal PDF with one text layer per page using reportlab."""
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(str(path), pagesize=A4)
    width, height = A4
    for text in text_per_page:
        c.setFont("Helvetica", 12)
        # Wrap text into lines
        y = height - 50
        for line in text.split("\n"):
            c.drawString(50, y, line[:100])
            y -= 20
        c.showPage()
    c.save()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLoadPDF:
    def test_loads_text_pdf_returns_documents(self, tmp_path):
        """load_pdf returns one Document per page for a simple text PDF."""
        from core.loader import load_pdf

        pdf = tmp_path / "simple.pdf"
        _make_text_pdf(["Hello World\nThis is page one.", "Page two content here."], pdf)

        docs = load_pdf(str(pdf))
        assert len(docs) == 2

    def test_document_metadata_keys_present(self, tmp_path):
        """Every returned Document carries the required metadata keys."""
        from core.loader import load_pdf

        pdf = tmp_path / "meta.pdf"
        _make_text_pdf(["Metadata test page."], pdf)

        docs = load_pdf(str(pdf))
        required_keys = {"source", "page", "ocr", "char_count", "file_hash"}
        for doc in docs:
            assert required_keys.issubset(doc.metadata.keys()), (
                f"Missing keys: {required_keys - doc.metadata.keys()}"
            )

    def test_page_numbering_starts_at_one(self, tmp_path):
        """Page numbers in metadata are 1-indexed."""
        from core.loader import load_pdf

        pdf = tmp_path / "pages.pdf"
        _make_text_pdf(["First page.", "Second page."], pdf)

        docs = load_pdf(str(pdf))
        pages = [d.metadata["page"] for d in docs]
        assert pages == [1, 2]

    def test_file_not_found_raises(self):
        """load_pdf raises FileNotFoundError for a non-existent path."""
        from core.loader import load_pdf

        with pytest.raises(FileNotFoundError):
            load_pdf("/non/existent/file.pdf")

    def test_non_pdf_raises_load_error(self, tmp_path):
        """load_pdf raises LoadError for a non-PDF extension."""
        from core.loader import load_pdf, LoadError

        txt_file = tmp_path / "doc.txt"
        txt_file.write_text("not a pdf")

        with pytest.raises(LoadError):
            load_pdf(str(txt_file))

    def test_file_hash_is_sha256_hex(self, tmp_path):
        """file_hash metadata is a 64-char hex string (SHA-256)."""
        from core.loader import load_pdf

        pdf = tmp_path / "hash.pdf"
        _make_text_pdf(["Hash test."], pdf)

        docs = load_pdf(str(pdf))
        fhash = docs[0].metadata["file_hash"]
        assert len(fhash) == 64
        assert all(c in "0123456789abcdef" for c in fhash)
