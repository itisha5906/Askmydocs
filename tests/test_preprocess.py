"""
tests/test_preprocess.py — Unit tests for core/preprocess.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.preprocess import clean_text, detect_and_remove_headers_footers


class TestCleanText:
    def test_removes_null_bytes(self):
        assert "\x00" not in clean_text("Hello\x00World")

    def test_removes_form_feeds(self):
        assert "\x0c" not in clean_text("Page break\x0chere")

    def test_fixes_soft_hyphen_linebreak(self):
        result = clean_text("infor-\nmation")
        assert "infor-\nmation" not in result
        assert "information" in result

    def test_collapses_multiple_blank_lines(self):
        text = "A\n\n\n\n\nB"
        result = clean_text(text)
        assert "\n\n\n" not in result

    def test_strips_leading_trailing_whitespace(self):
        result = clean_text("   hello world   ")
        assert result == "hello world"

    def test_empty_string_returns_empty(self):
        assert clean_text("") == ""

    def test_unicode_normalisation_ligature(self):
        # ﬁ (U+FB01) should normalise to 'fi'
        result = clean_text("\ufb01le")
        assert result == "file"


class TestDetectAndRemoveHeadersFooters:
    def _make_pages(self, n: int, header: str = "HEADER", footer: str = "FOOTER") -> list[str]:
        """Create n pages each with a repeated header and footer."""
        pages = []
        for i in range(n):
            pages.append(f"{header}\nPage {i + 1} body content here.\n{footer}")
        return pages

    def test_removes_repeated_header(self):
        pages = self._make_pages(5, header="Company Report 2024")
        cleaned = detect_and_remove_headers_footers(pages)
        for page in cleaned:
            assert "Company Report 2024" not in page

    def test_removes_repeated_footer(self):
        pages = self._make_pages(5, footer="Confidential — Page X")
        cleaned = detect_and_remove_headers_footers(pages)
        for page in cleaned:
            assert "Confidential — Page X" not in page

    def test_preserves_unique_body_content(self):
        pages = self._make_pages(5)
        cleaned = detect_and_remove_headers_footers(pages)
        for i, page in enumerate(cleaned):
            assert f"Page {i + 1} body content here." in page

    def test_fewer_than_3_pages_returned_unchanged(self):
        pages = ["Only one page with content."]
        result = detect_and_remove_headers_footers(pages)
        assert result == pages

    def test_non_repeated_lines_preserved(self):
        """Lines that appear on fewer than threshold pages are kept."""
        pages = [
            "HEADER\nUnique line A\nFOOTER",
            "HEADER\nUnique line B\nFOOTER",
            "HEADER\nUnique line C\nFOOTER",
            "HEADER\nUnique line D\nFOOTER",
            "HEADER\nThis is only here\nFOOTER",
        ]
        cleaned = detect_and_remove_headers_footers(pages)
        # "This is only here" appears only once — should not be removed
        assert "This is only here" in cleaned[4]
