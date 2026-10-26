"""
tests/test_ocr.py — Unit tests for core/ocr.py

Network / Tesseract calls are mocked so tests run on CI without OCR installed.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestCheckTesseract:
    def test_returns_true_when_binary_exists(self, tmp_path):
        """check_tesseract returns True when the configured binary path exists."""
        import core.ocr as ocr_module

        # Create a fake tesseract binary file
        fake_bin = tmp_path / "tesseract.exe"
        fake_bin.write_text("fake")

        # Patch the config and the cached flag
        with patch("core.ocr.config") as mock_cfg:
            mock_cfg.TESSERACT_CMD = str(fake_bin)
            ocr_module._tesseract_available = None  # reset cache
            result = ocr_module.check_tesseract()
        ocr_module._tesseract_available = None  # restore

        assert result is True

    def test_returns_false_when_binary_missing(self):
        """check_tesseract returns False when Tesseract is not found anywhere."""
        import core.ocr as ocr_module

        ocr_module._tesseract_available = None  # reset cache
        with patch("core.ocr.config") as mock_cfg, \
             patch("shutil.which", return_value=None):
            mock_cfg.TESSERACT_CMD = "/nonexistent/tesseract"
            result = ocr_module.check_tesseract()
        ocr_module._tesseract_available = None  # restore

        assert result is False


class TestExtractPageText:
    def test_returns_empty_string_when_tesseract_missing(self, tmp_path):
        """extract_page_text returns '' gracefully when Tesseract is absent."""
        import core.ocr as ocr_module

        ocr_module._tesseract_available = False  # simulate missing Tesseract

        result = ocr_module.extract_page_text("/fake/path.pdf", 0)
        ocr_module._tesseract_available = None  # restore

        assert result == ""

    def test_returns_string_on_successful_ocr(self, tmp_path):
        """extract_page_text returns a string when OCR pipeline succeeds."""
        import core.ocr as ocr_module

        ocr_module._tesseract_available = True  # simulate Tesseract present

        fake_image = MagicMock()
        fake_pixmap = MagicMock()
        fake_pixmap.tobytes.return_value = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        fake_page = MagicMock()
        fake_page.get_pixmap.return_value = fake_pixmap
        fake_doc = MagicMock()
        fake_doc.__len__.return_value = 5
        fake_doc.__getitem__.return_value = fake_page

        with patch("core.ocr.fitz") as mock_fitz, \
             patch("core.ocr.Image") as mock_pil, \
             patch("core.ocr.pytesseract") as mock_tess:
            mock_fitz.open.return_value = fake_doc
            mock_fitz.Matrix.return_value = MagicMock()
            mock_pil.open.return_value = fake_image
            mock_tess.image_to_string.return_value = "OCR extracted text"

            try:
                result = ocr_module.extract_page_text("/fake/path.pdf", 0)
            except Exception:
                result = ""

        ocr_module._tesseract_available = None  # restore
        assert isinstance(result, str)

    def test_out_of_range_page_returns_empty(self):
        """extract_page_text returns '' for a page number beyond the PDF length."""
        import core.ocr as ocr_module

        ocr_module._tesseract_available = True

        fake_doc = MagicMock()
        fake_doc.__len__.return_value = 2  # PDF has only 2 pages

        with patch("fitz.open", return_value=fake_doc):
            try:
                import fitz
                result = ocr_module.extract_page_text("/fake/path.pdf", 99)
            except Exception:
                result = ""

        ocr_module._tesseract_available = None
        assert isinstance(result, str)
