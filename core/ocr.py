"""
ocr.py — OCR extraction path for scanned / image-based PDF pages.

Uses PyMuPDF (fitz) to render each page to a high-DPI PIL image, then passes
it to pytesseract. If Tesseract is not installed, the module returns an empty
string and emits a warning instead of crashing — so the rest of the app
continues to work for text-based PDFs.

Public API
----------
    check_tesseract() -> bool
    extract_page_text(pdf_path: str, page_number: int) -> str
"""

from __future__ import annotations

import io
import shutil
from pathlib import Path

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import pytesseract
except ImportError:
    pytesseract = None

import config
from core.logger import get_logger

log = get_logger(__name__)

# Cache the Tesseract availability check so we don't probe on every page.
_tesseract_available: bool | None = None


# ---------------------------------------------------------------------------
# Tesseract availability
# ---------------------------------------------------------------------------

def check_tesseract() -> bool:
    """Return True if Tesseract is installed and callable.

    The check is cached after the first call. On Windows, the configured
    ``config.TESSERACT_CMD`` path is tested; on other platforms, the system
    PATH is searched for ``tesseract``.

    Returns
    -------
    bool
    """
    global _tesseract_available  # noqa: PLW0603

    if _tesseract_available is not None:
        return _tesseract_available

    # Check config-specified binary first
    explicit_cmd = Path(config.TESSERACT_CMD)
    if explicit_cmd.exists():
        _tesseract_available = True
        log.info("Tesseract found at: %s", explicit_cmd)
        return True

    # Fallback: look on PATH
    if shutil.which("tesseract") is not None:
        _tesseract_available = True
        log.info("Tesseract found on PATH.")
        return True

    log.warning(
        "Tesseract not found at '%s' and not on PATH. "
        "OCR will be disabled. Install from https://github.com/UB-Mannheim/tesseract/wiki "
        "(Windows) or via 'sudo apt install tesseract-ocr' (Linux).",
        config.TESSERACT_CMD,
    )
    _tesseract_available = False
    return False


# ---------------------------------------------------------------------------
# Public extraction function
# ---------------------------------------------------------------------------

def extract_page_text(pdf_path: str, page_number: int) -> str:
    """Render a PDF page to an image and extract text with Tesseract OCR.

    Parameters
    ----------
    pdf_path:
        Absolute path to the PDF file.
    page_number:
        0-indexed page number to process.

    Returns
    -------
    str
        Extracted text, or an empty string if OCR is unavailable or fails.
        Callers should check the return value and display a warning to the user
        if it is empty for a page that was expected to have content.
    """
    if not check_tesseract():
        log.warning(
            "Skipping OCR for page %d of '%s' — Tesseract not available.",
            page_number,
            pdf_path,
        )
        return ""

    # -- Render the page to a PIL image with PyMuPDF -------------------------
    if fitz is None:
        raise ImportError("PyMuPDF is not installed. Run: pip install PyMuPDF")

    try:
        doc = fitz.open(pdf_path)
        if page_number >= len(doc):
            log.error(
                "Page %d does not exist in '%s' (total pages: %d).",
                page_number,
                pdf_path,
                len(doc),
            )
            doc.close()
            return ""

        page = doc[page_number]
        # 300 DPI matrix: scale factor = 300/72 ≈ 4.17
        mat = fitz.Matrix(300 / 72, 300 / 72)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        doc.close()
    except Exception as exc:  # noqa: BLE001
        log.error(
            "PyMuPDF failed to render page %d of '%s': %s",
            page_number,
            pdf_path,
            exc,
        )
        return ""

    # -- Convert pixmap to PIL Image -----------------------------------------
    if Image is None:
        raise ImportError("Pillow is not installed. Run: pip install Pillow")

    try:
        img_data = pix.tobytes("png")
        image = Image.open(io.BytesIO(img_data))
    except Exception as exc:  # noqa: BLE001
        log.error("Failed to convert pixmap to PIL Image: %s", exc)
        return ""

    # -- Run Tesseract --------------------------------------------------------
    if pytesseract is None:
        raise ImportError("pytesseract is not installed. Run: pip install pytesseract")

    try:
        # Point pytesseract at the configured binary
        pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_CMD

        text: str = pytesseract.image_to_string(image, lang="eng")
    except Exception as exc:  # noqa: BLE001
        log.error(
            "pytesseract failed on page %d of '%s': %s",
            page_number,
            pdf_path,
            exc,
        )
        return ""

    log.debug(
        "OCR extracted %d characters from page %d of '%s'.",
        len(text),
        page_number,
        Path(pdf_path).name,
    )
    return text
