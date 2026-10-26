"""
loader.py — PDF text extraction for AskMyDocs.

Each page is extracted with PyPDF first. If the extractable character count
falls below MIN_PAGE_CHARS (config), the page is flagged as needing OCR and
delegated to ocr.extract_page_text(). PDFPlumber is used as a secondary
extractor when PyPDF returns garbled Unicode (high ratio of non-ASCII chars).

Public API
----------
    load_pdf(file_path: str | Path) -> list[Document]
    LoadError                        — raised for unreadable / encrypted PDFs
"""

from __future__ import annotations

import hashlib
import unicodedata
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.documents import Document

import config
from core.logger import get_logger

if TYPE_CHECKING:
    pass

log = get_logger(__name__)

# Threshold: if >25 % of characters are non-ASCII, try PDFPlumber instead.
_GARBLE_RATIO_THRESHOLD = 0.25


class LoadError(Exception):
    """Raised when a PDF cannot be opened or read."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_garbled(text: str) -> bool:
    """Return True if the text looks garbled (too many non-ASCII characters)."""
    if not text:
        return False
    non_ascii = sum(1 for c in text if ord(c) > 127 and unicodedata.category(c) not in ("Ll", "Lu", "Lt", "Lm", "Lo"))
    return (non_ascii / max(len(text), 1)) > _GARBLE_RATIO_THRESHOLD


def _extract_with_pypdf(pdf_path: Path) -> list[tuple[int, str]]:
    """Extract (page_number, text) pairs using PyPDF.

    Parameters
    ----------
    pdf_path:
        Absolute path to the PDF file.

    Returns
    -------
    list of (0-indexed page number, extracted text) tuples.

    Raises
    ------
    LoadError
        If the PDF is encrypted or cannot be read.
    """
    try:
        import pypdf  # lazy import to keep startup fast
    except ImportError as exc:
        raise ImportError("pypdf is not installed. Run: pip install pypdf") from exc

    try:
        reader = pypdf.PdfReader(str(pdf_path))
    except pypdf.errors.PdfReadError as exc:
        raise LoadError(f"Cannot read PDF '{pdf_path.name}': {exc}") from exc

    if reader.is_encrypted:
        raise LoadError(
            f"'{pdf_path.name}' is password-protected. "
            "Please provide an unlocked PDF."
        )

    pages: list[tuple[int, str]] = []
    for idx, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception as exc:  # noqa: BLE001
            log.warning("PyPDF failed on page %d of '%s': %s", idx, pdf_path.name, exc)
            text = ""
        pages.append((idx, text))

    return pages


def _extract_page_with_pdfplumber(pdf_path: Path, page_number: int) -> str:
    """Extract text from a single page using PDFPlumber.

    Parameters
    ----------
    pdf_path:
        Absolute path to the PDF file.
    page_number:
        0-indexed page number.

    Returns
    -------
    Extracted text string (may be empty if the page has no text layer).
    """
    try:
        import pdfplumber  # lazy import
    except ImportError as exc:
        raise ImportError("pdfplumber is not installed. Run: pip install pdfplumber") from exc

    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            if page_number >= len(pdf.pages):
                return ""
            text = pdf.pages[page_number].extract_text() or ""
        return text
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "PDFPlumber failed on page %d of '%s': %s", page_number, pdf_path.name, exc
        )
        return ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_pdf(file_path: str | Path) -> list[Document]:
    """Load a PDF and return a list of LangChain Documents, one per page.

    For each page:
    - PyPDF is tried first.
    - If the text looks garbled, PDFPlumber is tried instead.
    - If the final char count is below ``config.MIN_PAGE_CHARS``, the page is
      flagged ``ocr=True``; the caller (or this function) then routes it to the
      OCR module.

    Parameters
    ----------
    file_path:
        Path to the PDF file (string or ``pathlib.Path``).

    Returns
    -------
    list[Document]
        Each Document has page_content (str) and metadata dict:
        ``{"source": filename, "page": 1-indexed int, "ocr": bool,
           "char_count": int, "file_hash": sha256_hex}``

    Raises
    ------
    LoadError
        For encrypted, corrupt, or unreadable PDFs.
    FileNotFoundError
        If the path does not exist.
    """
    pdf_path = Path(file_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if pdf_path.suffix.lower() != ".pdf":
        raise LoadError(f"'{pdf_path.name}' is not a PDF file.")

    log.info("Loading PDF: %s", pdf_path.name)

    # Compute SHA-256 hash for deduplication
    file_hash = _compute_file_hash(pdf_path)

    # Primary extraction
    pages_raw = _extract_with_pypdf(pdf_path)

    documents: list[Document] = []
    ocr_page_count = 0

    for page_idx, raw_text in pages_raw:
        page_num = page_idx + 1  # convert to 1-indexed for human readability

        # If garbled, try PDFPlumber
        if _is_garbled(raw_text):
            log.debug("Page %d garbled; trying PDFPlumber.", page_num)
            plumber_text = _extract_page_with_pdfplumber(pdf_path, page_idx)
            if len(plumber_text) > len(raw_text):
                raw_text = plumber_text

        needs_ocr = len(raw_text.strip()) < config.MIN_PAGE_CHARS

        if needs_ocr:
            log.debug("Page %d of '%s' flagged for OCR.", page_num, pdf_path.name)
            try:
                from core.ocr import extract_page_text  # lazy — OCR optional

                raw_text = extract_page_text(str(pdf_path), page_idx)
                ocr_page_count += 1
                is_ocr = True
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "OCR failed for page %d of '%s': %s", page_num, pdf_path.name, exc
                )
                raw_text = ""
                is_ocr = True  # attempted OCR even if it failed
        else:
            is_ocr = False

        documents.append(
            Document(
                page_content=raw_text,
                metadata={
                    "source": pdf_path.name,
                    "page": page_num,
                    "ocr": is_ocr,
                    "char_count": len(raw_text),
                    "file_hash": file_hash,
                },
            )
        )

    log.info(
        "Loaded '%s': %d pages, %d OCR pages.",
        pdf_path.name,
        len(documents),
        ocr_page_count,
    )
    return documents


def _compute_file_hash(path: Path) -> str:
    """Return the SHA-256 hex digest of a file.

    Parameters
    ----------
    path:
        Path to the file.

    Returns
    -------
    str
        64-character lowercase hexadecimal SHA-256 digest.
    """
    sha256 = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()
