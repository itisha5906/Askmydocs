"""
preprocess.py — Text cleaning and normalisation for AskMyDocs.

Performs Unicode normalisation, whitespace collapse, soft-hyphen repair, and
repeated header/footer removal before chunking.

Public API
----------
    clean_text(text: str) -> str
    detect_and_remove_headers_footers(pages: list[str]) -> list[str]
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

from core.logger import get_logger

log = get_logger(__name__)

# Fraction of pages on which a line must appear to be treated as a header/footer.
_HEADER_FOOTER_THRESHOLD = 0.6
# How many lines from the top / bottom of a page to inspect for repeated content.
_HEADER_LINES = 3
_FOOTER_LINES = 3


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Clean and normalise a single string of extracted text.

    Steps
    -----
    1. Drop null bytes and form-feed characters.
    2. Unicode NFKC normalisation (converts ligatures, fullwidth chars, etc.).
    3. Repair soft-hyphen word-breaks: ``word-\\nnext`` → ``wordnext``.
    4. Collapse runs of blank lines to a single blank line.
    5. Strip leading/trailing whitespace from each line.
    6. Strip leading/trailing whitespace from the whole string.

    Parameters
    ----------
    text:
        Raw text extracted from a PDF page.

    Returns
    -------
    str
        Cleaned text.
    """
    if not text:
        return ""

    # 1. Remove null bytes and form-feeds
    text = text.replace("\x00", "").replace("\x0c", "\n")

    # 2. Unicode normalisation
    text = unicodedata.normalize("NFKC", text)

    # 3. Fix soft-hyphen line-breaks (hyphen at end of line followed by newline)
    #    e.g. "infor-\nmation" → "information"
    text = re.sub(r"-\s*\n\s*", "", text)

    # 4. Collapse multiple consecutive blank lines into one
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 5. Strip leading/trailing whitespace on each line
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(lines)

    # 6. Final strip
    return text.strip()


def detect_and_remove_headers_footers(pages: list[Any]) -> list[Any]:
    """Detect and remove repeated headers and footers across pages.

    Supports both a list of strings (one per page) and a list of LangChain
    Document objects (modifying and returning them with cleaned page_content).

    Parameters
    ----------
    pages:
        List of page text strings or LangChain Document objects.

    Returns
    -------
    list
        Pages or Documents with detected headers/footers removed.
    """
    if len(pages) < 3:  # Not enough pages to detect patterns reliably
        return pages

    is_docs = hasattr(pages[0], "page_content")
    raw_texts = [p.page_content if is_docs else str(p) for p in pages]

    total = len(raw_texts)
    header_candidates: Counter[str] = Counter()
    footer_candidates: Counter[str] = Counter()

    for page_text in raw_texts:
        lines = [l for l in page_text.splitlines() if l.strip()]
        for line in lines[:_HEADER_LINES]:
            if line.strip():
                header_candidates[line.strip()] += 1
        for line in lines[-_FOOTER_LINES:]:
            if line.strip():
                footer_candidates[line.strip()] += 1

    min_occurrences = max(2, int(total * _HEADER_FOOTER_THRESHOLD))

    repeated_headers: set[str] = {
        line for line, count in header_candidates.items()
        if count >= min_occurrences
    }
    repeated_footers: set[str] = {
        line for line, count in footer_candidates.items()
        if count >= min_occurrences
    }
    repeated = repeated_headers | repeated_footers

    if repeated:
        log.info(
            "Detected %d repeated header/footer lines to remove: %s",
            len(repeated),
            list(repeated)[:5],
        )

    cleaned_texts: list[str] = []
    for page_text in raw_texts:
        lines = page_text.splitlines()
        filtered = [l for l in lines if l.strip() not in repeated]
        cleaned_texts.append("\n".join(filtered))

    if is_docs:
        for doc, cleaned_str in zip(pages, cleaned_texts):
            doc.page_content = cleaned_str
        return pages

    return cleaned_texts
