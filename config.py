"""
config.py — Central configuration for AskMyDocs.

Loads settings from .env (python-dotenv) at import time, then falls back
to st.secrets when running inside Streamlit Cloud. All other modules import
from here instead of reading env vars directly.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load .env from the project root (parent of this file's directory)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).parent
load_dotenv(_PROJECT_ROOT / ".env", override=False)


def _secret(key: str, default: str = "") -> str:
    """Return value from env, falling back to st.secrets if available."""
    value = os.getenv(key, "")
    if not value:
        try:
            import streamlit as st  # noqa: PLC0415 — lazy import

            value = st.secrets.get(key, default)
        except Exception:  # noqa: BLE001
            value = default
    return value or default


# ---------------------------------------------------------------------------
# API Keys
# ---------------------------------------------------------------------------
GROQ_API_KEY: str = _secret("GROQ_API_KEY")
OPENAI_API_KEY: str = _secret("OPENAI_API_KEY")
OPENAI_BASE_URL: str = _secret("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL: str = _secret("OPENAI_MODEL", "gpt-4o-mini")

# ---------------------------------------------------------------------------
# Embedding model
# ---------------------------------------------------------------------------
EMBED_MODEL: str = "sentence-transformers/all-mpnet-base-v2"

# ---------------------------------------------------------------------------
# Chunking defaults (can be overridden from the Streamlit sidebar)
# ---------------------------------------------------------------------------
CHUNK_SIZE: int = 800
CHUNK_OVERLAP: int = 150
MIN_PAGE_CHARS: int = 50          # pages with fewer chars are sent to OCR

# ---------------------------------------------------------------------------
# Retrieval defaults
# ---------------------------------------------------------------------------
TOP_K: int = 6
MMR_FETCH_K_MULTIPLIER: int = 3   # fetch_k = TOP_K * this
MMR_LAMBDA: float = 0.5

# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------
FAISS_INDEX_PATH: Path = _PROJECT_ROOT / "data" / "faiss_index"
FAISS_INDEX_META: Path = FAISS_INDEX_PATH / "indexed_files.json"

# ---------------------------------------------------------------------------
# LLM fallback model list (used when Groq API call fails)
# ---------------------------------------------------------------------------
GROQ_FALLBACK_MODELS: list[str] = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
]

# ---------------------------------------------------------------------------
# Upload limits
# ---------------------------------------------------------------------------
MAX_UPLOAD_MB: int = 200

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_FILE: Path = _PROJECT_ROOT / "app.log"
LOG_MAX_BYTES: int = 5 * 1024 * 1024   # 5 MB
LOG_BACKUP_COUNT: int = 3

# ---------------------------------------------------------------------------
# Tesseract (OCR)
# ---------------------------------------------------------------------------
import platform as _platform  # noqa: E402

if _platform.system() == "Windows":
    TESSERACT_CMD: str = os.getenv(
        "TESSERACT_CMD",
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    )
else:
    TESSERACT_CMD = os.getenv("TESSERACT_CMD", "tesseract")

# ---------------------------------------------------------------------------
# Quick sanity-check when run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== AskMyDocs Configuration ===")
    print(f"GROQ_API_KEY    : {'set ✓' if GROQ_API_KEY else 'NOT SET ✗'}")
    print(f"OPENAI_API_KEY  : {'set ✓' if OPENAI_API_KEY else 'NOT SET ✗'}")
    print(f"OPENAI_BASE_URL : {OPENAI_BASE_URL}")
    print(f"OPENAI_MODEL    : {OPENAI_MODEL}")
    print(f"EMBED_MODEL     : {EMBED_MODEL}")
    print(f"FAISS_INDEX_PATH: {FAISS_INDEX_PATH}")
    print(f"LOG_FILE        : {LOG_FILE}")
    print(f"TESSERACT_CMD   : {TESSERACT_CMD}")
    print(f"CHUNK_SIZE      : {CHUNK_SIZE}")
    print(f"CHUNK_OVERLAP   : {CHUNK_OVERLAP}")
    print(f"TOP_K           : {TOP_K}")
