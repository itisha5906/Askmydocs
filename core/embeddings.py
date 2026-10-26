"""
embeddings.py — Hugging Face Embeddings wrapper for AskMyDocs.

Loads sentence-transformers/all-mpnet-base-v2 locally (no API key required).
Embeddings are normalized (unit length) so cosine similarity equals inner product.
Provides a cached singleton instance to prevent re-loading weights into memory.

Public API
----------
    get_embeddings() -> HuggingFaceEmbeddings
    reset_embeddings_cache() -> None
"""

from __future__ import annotations

import config
from core.logger import get_logger

log = get_logger(__name__)

_cached_embeddings = None


def get_embeddings(model_name: str | None = None):
    """Return a cached instance of HuggingFaceEmbeddings.

    Parameters
    ----------
    model_name:
        Model identifier from Hugging Face Hub. Defaults to config.EMBED_MODEL.

    Returns
    -------
    HuggingFaceEmbeddings
    """
    global _cached_embeddings  # noqa: PLW0603

    target_model = model_name or config.EMBED_MODEL

    if _cached_embeddings is not None and model_name is None:
        return _cached_embeddings

    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError as exc:
        raise ImportError(
            "langchain-huggingface is not installed. Run: pip install langchain-huggingface"
        ) from exc

    log.info("Loading embedding model '%s' (normalize_embeddings=True)...", target_model)

    embeddings = HuggingFaceEmbeddings(
        model_name=target_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    if model_name is None:
        _cached_embeddings = embeddings

    return embeddings


def reset_embeddings_cache() -> None:
    """Clear cached embeddings instance (useful for testing)."""
    global _cached_embeddings  # noqa: PLW0603
    _cached_embeddings = None
