"""
vectorstore.py — Vector Store persistence and management using FAISS.

Manages FAISS vector store creation, incremental merging, serialization to disk,
and file-hash deduplication so unchanged documents are not re-embedded.

Public API
----------
    load_vectorstore(embeddings=None, index_path=None) -> FAISS | None
    build_vectorstore(chunks, embeddings=None, index_path=None) -> FAISS
    add_documents(chunks, embeddings=None, index_path=None) -> FAISS
    is_file_indexed(filename, file_hash, index_path=None) -> bool
    get_indexed_files(index_path=None) -> dict
    clear_vectorstore(index_path=None) -> bool
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

import config
from core.embeddings import get_embeddings
from core.logger import get_logger

log = get_logger(__name__)


def _get_paths(index_path: str | Path | None = None) -> tuple[Path, Path]:
    """Return (index_dir, metadata_file) as resolved Path objects."""
    base_dir = Path(index_path) if index_path else Path(config.FAISS_INDEX_PATH)
    meta_file = base_dir / "indexed_files.json"
    return base_dir, meta_file


def get_indexed_files(index_path: str | Path | None = None) -> dict[str, dict]:
    """Return dictionary of currently indexed files from the metadata JSON."""
    _, meta_file = _get_paths(index_path)
    if not meta_file.exists():
        return {}
    try:
        with open(meta_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not read index metadata %s: %s", meta_file, exc)
        return {}


def _save_indexed_files(manifest: dict[str, dict], index_path: str | Path | None = None) -> None:
    """Save the indexed files dictionary to metadata JSON."""
    base_dir, meta_file = _get_paths(index_path)
    base_dir.mkdir(parents=True, exist_ok=True)
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def is_file_indexed(filename: str, file_hash: str, index_path: str | Path | None = None) -> bool:
    """Return True if the file with matching SHA-256 hash has already been indexed."""
    manifest = get_indexed_files(index_path)
    entry = manifest.get(filename)
    if not entry:
        return False
    return entry.get("file_hash") == file_hash


def load_vectorstore(
    embeddings=None,
    index_path: str | Path | None = None,
) -> FAISS | None:
    """Load an existing FAISS vector store from disk.

    Returns None if the index files do not exist.
    """
    base_dir, _ = _get_paths(index_path)
    index_file = base_dir / "index.faiss"
    pkl_file = base_dir / "index.pkl"

    if not (index_file.exists() and pkl_file.exists()):
        log.info("No existing FAISS index found at %s", base_dir)
        return None

    embed_model = embeddings or get_embeddings()
    try:
        store = FAISS.load_local(
            folder_path=str(base_dir),
            embeddings=embed_model,
            allow_dangerous_deserialization=True,
        )
        log.info("Successfully loaded FAISS index from %s", base_dir)
        return store
    except Exception as exc:  # noqa: BLE001
        log.error("Failed to load FAISS index from %s: %s", base_dir, exc)
        return None


def _update_manifest_for_chunks(
    chunks: Sequence[Document],
    index_path: str | Path | None = None,
) -> None:
    """Update metadata manifest for new chunks."""
    manifest = get_indexed_files(index_path)
    per_file: dict[str, dict] = {}

    for doc in chunks:
        src = doc.metadata.get("source", "unknown")
        f_hash = doc.metadata.get("file_hash", "")
        page = doc.metadata.get("page", 1)

        if src not in per_file:
            per_file[src] = {
                "file_hash": f_hash,
                "chunks": 0,
                "pages": set(),
                "indexed_at": datetime.now(timezone.utc).isoformat(),
            }
        per_file[src]["chunks"] += 1
        per_file[src]["pages"].add(page)

    for src, info in per_file.items():
        manifest[src] = {
            "file_hash": info["file_hash"],
            "chunks": info["chunks"],
            "page_count": len(info["pages"]),
            "indexed_at": info["indexed_at"],
        }

    _save_indexed_files(manifest, index_path)


def build_vectorstore(
    chunks: Sequence[Document],
    embeddings=None,
    index_path: str | Path | None = None,
) -> FAISS:
    """Create a new FAISS vector store from chunks and persist to disk.

    Overwrites any existing index at index_path.
    """
    if not chunks:
        raise ValueError("Cannot build vector store: chunks sequence is empty.")

    base_dir, _ = _get_paths(index_path)
    base_dir.mkdir(parents=True, exist_ok=True)

    embed_model = embeddings or get_embeddings()
    log.info("Building FAISS index with %d chunks...", len(chunks))

    store = FAISS.from_documents(list(chunks), embed_model)
    store.save_local(str(base_dir))

    _save_indexed_files({}, index_path)  # reset manifest for fresh build
    _update_manifest_for_chunks(chunks, index_path)

    log.info("Saved FAISS index to %s", base_dir)
    return store


def add_documents(
    chunks: Sequence[Document],
    embeddings=None,
    index_path: str | Path | None = None,
) -> FAISS:
    """Add chunks to an existing FAISS vector store, or create a new one.

    Persists the updated store and metadata to disk.
    """
    if not chunks:
        existing = load_vectorstore(embeddings, index_path)
        if existing is not None:
            return existing
        raise ValueError("Cannot create vector store: chunks sequence is empty.")

    base_dir, _ = _get_paths(index_path)
    base_dir.mkdir(parents=True, exist_ok=True)
    embed_model = embeddings or get_embeddings()

    existing = load_vectorstore(embed_model, index_path)
    if existing is None:
        log.info("No index found; creating fresh index with %d chunks.", len(chunks))
        return build_vectorstore(chunks, embed_model, index_path)

    log.info("Adding %d chunks to existing FAISS index...", len(chunks))
    existing.add_documents(list(chunks))
    existing.save_local(str(base_dir))
    _update_manifest_for_chunks(chunks, index_path)

    return existing


def clear_vectorstore(index_path: str | Path | None = None) -> bool:
    """Delete the persisted FAISS index files and metadata.

    Returns True if successfully cleared.
    """
    base_dir, _ = _get_paths(index_path)
    if not base_dir.exists():
        return True

    try:
        shutil.rmtree(base_dir)
        base_dir.mkdir(parents=True, exist_ok=True)
        log.info("Cleared vector store at %s", base_dir)
        return True
    except Exception as exc:  # noqa: BLE001
        log.error("Failed to clear vector store at %s: %s", base_dir, exc)
        return False
