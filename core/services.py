"""
services.py — Central Application Service Facade for AskMyDocs.

Provides a unified interface consumed by the Streamlit Presentation Layer and
evaluation scripts, isolating all orchestration details across document ingestion,
RAG answering, summarization, vector store operations, and provider management.

Public API
----------
    ingest_files(files) -> dict[str, dict]
    ask(question, history, provider, model, top_k, use_mmr, doc_filter) -> dict
    summarize(doc_name_or_all, style, provider, model) -> dict
    list_documents() -> list[dict]
    clear_index() -> bool
    get_models() -> dict[str, list[str]]
    provider_status() -> dict[str, dict]
    get_logs(level='ALL', max_lines=100) -> list[str]
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

import config
from core.chunker import chunk_documents
from core.llm_factory import check_provider_status, get_groq_models
from core.loader import _compute_file_hash, load_pdf
from core.logger import get_logger
from core.preprocess import clean_text, detect_and_remove_headers_footers
from core.rag_chain import answer_question
from core.summarizer import summarize_documents
from core.vectorstore import (
    add_documents,
    clear_vectorstore,
    get_indexed_files,
    is_file_indexed,
    load_vectorstore,
)

log = get_logger(__name__)


def ingest_files(files: list[Any]) -> dict[str, dict[str, Any]]:
    """Process and index one or more uploaded files or paths.

    Accepts file paths (str/Path) or Streamlit UploadedFile objects.
    Extracts text (with OCR fallback for scanned pages), cleans headers/footers,
    chunks the content, embeds and updates the FAISS vector store.

    Parameters
    ----------
    files:
        List of filepath strings, Path objects, or Streamlit UploadedFile instances.

    Returns
    -------
    dict[str, dict]
        Mapping from filename to status:
        {"filename": {"pages": int, "chars": int, "ocr_pages": int, "status": str}}
    """
    results: dict[str, dict[str, Any]] = {}
    temp_dir = Path(tempfile.gettempdir()) / "askmydocs_uploads"
    temp_dir.mkdir(parents=True, exist_ok=True)

    for item in files:
        # Determine filename and file path on disk
        if hasattr(item, "name") and hasattr(item, "getbuffer"):
            # Streamlit UploadedFile
            filename = item.name
            target_path = temp_dir / filename
            with open(target_path, "wb") as f:
                f.write(item.getbuffer())
        elif isinstance(item, (str, Path)):
            target_path = Path(item)
            filename = target_path.name
        else:
            log.warning("Unrecognized file type passed to ingest_files: %s", type(item))
            continue

        if not target_path.exists():
            results[filename] = {
                "pages": 0,
                "chars": 0,
                "ocr_pages": 0,
                "status": f"File not found: {target_path}",
            }
            continue

        try:
            # 1. Check if already indexed by hash
            f_hash = _compute_file_hash(target_path)
            if is_file_indexed(filename, f_hash):
                manifest = get_indexed_files()
                entry = manifest.get(filename, {})
                log.info("File '%s' is already indexed with matching hash. Skipping.", filename)
                results[filename] = {
                    "pages": entry.get("page_count", 0),
                    "chars": 0,
                    "ocr_pages": 0,
                    "status": "already_indexed",
                }
                continue

            # 2. Extract pages via loader (with OCR fallback)
            docs = load_pdf(target_path)
            total_pages = len(docs)
            total_chars = sum(len(d.page_content) for d in docs)
            ocr_pages = sum(1 for d in docs if d.metadata.get("ocr"))

            # 3. Clean text & remove headers/footers
            cleaned_docs: list[Document] = []
            for d in docs:
                cleaned_text = clean_text(d.page_content)
                cleaned_docs.append(
                    Document(page_content=cleaned_text, metadata=dict(d.metadata))
                )

            cleaned_docs = detect_and_remove_headers_footers(cleaned_docs)

            # 4. Chunk documents
            chunks = chunk_documents(
                cleaned_docs,
                chunk_size=config.CHUNK_SIZE,
                chunk_overlap=config.CHUNK_OVERLAP,
            )

            # 5. Add chunks to vector store
            add_documents(chunks)

            results[filename] = {
                "pages": total_pages,
                "chars": total_chars,
                "ocr_pages": ocr_pages,
                "status": "indexed",
            }
            log.info(
                "Successfully indexed '%s': %d pages, %d chars, %d chunks (%d OCR pages)",
                filename,
                total_pages,
                total_chars,
                len(chunks),
                ocr_pages,
            )

        except Exception as exc:  # noqa: BLE001
            log.error("Failed to ingest file '%s': %s", filename, exc, exc_info=True)
            results[filename] = {
                "pages": 0,
                "chars": 0,
                "ocr_pages": 0,
                "status": f"Error: {exc}",
            }

    return results


def ask(
    question: str,
    history: list[dict[str, str]] | None = None,
    provider: str = "Groq",
    model: str | None = None,
    top_k: int = 6,
    use_mmr: bool = False,
    doc_filter: str | None = None,
) -> dict[str, Any]:
    """Execute a RAG query and return the answer, citations, and latency.

    Returns
    -------
    dict
        {"answer": str, "sources": list[dict], "latency": float, "provider_used": str}
    """
    return answer_question(
        query=question,
        history=history,
        provider=provider,
        model=model,
        top_k=top_k,
        use_mmr=use_mmr,
        doc_filter=doc_filter,
    )


def summarize(
    doc_name_or_all: str = "all",
    style: str = "detailed",
    provider: str = "Groq",
    model: str | None = None,
) -> dict[str, Any]:
    """Summarize a single document or all documents in the vector store.

    Parameters
    ----------
    doc_name_or_all:
        Specific filename or 'all' to summarize everything.
    style:
        'short', 'detailed', or 'bullets'.
    provider:
        'Groq' or 'OpenAI-compatible'.
    model:
        Model ID or None for default.

    Returns
    -------
    dict
        {"summary": str, "pages": list[int], "chunks_processed": int}
    """
    store = load_vectorstore()
    if store is None or not hasattr(store, "docstore"):
        return {
            "summary": "No documents are currently indexed in the vector store.",
            "pages": [],
            "chunks_processed": 0,
        }

    # Retrieve all chunks from the in-memory docstore
    all_chunks: list[Document] = list(store.docstore._dict.values())  # type: ignore[attr-defined]

    target = doc_name_or_all.strip()
    if target.lower() not in ("all", "all documents", ""):
        filtered = [c for c in all_chunks if c.metadata.get("source") == target]
    else:
        filtered = all_chunks

    if not filtered:
        return {
            "summary": f"No chunks found for document '{doc_name_or_all}'.",
            "pages": [],
            "chunks_processed": 0,
        }

    # Sort chunks by source and page order
    filtered.sort(
        key=lambda c: (
            str(c.metadata.get("source", "")),
            int(c.metadata.get("page", 0)),
            int(c.metadata.get("chunk_index", 0)),
        )
    )

    return summarize_documents(
        chunks=filtered,
        style=style,
        provider=provider,
        model=model,
    )


def list_documents() -> list[dict[str, Any]]:
    """Return a list of all currently indexed documents and their stats."""
    manifest = get_indexed_files()
    documents: list[dict[str, Any]] = []
    for filename, info in manifest.items():
        documents.append(
            {
                "filename": filename,
                "chunks": info.get("chunks", 0),
                "page_count": info.get("page_count", 0),
                "indexed_at": info.get("indexed_at", ""),
                "file_hash": info.get("file_hash", ""),
            }
        )
    return documents


def clear_index() -> bool:
    """Clear all stored vectors and metadata."""
    return clear_vectorstore()


def get_models() -> dict[str, list[str]]:
    """Return dictionary of available models per provider."""
    return {
        "Groq": get_groq_models(),
        "OpenAI-compatible": [config.OPENAI_MODEL],
    }


def provider_status() -> dict[str, dict]:
    """Return configuration and connectivity status of LLM providers."""
    return check_provider_status()


def get_logs(level: str = "ALL", max_lines: int = 100) -> list[str]:
    """Return the last max_lines log lines, optionally filtered by level."""
    from core.logger import get_log_contents

    raw_text = get_log_contents()
    lines = [line for line in raw_text.splitlines() if line.strip()]

    # If in-memory buffer is small, fall back to reading from config.LOG_FILE if available
    if len(lines) < 5 and config.LOG_FILE.exists():
        try:
            with open(config.LOG_FILE, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
        except Exception:  # noqa: BLE001
            pass

    lvl = level.upper().strip()
    if lvl not in ("ALL", ""):
        lines = [line for line in lines if f"| {lvl}" in line or f"| {lvl} " in line]

    return lines[-max_lines:]

