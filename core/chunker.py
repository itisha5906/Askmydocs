"""
chunker.py — Document chunking for AskMyDocs.

Splits LangChain Documents into overlapping chunks using
RecursiveCharacterTextSplitter. Every chunk inherits and extends the source
Document's metadata so that the retrieval layer can cite the exact filename,
page number, and chunk position.

Public API
----------
    chunk_documents(docs, chunk_size, chunk_overlap) -> list[Document]
"""

from __future__ import annotations

import uuid
from typing import Sequence

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.logger import get_logger

log = get_logger(__name__)

# Minimum characters a chunk must have to be kept (avoids noise fragments).
_MIN_CHUNK_CHARS = 20


def chunk_documents(
    docs: Sequence[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> list[Document]:
    """Split a sequence of Documents into overlapping text chunks.

    Each input Document is split independently so that page boundaries are
    preserved. The resulting chunks inherit the parent's metadata and receive
    three extra fields:

    - ``chunk_id``     — UUID4 hex string uniquely identifying this chunk.
    - ``chunk_index``  — 0-based position of the chunk within its source page.
    - ``total_chars``  — character length of the chunk's content.
    - ``is_ocr``       — forwarded from the parent document's ``ocr`` flag.

    Parameters
    ----------
    docs:
        Iterable of LangChain ``Document`` objects (e.g. from ``loader.load_pdf``).
    chunk_size:
        Maximum number of characters per chunk (default 800).
    chunk_overlap:
        Number of characters to overlap between consecutive chunks (default 150).

    Returns
    -------
    list[Document]
        Flat list of chunk Documents with enriched metadata.
        Chunks shorter than ``_MIN_CHUNK_CHARS`` are discarded.
    """
    if not docs:
        log.warning("chunk_documents() received an empty document list.")
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
        is_separator_regex=False,
    )

    all_chunks: list[Document] = []

    for doc in docs:
        if not doc.page_content or not doc.page_content.strip():
            log.debug(
                "Skipping empty page %s of '%s'.",
                doc.metadata.get("page", "?"),
                doc.metadata.get("source", "unknown"),
            )
            continue

        raw_chunks = splitter.split_text(doc.page_content)

        for idx, chunk_text in enumerate(raw_chunks):
            # Discard noise fragments
            if len(chunk_text.strip()) < _MIN_CHUNK_CHARS:
                continue

            # Build metadata: copy parent metadata, then add chunk-specific fields
            metadata = {
                **doc.metadata,
                "chunk_id": uuid.uuid4().hex,
                "chunk_index": idx,
                "total_chars": len(chunk_text),
                # Normalise the OCR flag name to "is_ocr" for consistency
                "is_ocr": doc.metadata.get("ocr", False),
            }

            all_chunks.append(Document(page_content=chunk_text, metadata=metadata))

    log.info(
        "Chunked %d document(s) into %d chunks (size=%d, overlap=%d).",
        len(docs),
        len(all_chunks),
        chunk_size,
        chunk_overlap,
    )
    return all_chunks
