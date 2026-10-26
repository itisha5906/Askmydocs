"""
retriever.py — Semantic search and MMR retrieval with document filtering.

Supports top-k similarity search, Maximal Marginal Relevance (MMR) for diversity,
and metadata filtering by document source filename.

Public API
----------
    get_retriever(vectorstore=None, top_k=6, use_mmr=False, doc_filter=None)
    retrieve(query, vectorstore=None, top_k=6, use_mmr=False, doc_filter=None) -> list[Document]
"""

from __future__ import annotations

from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

import config
from core.logger import get_logger
from core.vectorstore import load_vectorstore

log = get_logger(__name__)


def get_retriever(
    vectorstore: FAISS | None = None,
    top_k: int = 6,
    use_mmr: bool = False,
    doc_filter: str | None = None,
) -> BaseRetriever:
    """Configure and return a LangChain retriever.

    Parameters
    ----------
    vectorstore:
        FAISS instance. If None, loaded from disk via load_vectorstore().
    top_k:
        Number of documents to return.
    use_mmr:
        If True, use Maximal Marginal Relevance to balance relevance & diversity.
    doc_filter:
        Optional filename to filter results by (matching chunk metadata['source']).
        If None, 'all', or 'All Documents', no filter is applied.

    Returns
    -------
    BaseRetriever
    """
    store = vectorstore or load_vectorstore()
    if store is None:
        raise ValueError("Vector store is not initialized or index directory is empty.")

    k = max(1, int(top_k))
    search_type = "mmr" if use_mmr else "similarity"
    search_kwargs: dict[str, Any] = {"k": k}

    if use_mmr:
        search_kwargs["fetch_k"] = k * config.MMR_FETCH_K_MULTIPLIER
        search_kwargs["lambda_mult"] = config.MMR_LAMBDA

    # Apply document filter if requested and not "all"
    if doc_filter and doc_filter.strip().lower() not in ("all", "all documents", ""):
        search_kwargs["filter"] = {"source": doc_filter.strip()}

    log.debug(
        "Creating retriever: type=%s, k=%d, filter=%s",
        search_type,
        k,
        search_kwargs.get("filter"),
    )

    return store.as_retriever(search_type=search_type, search_kwargs=search_kwargs)


def retrieve(
    query: str,
    vectorstore: FAISS | None = None,
    top_k: int = 6,
    use_mmr: bool = False,
    doc_filter: str | None = None,
) -> list[Document]:
    """Execute retrieval directly and return list of matched Documents.

    Parameters
    ----------
    query:
        The search string or question.
    vectorstore:
        FAISS instance or None to load from disk.
    top_k:
        Number of relevant documents.
    use_mmr:
        Whether to use MMR search.
    doc_filter:
        Optional filename filter.

    Returns
    -------
    list[Document]
    """
    retriever = get_retriever(
        vectorstore=vectorstore,
        top_k=top_k,
        use_mmr=use_mmr,
        doc_filter=doc_filter,
    )
    return retriever.invoke(query)
