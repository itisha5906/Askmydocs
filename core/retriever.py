"""
retriever.py — Semantic search and MMR retrieval with document filtering.

Supports top-k similarity search, Maximal Marginal Relevance (MMR) for diversity,
and metadata filtering by document source filename.

Public API
----------
    get_retriever(vectorstore=None, top_k=6, use_mmr=False, doc_filter=None)
    retrieve(query, vectorstore=None, top_k=6, use_mmr=False, doc_filter=None) -> list[Document]
    retrieve_with_scores(query, vectorstore=None, top_k=6, use_mmr=False, doc_filter=None) -> list[tuple[Document, float]]
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


def retrieve_with_scores(
    query: str,
    vectorstore: FAISS | None = None,
    top_k: int = 6,
    use_mmr: bool = False,
    doc_filter: str | None = None,
) -> list[tuple[Document, float]]:
    """Execute retrieval and return list of (Document, score) pairs.

    For cosine similarity on normalized vectors, the score represents distance.
    Lower distance means higher similarity.
    """
    store = vectorstore or load_vectorstore()
    if store is None:
        raise ValueError("Vector store is not initialized or index directory is empty.")

    k = max(1, int(top_k))
    filter_dict = None
    if doc_filter and doc_filter.strip().lower() not in ("all", "all documents", ""):
        filter_dict = {"source": doc_filter.strip()}

    if use_mmr:
        fetch_k = k * config.MMR_FETCH_K_MULTIPLIER
        docs = store.max_marginal_relevance_search(
            query,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=config.MMR_LAMBDA,
            filter=filter_dict,
        )
        # For MMR results, compute similarity score for each returned document
        results: list[tuple[Document, float]] = []
        for d in docs:
            # Check if d has a score, or do a targeted similarity search
            sim_hits = store.similarity_search_with_score(d.page_content[:200], k=1)
            score = round(float(sim_hits[0][1]), 4) if sim_hits else 0.0
            d.metadata["score"] = score
            results.append((d, score))
        return results

    # Standard similarity search with score
    hits = store.similarity_search_with_score(query, k=k, filter=filter_dict)
    results = []
    for doc, score in hits:
        doc.metadata["score"] = round(float(score), 4)
        results.append((doc, round(float(score), 4)))
    return results


def retrieve(
    query: str,
    vectorstore: FAISS | None = None,
    top_k: int = 6,
    use_mmr: bool = False,
    doc_filter: str | None = None,
) -> list[Document]:
    """Execute retrieval directly and return list of matched Documents.

    Each document's metadata will include 'score' (similarity score/distance).

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
    try:
        scored_pairs = retrieve_with_scores(
            query=query,
            vectorstore=vectorstore,
            top_k=top_k,
            use_mmr=use_mmr,
            doc_filter=doc_filter,
        )
        return [doc for doc, _ in scored_pairs]
    except Exception as exc:
        log.warning("retrieve_with_scores failed (%s); falling back to retriever object.", exc)
        retriever = get_retriever(
            vectorstore=vectorstore,
            top_k=top_k,
            use_mmr=use_mmr,
            doc_filter=doc_filter,
        )
        return retriever.invoke(query)
