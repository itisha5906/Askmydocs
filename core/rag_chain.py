"""
rag_chain.py — Grounded Question Answering Chain with Citation Generation.

Retrieves context documents, formats a strictly grounded prompt that prevents
hallucinations, maintains conversation memory across the last N turns, and returns
the answer accompanied by exact document name, page number, and text snippets.

Public API
----------
    answer_question(query, history, provider, model, top_k, use_mmr, doc_filter, ...) -> dict
"""

from __future__ import annotations

import time
from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from core.llm_factory import invoke_with_fallback
from core.logger import get_logger
from core.retriever import retrieve

log = get_logger(__name__)

NOT_FOUND_MESSAGE = "I could not find the answer in the provided documents."

SYSTEM_PROMPT = """You are a strict, factual document retrieval assistant for the AskMyDocs system.
Your task is to answer the user's question using ONLY the retrieved document excerpts provided below.

Strict Constraints:
1. Base your answer ENTIRELY on the provided Context.
2. If the answer is not directly mentioned in the Context, respond EXACTLY with:
   "I could not find the answer in the provided documents."
3. Do NOT extrapolate, speculate, or bring in outside knowledge.
4. When stating facts, refer naturally to the source if relevant (e.g. 'According to [source]...').
"""


def _format_context(docs: list[Document]) -> str:
    """Format retrieved documents into labeled context blocks."""
    if not docs:
        return "No relevant documents found."

    blocks: list[str] = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown Document")
        page = doc.metadata.get("page", "?")
        ocr_flag = " (OCR)" if doc.metadata.get("is_ocr") else ""
        content = doc.page_content.strip()
        blocks.append(f"--- Document [{i}]: {source}, Page {page}{ocr_flag} ---\n{content}")

    return "\n\n".join(blocks)


def _format_history(history: list[dict[str, str]] | None, max_turns: int = 6) -> list[Any]:
    """Convert conversation history dicts to LangChain Message objects."""
    if not history:
        return []

    recent = history[-max_turns:]
    messages: list[Any] = []
    for turn in recent:
        role = turn.get("role", "").lower()
        content = turn.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role in ("assistant", "ai"):
            messages.append(AIMessage(content=content))
    return messages


def _extract_sources(docs: list[Document]) -> list[dict[str, Any]]:
    """Format citation metadata and snippets from retrieved documents."""
    sources: list[dict[str, Any]] = []
    for doc in docs:
        content = doc.page_content.strip()
        snippet = content[:250] + ("..." if len(content) > 250 else "")
        sources.append(
            {
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", 1),
                "snippet": snippet,
                "is_ocr": bool(doc.metadata.get("is_ocr", False)),
                "chunk_id": doc.metadata.get("chunk_id", ""),
            }
        )
    return sources


def answer_question(
    query: str,
    history: list[dict[str, str]] | None = None,
    provider: str = "Groq",
    model: str | None = None,
    top_k: int = 6,
    use_mmr: bool = False,
    doc_filter: str | None = None,
    vectorstore: Any = None,
) -> dict[str, Any]:
    """Answer a user query with strict document grounding and citations.

    Parameters
    ----------
    query:
        The user's question.
    history:
        List of preceding conversation turns: [{'role': 'user'|'assistant', 'content': str}].
    provider:
        'Groq' or 'OpenAI-compatible'.
    model:
        Model ID or None for default.
    top_k:
        Number of chunks to retrieve.
    use_mmr:
        Whether to use MMR search for diverse chunks.
    doc_filter:
        Optional filename to restrict search to a single document.
    vectorstore:
        Optional pre-loaded FAISS vector store.

    Returns
    -------
    dict
        {
            "answer": str,
            "sources": list[dict],
            "latency": float,
            "provider_used": str,
        }
    """
    start_time = time.perf_counter()

    # 1. Retrieve relevant chunks
    try:
        retrieved_docs = retrieve(
            query=query,
            vectorstore=vectorstore,
            top_k=top_k,
            use_mmr=use_mmr,
            doc_filter=doc_filter,
        )
    except Exception as exc:
        log.warning("Retrieval failed (%s); treating as empty context.", exc)
        retrieved_docs = []

    if not retrieved_docs:
        elapsed = round(time.perf_counter() - start_time, 2)
        return {
            "answer": NOT_FOUND_MESSAGE,
            "sources": [],
            "latency": elapsed,
            "provider_used": "None (No docs retrieved)",
        }

    # 2. Build messages with context and history
    context_text = _format_context(retrieved_docs)
    user_prompt = f"Context:\n{context_text}\n\nUser Question:\n{query}"

    messages: list[Any] = [SystemMessage(content=SYSTEM_PROMPT)]
    messages.extend(_format_history(history, max_turns=6))
    messages.append(HumanMessage(content=user_prompt))

    # 3. Call LLM with automatic fallback
    answer_text, provider_used = invoke_with_fallback(
        messages=messages,
        primary_provider=provider,
        primary_model=model,
        temperature=0.0,
    )

    elapsed = round(time.perf_counter() - start_time, 2)
    sources = _extract_sources(retrieved_docs)

    return {
        "answer": answer_text.strip(),
        "sources": sources,
        "latency": elapsed,
        "provider_used": provider_used,
    }
