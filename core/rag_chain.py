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

CONDENSE_PROMPT = """Given the chat history and a follow-up question, rephrase the follow-up question into a single, complete standalone question that can be understood without the chat history.
Do NOT answer the question. If the question is already complete and standalone, return it unchanged.

Chat History:
{history}

Follow-up Question: {question}
Standalone Question:"""

SYSTEM_PROMPT = """You are a strict, factual document retrieval assistant for the AskMyDocs system.
Your task is to answer the user's question using ONLY the retrieved document excerpts provided below.

Strict Constraints:
1. Base your answer ENTIRELY on the provided Context.
2. If the answer is not directly mentioned in the Context, respond EXACTLY with:
   "I could not find the answer in the provided documents."
3. Do NOT extrapolate, speculate, or bring in outside knowledge.
4. When stating facts, refer naturally to the source if relevant (e.g. 'According to [source]...').
"""


def condense_question(
    query: str,
    history: list[dict[str, str]] | None = None,
    provider: str = "Groq",
    model: str | None = None,
) -> str:
    """Rewrite follow-up question into a standalone query using recent history."""
    if not history:
        return query.strip()

    history_str = "\n".join(
        f"{turn.get('role', 'user').capitalize()}: {turn.get('content', '')}"
        for turn in history[-4:]
    )
    prompt = CONDENSE_PROMPT.format(history=history_str, question=query)

    try:
        rephrased, _ = invoke_with_fallback(
            messages=[HumanMessage(content=prompt)],
            primary_provider=provider,
            primary_model=model,
            temperature=0.0,
            max_tokens=256,
        )
        cleaned = rephrased.strip().strip('"').strip("'")
        if cleaned:
            log.debug("Condensed query '%s' -> '%s'", query, cleaned)
            return cleaned
    except Exception as exc:
        log.warning("Query condensation failed (%s); using original query.", exc)

    return query.strip()


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
        score = doc.metadata.get("score")
        sources.append(
            {
                "source": doc.metadata.get("source", "Unknown"),
                "filename": doc.metadata.get("filename", doc.metadata.get("source", "Unknown")),
                "page": doc.metadata.get("page", 1),
                "snippet": snippet,
                "is_ocr": bool(doc.metadata.get("is_ocr", False)),
                "chunk_id": doc.metadata.get("chunk_id", ""),
                "chunk_number": doc.metadata.get("chunk_number", doc.metadata.get("chunk_index", 0) + 1),
                "score": score,
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
    temperature: float = 0.0,
    max_tokens: int = 1024,
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
    temperature:
        Sampling temperature (default 0.0 for strict factual answers).
    max_tokens:
        Maximum output tokens.

    Returns
    -------
    dict
        {
            "answer": str,
            "sources": list[dict],
            "latency": float,
            "provider_used": str,
            "fallback_used": bool,
            "standalone_query": str,
        }
    """
    start_time = time.perf_counter()

    # 1. Condense query if history is present
    standalone_query = condense_question(
        query=query,
        history=history,
        provider=provider,
        model=model,
    )

    # 2. Retrieve relevant chunks using standalone query
    try:
        retrieved_docs = retrieve(
            query=standalone_query,
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
            "fallback_used": False,
            "standalone_query": standalone_query,
        }

    # 3. Build messages with context and history
    context_text = _format_context(retrieved_docs)
    user_prompt = f"Context:\n{context_text}\n\nUser Question:\n{query}"

    messages: list[Any] = [SystemMessage(content=SYSTEM_PROMPT)]
    messages.extend(_format_history(history, max_turns=6))
    messages.append(HumanMessage(content=user_prompt))

    # 4. Call LLM with automatic fallback
    llm_res = invoke_with_fallback(
        messages=messages,
        primary_provider=provider,
        primary_model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    answer_text = llm_res[0] if isinstance(llm_res, tuple) else str(llm_res)
    provider_used = llm_res[1] if isinstance(llm_res, tuple) and len(llm_res) > 1 else provider
    fallback_used = getattr(llm_res, "fallback_used", False) or ("fallback" in str(provider_used).lower())

    elapsed = round(time.perf_counter() - start_time, 2)
    sources = _extract_sources(retrieved_docs)

    return {
        "answer": answer_text.strip(),
        "sources": sources,
        "latency": elapsed,
        "provider_used": provider_used,
        "fallback_used": fallback_used,
        "standalone_query": standalone_query,
    }
