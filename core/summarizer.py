"""
summarizer.py — Map-Reduce Document Summarization.

Produces summaries of single documents or across all uploaded documents using a
hierarchical map-reduce approach to stay well within LLM context limits.
Supports multiple styles: 'short', 'detailed', and 'bullets'.

Public API
----------
    summarize_documents(chunks, style, provider, model) -> dict
"""

from __future__ import annotations

from typing import Any, Sequence

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from core.llm_factory import invoke_with_fallback
from core.logger import get_logger

log = get_logger(__name__)

# Number of chunks to process together in a single map batch
_MAP_BATCH_SIZE = 4

MAP_SYSTEM_PROMPT = """You are a precise technical summarizer.
Summarize the following document excerpt into a concise digest preserving key facts,
methodologies, terms, and findings. Be factual and objective.
"""

STYLE_PROMPTS = {
    "short": (
        "Provide a concise, cohesive 2-to-3 paragraph executive summary of the document. "
        "Highlight the overarching purpose, core findings, and overall conclusion."
    ),
    "detailed": (
        "Provide a structured, in-depth summary with clear sections (e.g., Background/Objective, "
        "Core Methodology/Architecture, Key Details & Results, and Conclusions). "
        "Preserve technical specifics and critical takeaways."
    ),
    "bullets": (
        "Provide a comprehensive list of bullet points detailing the most critical takeaways, "
        "specifications, findings, and actionable conclusions from the document."
    ),
}


def _summarize_batch(
    chunks_batch: Sequence[Document],
    provider: str,
    model: str | None,
) -> str:
    """Map phase: summarize a small batch of chunks."""
    combined_text = "\n\n".join(doc.page_content.strip() for doc in chunks_batch)
    messages = [
        SystemMessage(content=MAP_SYSTEM_PROMPT),
        HumanMessage(content=f"Excerpt:\n{combined_text}\n\nProvide concise digest:"),
    ]
    summary, _ = invoke_with_fallback(
        messages=messages,
        primary_provider=provider,
        primary_model=model,
        temperature=0.2,
    )
    return summary.strip()


def summarize_documents(
    chunks: Sequence[Document],
    style: str = "detailed",
    provider: str = "Groq",
    model: str | None = None,
) -> dict[str, Any]:
    """Execute map-reduce summarization across a sequence of document chunks.

    Parameters
    ----------
    chunks:
        List of LangChain Document chunks representing the document(s).
    style:
        'short', 'detailed', or 'bullets'. Defaults to 'detailed'.
    provider:
        'Groq' or 'OpenAI-compatible'.
    model:
        Model ID or None for default.

    Returns
    -------
    dict
        {
            "summary": str,
            "pages": list[int],  # sorted unique 1-indexed page numbers
            "chunks_processed": int,
        }
    """
    if not chunks:
        return {
            "summary": "No document content provided to summarize.",
            "pages": [],
            "chunks_processed": 0,
        }

    # Extract all unique page numbers
    unique_pages = sorted(
        {doc.metadata.get("page") for doc in chunks if doc.metadata.get("page") is not None}
    )

    clean_style = style.lower().strip()
    style_instruction = STYLE_PROMPTS.get(clean_style, STYLE_PROMPTS["detailed"])

    # If small number of chunks, skip Map phase and do direct Reduce
    if len(chunks) <= _MAP_BATCH_SIZE:
        combined_text = "\n\n".join(doc.page_content.strip() for doc in chunks)
        reduce_prompt = (
            f"Document Content:\n{combined_text}\n\n"
            f"Instructions:\n{style_instruction}\n\nSummary:"
        )
        final_summary, _ = invoke_with_fallback(
            messages=[HumanMessage(content=reduce_prompt)],
            primary_provider=provider,
            primary_model=model,
            temperature=0.3,
        )
        return {
            "summary": final_summary.strip(),
            "pages": unique_pages,
            "chunks_processed": len(chunks),
        }

    # 1. Map Phase: batch chunks and summarize each
    intermediate_summaries: list[str] = []
    total_chunks = len(chunks)

    for i in range(0, total_chunks, _MAP_BATCH_SIZE):
        batch = chunks[i : i + _MAP_BATCH_SIZE]
        batch_summary = _summarize_batch(batch, provider=provider, model=model)
        if batch_summary:
            intermediate_summaries.append(batch_summary)

    # 2. Reduce Phase: combine intermediate summaries according to requested style
    combined_intermediates = "\n\n".join(
        f"Section {idx}:\n{s}" for idx, s in enumerate(intermediate_summaries, 1)
    )
    final_prompt = (
        f"Combined Section Summaries:\n{combined_intermediates}\n\n"
        f"Task:\nSynthesize the above sections into a unified, high-quality document summary.\n"
        f"Style Requirements:\n{style_instruction}\n\nFinal Summary:"
    )

    final_summary, _ = invoke_with_fallback(
        messages=[HumanMessage(content=final_prompt)],
        primary_provider=provider,
        primary_model=model,
        temperature=0.3,
    )

    return {
        "summary": final_summary.strip(),
        "pages": unique_pages,
        "chunks_processed": total_chunks,
    }
