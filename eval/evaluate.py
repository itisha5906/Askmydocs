"""
evaluate.py — Automated Benchmark and Model Evaluation for AskMyDocs.

Evaluates retrieval accuracy (Hit@k), answer quality (keyword inclusion),
refusal accuracy on unanswerable questions, and average response latency across
configured LLM models.

Usage:
    python eval/evaluate.py
Output:
    eval/results.csv
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import HumanMessage, SystemMessage

import config
from core.llm_factory import get_groq_models, get_llm
from core.logger import get_logger
from core.rag_chain import NOT_FOUND_MESSAGE, SYSTEM_PROMPT, _format_context
from core.retriever import retrieve
from core.services import ingest_files, list_documents

log = get_logger(__name__)

EVAL_DIR = Path(__file__).parent
QUESTIONS_FILE = EVAL_DIR / "questions.json"
RESULTS_FILE = EVAL_DIR / "results.csv"
SAMPLE_PDF = Path(__file__).parent.parent / "sample_docs" / "sample_text.pdf"


def ensure_index() -> None:
    """Ensure sample document is indexed in the vector store before evaluation."""
    docs = list_documents()
    indexed_names = [d["filename"] for d in docs]
    if "sample_text.pdf" not in indexed_names and SAMPLE_PDF.exists():
        print("Indexing sample_docs/sample_text.pdf for evaluation...")
        ingest_files([SAMPLE_PDF])


def load_questions() -> list[dict[str, Any]]:
    """Load evaluation questions from questions.json."""
    if not QUESTIONS_FILE.exists():
        raise FileNotFoundError(f"Questions file not found: {QUESTIONS_FILE}")
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def is_refusal(text: str) -> bool:
    """Check if model correctly refused to answer when information is absent."""
    t = text.lower()
    return (
        "not found in the documents" in t
        or "could not find" in t
        or "not mentioned" in t
        or "not provided" in t
        or "does not contain" in t
        or NOT_FOUND_MESSAGE.lower() in t
    )


def run_evaluation(num_runs_per_q: int = 3) -> None:
    """Run full evaluation suite across all questions and models."""
    ensure_index()
    questions = load_questions()

    # Determine candidate models to test
    models_to_test: list[tuple[str, str]] = []
    groq_models = get_groq_models()

    # Pick large and small model from Groq
    large_model = next((m for m in groq_models if "70b" in m.lower()), groq_models[0] if groq_models else None)
    small_model = next((m for m in groq_models if "8b" in m.lower()), groq_models[1] if len(groq_models) > 1 else None)

    if large_model:
        models_to_test.append(("Groq", large_model))
    if small_model and small_model != large_model:
        models_to_test.append(("Groq", small_model))

    if config.OPENAI_API_KEY and config.OPENAI_API_KEY.strip():
        models_to_test.append(("OpenAI-compatible", config.OPENAI_MODEL))

    if not models_to_test:
        print("No configured LLM models available to evaluate. Please set GROQ_API_KEY in .env.")
        return

    print(f"\n=======================================================")
    print(f" Starting AskMyDocs Benchmark ({len(questions)} questions)")
    print(f" Models to evaluate: {[f'{p}/{m}' for p, m in models_to_test]}")
    print(f" Runs per question for latency averaging: {num_runs_per_q}")
    print(f"=======================================================\n")

    csv_rows: list[dict[str, Any]] = []
    summary_stats: dict[str, dict[str, Any]] = {}

    for provider, model_id in models_to_test:
        model_key = f"{provider}:{model_id}"
        summary_stats[model_key] = {
            "total_questions": len(questions),
            "answerable_count": 0,
            "unanswerable_count": 0,
            "hit_at_k_count": 0,
            "keyword_match_count": 0,
            "refusal_correct_count": 0,
            "latencies": [],
        }

    for idx, q_item in enumerate(questions, 1):
        q_id = q_item["id"]
        q_text = q_item["question"]
        doc_filter = q_item.get("document")
        expected_page = q_item.get("expected_page")
        expected_keywords = q_item.get("expected_keywords", [])
        is_answerable = q_item.get("answerable", True)

        # 1. Retrieve context ONCE for all models
        retrieved_docs = retrieve(
            query=q_text,
            top_k=config.TOP_K,
            use_mmr=False,
            doc_filter=doc_filter,
        )

        # Check retrieval Hit@k
        retrieved_pages = [d.metadata.get("page") for d in retrieved_docs]
        hit_at_k = bool(expected_page in retrieved_pages) if is_answerable else False

        context_text = _format_context(retrieved_docs)
        user_prompt = f"Context:\n{context_text}\n\nUser Question:\n{q_text}"
        messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_prompt)]

        print(f"[{idx}/{len(questions)}] Q{q_id}: {q_text[:50]}... (Hit@{config.TOP_K}: {hit_at_k})")

        # 2. Query each model
        for provider, model_id in models_to_test:
            model_key = f"{provider}:{model_id}"
            stats = summary_stats[model_key]

            latencies = []
            final_answer = ""

            for run_idx in range(num_runs_per_q):
                t0 = time.perf_counter()
                try:
                    llm = get_llm(provider=provider, model=model_id, temperature=0.0)
                    resp = llm.invoke(messages)
                    content = resp.content if hasattr(resp, "content") else str(resp)
                    elapsed = round(time.perf_counter() - t0, 3)
                    latencies.append(elapsed)
                    if run_idx == 0:
                        final_answer = str(content).strip()
                except Exception as exc:
                    elapsed = round(time.perf_counter() - t0, 3)
                    latencies.append(elapsed)
                    if run_idx == 0:
                        final_answer = f"ERROR: {exc}"

            avg_latency = round(sum(latencies) / len(latencies), 3) if latencies else 0.0
            stats["latencies"].append(avg_latency)

            # Evaluate Answer Quality
            if is_answerable:
                stats["answerable_count"] += 1
                if hit_at_k:
                    stats["hit_at_k_count"] += 1

                # Check keyword inclusion
                if expected_keywords:
                    found_any = any(kw.lower() in final_answer.lower() for kw in expected_keywords)
                    kw_match = found_any
                else:
                    kw_match = True
                if kw_match:
                    stats["keyword_match_count"] += 1
                refusal_ok = False
            else:
                stats["unanswerable_count"] += 1
                kw_match = False
                refusal_ok = is_refusal(final_answer)
                if refusal_ok:
                    stats["refusal_correct_count"] += 1

            csv_rows.append(
                {
                    "model": model_id,
                    "provider": provider,
                    "question_id": q_id,
                    "question": q_text,
                    "answerable": is_answerable,
                    "expected_page": expected_page if expected_page else "N/A",
                    "retrieved_pages": str(retrieved_pages),
                    "hit_at_k": hit_at_k,
                    "keyword_match": kw_match,
                    "refusal_correct": refusal_ok if not is_answerable else "N/A",
                    "latency_run1": latencies[0] if len(latencies) > 0 else 0,
                    "latency_run2": latencies[1] if len(latencies) > 1 else 0,
                    "latency_run3": latencies[2] if len(latencies) > 2 else 0,
                    "avg_latency": avg_latency,
                    "answer_text": final_answer.replace("\n", " "),
                    "manual_rating_1_to_5": "",  # Empty column for manual user scoring
                }
            )

    # 3. Write results.csv
    fieldnames = [
        "model",
        "provider",
        "question_id",
        "question",
        "answerable",
        "expected_page",
        "retrieved_pages",
        "hit_at_k",
        "keyword_match",
        "refusal_correct",
        "latency_run1",
        "latency_run2",
        "latency_run3",
        "avg_latency",
        "answer_text",
        "manual_rating_1_to_5",
    ]
    with open(RESULTS_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\nSaved detailed evaluation results to: {RESULTS_FILE}\n")

    # 4. Print Summary Table
    print("=" * 88)
    print(f"{'Model':<35} | {'Avg Latency':<12} | {'Hit@k %':<10} | {'Keyword Match %':<16} | {'Refusal %':<10}")
    print("-" * 88)

    for model_key, stats in summary_stats.items():
        ans_count = max(1, stats["answerable_count"])
        unans_count = max(1, stats["unanswerable_count"])
        avg_lat = round(sum(stats["latencies"]) / max(1, len(stats["latencies"])), 2)
        hit_pct = round((stats["hit_at_k_count"] / ans_count) * 100, 1)
        kw_pct = round((stats["keyword_match_count"] / ans_count) * 100, 1)
        ref_pct = round((stats["refusal_correct_count"] / unans_count) * 100, 1)

        print(
            f"{model_key:<35} | {str(avg_lat) + 's':<12} | {str(hit_pct) + '%':<10} | {str(kw_pct) + '%':<16} | {str(ref_pct) + '%'}"
        )
    print("=" * 88)


if __name__ == "__main__":
    run_evaluation(num_runs_per_q=3)
