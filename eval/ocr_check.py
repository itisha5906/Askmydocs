"""
ocr_check.py — Character Error Rate (CER) Benchmark for OCR Pipeline.

Computes Character Error Rate (CER) between OCR-extracted text and a ground-truth
text transcription using Levenshtein edit distance:
    CER = (Substitutions + Deletions + Insertions) / Total_Reference_Characters

Usage:
    python eval/ocr_check.py [scanned_pdf_path] [ground_truth_txt_path]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.logger import get_logger
from core.ocr import check_tesseract, extract_page_text

log = get_logger(__name__)


def levenshtein_distance(ref: str, hyp: str) -> tuple[int, int, int, int]:
    """Calculate Levenshtein distance and edit operations between ref and hyp.

    Returns
    -------
    tuple[int, int, int, int]
        (total_edits, substitutions, deletions, insertions)
    """
    n, m = len(ref), len(hyp)
    if n == 0:
        return m, 0, 0, m
    if m == 0:
        return n, 0, n, 0

    # dp[i][j] stores (edits, subs, dels, ins)
    dp = [[(0, 0, 0, 0)] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = (i, 0, i, 0)
    for j in range(1, m + 1):
        dp[0][j] = (j, 0, 0, j)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                sub_cost, s_s, s_d, s_i = dp[i - 1][j - 1]
                del_cost, d_s, d_d, d_i = dp[i - 1][j]
                ins_cost, i_s, i_d, i_i = dp[i][j - 1]

                # Find operation with minimum total edits
                best = min(
                    (sub_cost + 1, s_s + 1, s_d, s_i),
                    (del_cost + 1, d_s, d_d + 1, d_i),
                    (ins_cost + 1, i_s, i_d, i_i + 1),
                    key=lambda x: x[0],
                )
                dp[i][j] = best

    return dp[n][m]


def compute_cer(reference_text: str, ocr_text: str) -> dict[str, float | int]:
    """Compute Character Error Rate (CER) and metrics."""
    ref_clean = " ".join(reference_text.split())
    ocr_clean = " ".join(ocr_text.split())

    n = len(ref_clean)
    if n == 0:
        return {"cer_percent": 0.0, "edits": 0, "substitutions": 0, "deletions": 0, "insertions": 0, "ref_length": 0}

    total_edits, subs, dels, ins = levenshtein_distance(ref_clean, ocr_clean)
    cer = (total_edits / n) * 100.0

    return {
        "cer_percent": round(cer, 2),
        "edits": total_edits,
        "substitutions": subs,
        "deletions": dels,
        "insertions": ins,
        "ref_length": n,
        "ocr_length": len(ocr_clean),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate OCR Character Error Rate (CER).")
    parser.add_argument("pdf_path", nargs="?", help="Path to scanned PDF")
    parser.add_argument("ground_truth_path", nargs="?", help="Path to ground truth text file")
    args = parser.parse_args()

    sample_dir = Path(__file__).parent.parent / "sample_docs"
    default_pdf = sample_dir / "sample_scanned.pdf"
    default_gt = sample_dir / "sample_ground_truth.txt"

    pdf_file = Path(args.pdf_path) if args.pdf_path else default_pdf
    gt_file = Path(args.ground_truth_path) if args.ground_truth_path else default_gt

    print("\n=======================================================")
    print(" AskMyDocs — OCR Accuracy Check (Character Error Rate)")
    print("=======================================================")

    if not check_tesseract():
        print("[WARNING] Tesseract is not installed on this machine.")
        print("          Please install Tesseract-OCR to run the physical OCR engine.")
        print("          Running synthetic/mock verification...")

    if not gt_file.exists() or not pdf_file.exists():
        print(f"Target files not found ({pdf_file.name} or {gt_file.name}).")
        print("Creating demo ground-truth and sample evaluation...")
        ref_sample = (
            "AskMyDocs is an advanced Retrieval-Augmented Generation system. "
            "It extracts text from documents, performs semantic search using FAISS, "
            "and generates grounded answers using Groq Llama 3 models."
        )
        # Sample simulated OCR output with a few typical OCR character errors (e.g., '1' for 'I', 'rn' for 'm')
        ocr_sample = (
            "AskMyDocs is an advanced Retrieval-Augmented Generation system. "
            "It extracts text frorn documents, performs semantic search using FAISS, "
            "and generates grounded answers using Groq L1ama 3 models."
        )
        res = compute_cer(ref_sample, ocr_sample)
        print(f"\nGround Truth Length: {res['ref_length']} characters")
        print(f"OCR Output Length:   {res['ocr_length']} characters")
        print(f"Total Edits (S+D+I): {res['edits']} (Subs: {res['substitutions']}, Dels: {res['deletions']}, Ins: {res['insertions']})")
        print(f"Character Error Rate (CER): {res['cer_percent']}%\n")
        return

    with open(gt_file, "r", encoding="utf-8") as f:
        ground_truth = f.read()

    # Extract OCR text from page 0
    ocr_result = extract_page_text(str(pdf_file), page_number=0)

    res = compute_cer(ground_truth, ocr_result)
    print(f"\nDocument:            {pdf_file.name}")
    print(f"Reference Length:    {res['ref_length']} characters")
    print(f"OCR Length:          {res['ocr_length']} characters")
    print(f"Total Edits:         {res['edits']}")
    print(f"  - Substitutions:   {res['substitutions']}")
    print(f"  - Deletions:       {res['deletions']}")
    print(f"  - Insertions:      {res['insertions']}")
    print(f"\nCharacter Error Rate (CER): {res['cer_percent']}%")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
