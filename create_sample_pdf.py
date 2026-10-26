"""
create_sample_pdf.py — Generates sample_docs/sample_text.pdf for testing.

Run once:  python create_sample_pdf.py
Requires:  reportlab (already in requirements.txt)
"""

from __future__ import annotations

import sys
from pathlib import Path

SAMPLE_DOCS_DIR = Path(__file__).parent / "sample_docs"
SAMPLE_DOCS_DIR.mkdir(exist_ok=True)


def create_sample_text_pdf() -> None:
    """Create a 5-page text-based PDF about Retrieval-Augmented Generation."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.units import inch
    except ImportError:
        print("ERROR: reportlab not installed. Run: pip install reportlab")
        sys.exit(1)

    output_path = SAMPLE_DOCS_DIR / "sample_text.pdf"

    doc = SimpleDocTemplate(str(output_path), pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    pages_content = [
        # Page 1
        {
            "title": "AskMyDocs: LLM-Powered Document Summarization & Semantic Information Retrieval",
            "body": (
                "Authors: Itisha Sharma and Kanishka Gupta\n"
                "Guide: Dr. Shanu Sharma\n"
                "Institution: ABESEC Ghaziabad | Session: 2026-27\n\n"
                "Abstract\n"
                "AskMyDocs is a Retrieval-Augmented Generation (RAG) web application "
                "that allows users to upload PDF documents and query them using natural language. "
                "The system combines semantic search via FAISS vector stores with large language "
                "models (LLMs) hosted on Groq to produce accurate, grounded answers with source citations. "
                "The application supports both text-based and scanned PDF documents through an integrated "
                "OCR pipeline using PyMuPDF and Tesseract."
            ),
        },
        # Page 2
        {
            "title": "Chapter 1: Introduction to Retrieval-Augmented Generation",
            "body": (
                "Retrieval-Augmented Generation (RAG) is a hybrid AI approach that combines "
                "the strengths of information retrieval systems with generative language models. "
                "Unlike pure generative models, which rely solely on learned parametric knowledge, "
                "RAG systems augment generation with dynamically retrieved context from external knowledge bases.\n\n"
                "Key Benefits of RAG:\n"
                "1. Grounded responses — answers are based on actual documents, reducing hallucination.\n"
                "2. Updatable knowledge — documents can be swapped without retraining the model.\n"
                "3. Source attribution — users can verify the provenance of each answer.\n"
                "4. Domain adaptability — works across legal, medical, technical, and academic domains.\n\n"
                "The RAG pipeline in AskMyDocs consists of nine stages: document upload, text extraction "
                "or OCR, preprocessing and chunking, embedding generation, FAISS vector store indexing, "
                "query embedding, semantic retrieval, LLM generation, and structured answer output."
            ),
        },
        # Page 3
        {
            "title": "Chapter 2: System Architecture",
            "body": (
                "AskMyDocs follows a modular five-layer architecture:\n\n"
                "Layer 1 — Presentation: A Streamlit web interface with tabs for Chat, Summary, "
                "Documents, and Logs. Users upload PDFs, configure retrieval parameters, and interact "
                "with the system through a clean, responsive UI.\n\n"
                "Layer 2 — Processing: PDFs are loaded using PyPDF and PDFPlumber. Pages with fewer "
                "than 50 extractable characters are automatically routed to the OCR module, which uses "
                "PyMuPDF to render pages at 300 DPI before passing them to Tesseract.\n\n"
                "Layer 3 — Embedding and Retrieval: The all-mpnet-base-v2 model from Hugging Face "
                "generates 768-dimensional embeddings. These are stored in a FAISS flat index persisted "
                "to disk. At query time, the top-k most similar chunks are retrieved using cosine "
                "similarity or Maximum Marginal Relevance (MMR).\n\n"
                "Layer 4 — Generation: LangChain orchestrates a ConversationalRetrievalChain that "
                "condenses follow-up questions into standalone queries, retrieves context, and prompts "
                "the selected LLM. Groq's Llama 3.x models serve as the primary provider.\n\n"
                "Layer 5 — Output: Each answer is displayed with collapsible source expanders showing "
                "the filename, page number, and a short text snippet for verification."
            ),
        },
        # Page 4
        {
            "title": "Chapter 3: Technology Stack",
            "body": (
                "The AskMyDocs system is built on the following technologies:\n\n"
                "- Python 3.11+: Core programming language.\n"
                "- Streamlit 1.44: Web framework for the interactive user interface.\n"
                "- LangChain 0.3.x: RAG orchestration, chain management, and memory.\n"
                "- FAISS-CPU 1.9: High-performance vector similarity search.\n"
                "- Sentence Transformers (all-mpnet-base-v2): Local embedding model, no API key required.\n"
                "- Groq API: Cloud LLM inference using Llama 3.3 70B Versatile and Llama 3.1 8B Instant.\n"
                "- PyPDF and PDFPlumber: Text extraction from text-layer PDFs.\n"
                "- PyMuPDF (fitz): Page rendering for the OCR pipeline.\n"
                "- pytesseract + Tesseract: OCR engine for scanned documents.\n"
                "- python-dotenv: Secure loading of API keys from .env files.\n\n"
                "All LLM inference runs through cloud APIs, making the application suitable for "
                "standard laptops without GPUs. Embedding generation runs locally via CPU, "
                "cached using Streamlit's st.cache_resource decorator."
            ),
        },
        # Page 5
        {
            "title": "Chapter 4: Evaluation and Future Work",
            "body": (
                "Evaluation Methodology:\n"
                "The system is evaluated across four dimensions:\n"
                "1. Retrieval Accuracy (Hit@k): Whether the expected source page appears in the top-k results.\n"
                "2. Summary Quality: Human evaluation of short, detailed, and bullet-point summaries.\n"
                "3. OCR Accuracy: Character Error Rate (CER) computed against ground-truth transcriptions.\n"
                "4. Response Latency: Time from query submission to full response across different models.\n\n"
                "Preliminary Results:\n"
                "- Llama 3.3 70B Versatile achieves the highest answer quality with average latency of 2.3s.\n"
                "- Llama 3.1 8B Instant is 3x faster with slightly lower verbosity.\n"
                "- FAISS retrieval with all-mpnet-base-v2 achieves Hit@6 of 0.91 on the test set.\n\n"
                "Future Scope:\n"
                "- Support for DOCX, TXT, and Excel document formats.\n"
                "- Multilingual document queries using multilingual embedding models.\n"
                "- Voice interaction through speech-to-text (Whisper API).\n"
                "- Cloud deployment on AWS/GCP with Streamlit Community Cloud.\n"
                "- Fine-tuned retrieval using domain-specific embedding models."
            ),
        },
    ]

    for page_data in pages_content:
        story.append(Paragraph(page_data["title"], styles["Heading1"]))
        story.append(Spacer(1, 0.2 * inch))
        # Split body into paragraphs at double newlines
        for para in page_data["body"].split("\n\n"):
            story.append(Paragraph(para.replace("\n", "<br/>"), styles["BodyText"]))
            story.append(Spacer(1, 0.1 * inch))
        story.append(Spacer(1, 0.5 * inch))

    doc.build(story)
    print(f"Created: {output_path}")


if __name__ == "__main__":
    create_sample_text_pdf()
    print("Sample PDF created in sample_docs/")
