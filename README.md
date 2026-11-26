# AskMyDocs: LLM-Powered Document Summarization & Semantic Information Retrieval

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/frontend-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/orchestration-LangChain-1C3C3C.svg)](https://www.langchain.com/)
[![FAISS](https://img.shields.io/badge/vectorstore-FAISS-008080.svg)](https://github.com/facebookresearch/faiss)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**AskMyDocs** is a Retrieval-Augmented Generation (RAG) web application developed for B.Tech Computer Science & Engineering (Session 2026–27). It enables natural language chat with multiple uploaded PDF documents (both text-based and scanned images) with strict grounding, precise page/snippet citations, and hierarchical map-reduce summarization.

---

## 👥 Project Team

- **Authors:** Itisha Sharma & Kanishka Gupta
- **Project Guide:** Dr. Shanu Sharma
- **Department:** Department of Computer Science & Engineering
- **Institution:** ABESEC Ghaziabad (Session 2026–27)

---

## 🌟 Key Features

1. **Dual PDF Extraction Pipeline:**
   - **Text-based PDFs:** Fast text extraction via `PyPDF` with automatic fallback to `PDFPlumber` when character garbling is detected.
   - **Scanned / Image PDFs:** Optical Character Recognition (OCR) path rendering high-DPI pages via `PyMuPDF` (fitz) and extracting text via `pytesseract` (graceful fallback if Tesseract binary is absent).
2. **Context-Preserving Chunking & Deduplication:**
   - `RecursiveCharacterTextSplitter` with 800-character chunks and 150-character overlap.
   - Rich chunk metadata tracking `source`, `page`, `chunk_id`, and `is_ocr` flags.
   - SHA-256 document hashing avoids re-embedding unchanged documents.
3. **Local Vector Search:**
   - Embeddings generated locally via `sentence-transformers/all-mpnet-base-v2` (normalized for cosine distance; runs CPU-only without API keys).
   - `faiss-cpu` vector store persisted to disk with incremental merge and reset controls.
   - Retrieval with top-$k$ similarity and optional Maximal Marginal Relevance (MMR) for diversity.
4. **Multi-LLM Generation Layer with Grounding:**
   - **Groq API:** Ultra-fast Llama 3.x inference (`llama-3.3-70b-versatile`, `llama-3.1-8b-instant`). Fetches active models live from Groq API.
   - **OpenAI-Compatible API:** Configurable `OPENAI_BASE_URL` and `OPENAI_API_KEY` for flexible fallback.
   - **Automatic Fallback:** Retries and switches providers seamlessly on rate limits or API errors.
   - **Strict Grounding:** Strict system prompt prevents hallucinations; answers ONLY from retrieved context with page citations.
5. **Hierarchical Map-Reduce Summarization:**
   - Summarizes long documents across multiple chunks without context window truncation.
   - Supports 3 distinct styles: **Short** (executive summary), **Detailed** (structured technical sections), and **Bullets** (key takeaways).
6. **Streamlit UI:**
   - Tabs for Chat, Summarizer, Document Index inspection, and live System Logs.
   - Conversation export to `.txt` and `.md`.

---

## 🏗️ System Architecture

```mermaid
graph TD
    A[User PDF Upload] --> B{Text Extractable?}
    B -->|Yes| C[PyPDF / PDFPlumber]
    B -->|No / Scanned| D[PyMuPDF + Tesseract OCR]
    C --> E[Text Preprocessing & Header Cleaning]
    D --> E
    E --> F[Recursive Chunker with Page Metadata]
    F --> G[Sentence-Transformers all-mpnet-base-v2]
    G --> H[(FAISS Vector Store Disk)]
    I[User Query] --> J[Top-K / MMR Retriever]
    H --> J
    J --> K[Grounded Prompt + Context + Memory]
    K --> L{LLM Provider}
    L -->|Primary| M[Groq API Llama 3.x]
    L -->|Fallback| N[OpenAI-Compatible API]
    M --> O[Answer + Citations + Latency]
    N --> O
```

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.10 or higher
- Git
- **(Optional for OCR)** [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) for scanned PDFs:
  - **Windows**: Download installer from [UB-Mannheim](https://github.com/UB-Mannheim/tesseract/wiki) or run `winget install UB-Mannheim.TesseractOCR`
  - **Linux (Ubuntu/Debian)**: `sudo apt-get update && sudo apt-get install -y tesseract-ocr`
  - **macOS**: `brew install tesseract`

### 2. Clone the Repository
```bash
git clone https://github.com/itisha5906/Askmydocs.git
cd Askmydocs
```

### 3. Setup Virtual Environment & Install Dependencies
On Windows:
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

On Linux/macOS:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 4. Configure Environment
Copy `.env.example` to `.env` and fill in your API keys:
```bash
cp .env.example .env
```
Edit `.env`:
```env
GROQ_API_KEY=your_groq_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
```

### 5. Run the Application
On Windows, you can simply double-click `run_app.bat` or run:
```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

---

## 🧪 Testing & Evaluation

### Automated Unit Tests
Run all 64 unit tests with pytest:
```bash
pytest tests/ -v
```
All **64 unit tests** run with mocks and do not require external API tokens or GPU acceleration.

### End-to-End Model Evaluation
Run the automated benchmark across configured models on the 26 evaluation questions:
```bash
python eval/evaluate.py
```
Outputs `eval/results.csv` tracking retrieval Hit@k, keyword inclusion, refusal accuracy on unanswerable questions, and average latency across models.

### OCR Accuracy Evaluation (CER)
Benchmark Character Error Rate on scanned documents using Levenshtein distance:
```bash
python eval/ocr_check.py sample_docs/sample_text.pdf sample_docs/sample_ground_truth.txt
```

---

## 📂 Sample Documents Guidelines

Recommended PDF documents to test with and place in `sample_docs/`:
1. **Academic / Research Papers**: arXiv research paper (e.g., Attention Is All You Need) for multi-column text extraction and complex citations.
2. **Class Notes / Lecture Slides**: PDF slides or notes for testing section headings, bulleted concepts, and summaries.
3. **Scanned / Image Document**: A scanned receipt, certificate, or handwritten/printed page to exercise the Tesseract OCR fallback engine.
4. **Business Report / Technical Manual**: Multi-page report with repeated running headers and footers to test header deduplication and map-reduce summarization.

---

## 📂 Project Structure

```
askmydocs/
├── app.py                  # Streamlit Presentation Layer (Tabbed UI, Compare mode, Logs)
├── config.py               # Central configuration & secrets loader
├── requirements.txt        # Pinned dependencies
├── .env.example            # Environment variable template
├── .gitignore              # Secrets & FAISS index exclusion
├── README.md               # Project documentation
├── QUICKSTART.md           # Fast setup guide
├── Dockerfile              # Container deployment recipe
├── run_app.bat             # 1-click launcher for Windows
├── core/                   # Processing, Retrieval, and LLM services
│   ├── loader.py           # Dual PDF extraction (PyPDF / PDFPlumber)
│   ├── ocr.py              # OCR path for image pages (PyMuPDF + Tesseract)
│   ├── preprocess.py       # Text cleaning & header/footer removal
│   ├── chunker.py          # Recursive text splitter with metadata
│   ├── embeddings.py       # Local HuggingFace all-mpnet-base-v2
│   ├── vectorstore.py      # FAISS build, persistence, and merge
│   ├── retriever.py        # Top-K & MMR retriever with scores & doc filtering
│   ├── llm_factory.py      # Groq & OpenAI provider with retry & fallback
│   ├── rag_chain.py        # Grounded QA chain with query condensation & citations
│   ├── summarizer.py       # Map-reduce document summarizer (brief/detailed/bullets)
│   ├── services.py         # Unified facade consumed by UI (includes compare mode)
│   └── logger.py           # File + Streamlit log buffer
├── eval/                   # Benchmark and evaluation suite
│   ├── questions.json      # 26 benchmark QA pairs with unanswerable traps
│   ├── evaluate.py         # Latency, Hit@k, and keyword-inclusion evaluator
│   └── ocr_check.py        # CER Levenshtein distance calculator
├── tests/                  # Pytest test suites (64 unit tests)
└── sample_docs/            # Sample documents for smoke tests
```

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).
