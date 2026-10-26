# AskMyDocs — Quickstart Guide

Get AskMyDocs up and running in under 3 minutes.

---

### Step 1: Install Dependencies
Ensure Python 3.10+ is installed:
```bash
python --version
```
Install required packages:
```bash
pip install -r requirements.txt
```

---

### Step 2: Configure Keys
Copy the template and supply your free Groq API key:
```bash
cp .env.example .env
```
Inside `.env`:
```ini
GROQ_API_KEY=gsk_your_key_here
```

---

### Step 3: Run the Web App
```bash
streamlit run app.py
```
Visit **`http://localhost:8501`** in your browser.

---

### Step 4: Verify with Sample PDF
1. Go to the **💬 Chat & Search** tab.
2. In the upload drawer, upload `sample_docs/sample_text.pdf`.
3. Click **🚀 Process & Index Files**.
4. Ask: *"What are the core evaluation metrics of AskMyDocs?"*
5. Inspect the grounded answer and collapsible **📑 Sources & Citations**.
