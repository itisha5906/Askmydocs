"""
app.py — Streamlit Presentation Layer for AskMyDocs.

A modern, responsive web application for document QA and summarization.
Contains UI rendering only; all business logic and orchestration are delegated
strictly to core/services.py.
"""

from __future__ import annotations

import streamlit as st

import config
from core import services

# ---------------------------------------------------------------------------
# Page configuration & styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AskMyDocs — LLM Document QA & Summarizer",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished, clean, academic-ready presentation
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        background: linear-gradient(90deg, #1E88E5, #43A047);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-title {
        font-size: 0.95rem;
        color: #666;
        margin-bottom: 1.2rem;
    }
    .author-badge {
        font-size: 0.8rem;
        color: #888;
        border-top: 1px solid #e0e0e0;
        padding-top: 0.4rem;
        margin-top: 0.4rem;
    }
    .badge-ok {
        background-color: #E8F5E9;
        color: #2E7D32;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-warn {
        background-color: #FFF3E0;
        color: #E65100;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 4px 4px 0 0;
        padding: 8px 16px;
    }
    .source-box {
        background-color: #f7f9fa;
        border-left: 3px solid #1E88E5;
        padding: 8px 12px;
        margin: 6px 0;
        border-radius: 0 4px 4px 0;
        font-size: 0.88rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "ingest_status" not in st.session_state:
    st.session_state.ingest_status = {}
if "last_summary" not in st.session_state:
    st.session_state.last_summary = None

# ---------------------------------------------------------------------------
# Sidebar: Configuration & Index Controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Engine Settings")

    # 1. Provider & Model selection
    models_dict = services.get_models()
    providers = list(models_dict.keys())
    selected_provider = st.selectbox("LLM Provider", providers, index=0)

    available_models = models_dict.get(selected_provider, [])
    if available_models:
        selected_model = st.selectbox("Model", available_models, index=0)
    else:
        selected_model = None
        st.warning("No models found for selected provider.")

    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.05, help="0.0 = strict & factual")

    st.markdown("---")
    st.markdown("### 🔍 Retrieval Parameters")
    top_k = st.slider("Top-K Chunks", 1, 15, config.TOP_K, 1)
    use_mmr = st.toggle("Enable MMR (Diverse Chunks)", value=False)
    st.caption(f"📏 Chunk size: {config.CHUNK_SIZE} chars (overlap: {config.CHUNK_OVERLAP})")

    st.markdown("---")
    st.markdown("### 🔌 API Status")
    status = services.provider_status()
    for prov_name, p_stat in status.items():
        badge_cls = "badge-ok" if p_stat.get("configured") else "badge-warn"
        label = "READY" if p_stat.get("configured") else "MISSING KEY"
        st.markdown(
            f"**{prov_name}**: <span class='{badge_cls}'>{label}</span><br>"
            f"<small style='color:gray;'>{p_stat.get('message', '')}</small>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("### 🗄️ Index Operations")
    col_idx1, col_idx2 = st.columns(2)
    with col_idx1:
        if st.button("Clear Index", use_container_width=True, type="secondary"):
            if services.clear_index():
                st.session_state.ingest_status = {}
                st.session_state.messages = []
                st.success("Index wiped clean!")
                st.rerun()
            else:
                st.error("Failed to clear index.")

    with col_idx2:
        if st.button("Refresh", use_container_width=True):
            st.rerun()

    st.markdown("---")
    st.markdown("### 💾 Export Conversation")
    if st.session_state.messages:
        # Build plain text export
        txt_lines = []
        md_lines = ["# AskMyDocs Conversation Export\n"]
        for m in st.session_state.messages:
            role = m["role"].capitalize()
            txt_lines.append(f"{role}: {m['content']}\n")
            md_lines.append(f"### {role}\n{m['content']}\n")
            if m.get("sources"):
                txt_lines.append("  Sources:\n")
                md_lines.append("**Sources:**\n")
                for s in m["sources"]:
                    txt_lines.append(f"  - {s['source']} (Page {s['page']})\n")
                    md_lines.append(f"- *{s['source']}* (Page {s['page']})\n")
            txt_lines.append("\n" + "=" * 40 + "\n\n")
            md_lines.append("\n---\n")

        c_down1, c_down2 = st.columns(2)
        with c_down1:
            st.download_button(
                "📥 .txt",
                "".join(txt_lines),
                file_name="chat_history.txt",
                mime="text/plain",
                use_container_width=True,
            )
        with c_down2:
            st.download_button(
                "📥 .md",
                "".join(md_lines),
                file_name="chat_history.md",
                mime="text/markdown",
                use_container_width=True,
            )
    else:
        st.caption("No chat messages to export yet.")

    st.markdown(
        """
        <div class='author-badge'>
        <b>AskMyDocs</b> · B.Tech CSE Project (2026-27)<br>
        <b>Authors:</b> Itisha Sharma & Kanishka Gupta<br>
        <b>Guide:</b> Dr. Shanu Sharma<br>
        <b>Institution:</b> ABESEC Ghaziabad
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown('<div class="main-title">📚 AskMyDocs</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">LLM-Powered Document Summarization & Semantic Retrieval System with Strict Grounding</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Main Tabs
# ---------------------------------------------------------------------------
tab_chat, tab_summary, tab_docs, tab_logs = st.tabs(
    ["💬 Chat & Search", "📝 Document Summarizer", "📁 Indexed Documents", "📜 System Logs"]
)

# ===========================================================================
# TAB 1: Chat & Search
# ===========================================================================
with tab_chat:
    # 1. Document Upload Section
    with st.expander("📤 Upload & Ingest PDF Documents", expanded=not bool(services.list_documents())):
        uploaded_files = st.file_uploader(
            "Upload one or more PDF files (text-based or scanned images, up to 200 MB)",
            type=["pdf"],
            accept_multiple_files=True,
        )
        if uploaded_files:
            if st.button("🚀 Process & Index Files", type="primary"):
                progress_bar = st.progress(0, text="Starting document ingestion...")
                with st.spinner("Extracting text, running OCR if needed, and generating embeddings..."):
                    results = services.ingest_files(uploaded_files)
                    progress_bar.progress(100, text="Indexing complete!")
                    st.session_state.ingest_status = results

        if st.session_state.ingest_status:
            st.markdown("##### Ingestion Summary")
            for fname, r in st.session_state.ingest_status.items():
                stat = r.get("status", "")
                if stat == "indexed":
                    st.success(
                        f"✅ **{fname}**: {r.get('pages', 0)} pages, {r.get('chars', 0):,} chars indexed "
                        f"({r.get('ocr_pages', 0)} OCR pages used)"
                    )
                elif stat == "already_indexed":
                    st.info(f"ℹ️ **{fname}**: Already indexed with identical SHA-256 hash (skipped re-embedding).")
                else:
                    st.error(f"❌ **{fname}**: {stat}")

    # 2. Document filter & Debug toggle controls
    indexed_doc_list = services.list_documents()
    col_filt, col_dbg = st.columns([3, 1])
    with col_filt:
        filter_options = ["All Documents"] + [d["filename"] for d in indexed_doc_list]
        selected_filter = st.selectbox(
            "Search Scope Filter",
            filter_options,
            index=0,
            help="Restrict answers to a specific uploaded document",
        )
        doc_filter = None if selected_filter == "All Documents" else selected_filter
    with col_dbg:
        st.write("")
        st.write("")
        debug_mode = st.toggle("🛠️ Debug Chunks", value=False)

    st.markdown("---")

    # 3. Chat History Display
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # Display sources collapsible if present
            sources = msg.get("sources", [])
            if sources:
                with st.expander(f"📑 Sources & Citations ({len(sources)} retrieved chunks)"):
                    for i, s in enumerate(sources, 1):
                        ocr_label = " · [OCR]" if s.get("is_ocr") else ""
                        st.markdown(
                            f"**[{i}] {s['source']} — Page {s['page']}**{ocr_label}\n"
                            f"<div class='source-box'>\"{s['snippet']}\"</div>",
                            unsafe_allow_html=True,
                        )
            if msg.get("latency"):
                prov_label = msg.get("provider", "LLM")
                st.caption(f"⚡ Latency: {msg['latency']}s · Engine: {prov_label}")

    # 4. User Chat Input
    user_query = st.chat_input("Ask a question grounded in your uploaded documents...")
    if user_query:
        # Check if vector store has any documents
        if not indexed_doc_list:
            st.warning("⚠️ No documents indexed yet. Please upload and process a PDF above before asking questions.")
        else:
            # Display user message immediately
            with st.chat_message("user"):
                st.markdown(user_query)

            # Record in session state
            st.session_state.messages.append({"role": "user", "content": user_query})

            # Call ask facade
            with st.chat_message("assistant"):
                with st.spinner("Retrieving relevant context and generating grounded answer..."):
                    history_payload = [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages[:-1]
                    ]
                    result = services.ask(
                        question=user_query,
                        history=history_payload,
                        provider=selected_provider,
                        model=selected_model,
                        top_k=top_k,
                        use_mmr=use_mmr,
                        doc_filter=doc_filter,
                    )

                st.markdown(result["answer"])

                sources = result.get("sources", [])
                if sources:
                    with st.expander(f"📑 Sources & Citations ({len(sources)} retrieved chunks)", expanded=False):
                        for i, s in enumerate(sources, 1):
                            ocr_label = " · [OCR]" if s.get("is_ocr") else ""
                            st.markdown(
                                f"**[{i}] {s['source']} — Page {s['page']}**{ocr_label}\n"
                                f"<div class='source-box'>\"{s['snippet']}\"</div>",
                                unsafe_allow_html=True,
                            )

                if debug_mode and sources:
                    with st.expander("🛠️ Raw Retrieved Chunks (Debug Mode)", expanded=True):
                        st.json(sources)

                latency_str = f"⚡ Latency: {result.get('latency', 0.0)}s · Engine: {result.get('provider_used', selected_provider)}"
                st.caption(latency_str)

            # Append assistant response to session state
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": sources,
                    "latency": result.get("latency", 0.0),
                    "provider": result.get("provider_used", selected_provider),
                }
            )

# ===========================================================================
# TAB 2: Document Summarizer
# ===========================================================================
with tab_summary:
    st.markdown("### 📝 Map-Reduce Document Summarization")
    st.caption("Generate comprehensive, structured summaries across large PDFs without exceeding LLM context limits.")

    indexed_docs = services.list_documents()
    if not indexed_docs:
        st.info("ℹ️ No documents indexed yet. Upload documents in the Chat tab first.")
    else:
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            target_doc_options = ["All Documents"] + [d["filename"] for d in indexed_docs]
            chosen_doc = st.selectbox("Select Target Document", target_doc_options, index=0)
        with col_s2:
            summary_style = st.selectbox(
                "Summary Style",
                ["detailed", "short", "bullets"],
                format_func=lambda x: {
                    "detailed": "Detailed (Structured Sections & Methodology)",
                    "short": "Short (2-3 Paragraph Executive Summary)",
                    "bullets": "Bullets (Key Takeaways & Findings)",
                }.get(x, x),
            )

        if st.button("✨ Generate Summary", type="primary"):
            with st.spinner("Executing hierarchical map-reduce summarization across document chunks..."):
                doc_arg = "all" if chosen_doc == "All Documents" else chosen_doc
                summary_result = services.summarize(
                    doc_name_or_all=doc_arg,
                    style=summary_style,
                    provider=selected_provider,
                    model=selected_model,
                )
                st.session_state.last_summary = summary_result

        if st.session_state.last_summary:
            res = st.session_state.last_summary
            st.markdown("---")
            st.markdown("#### Summary Output")
            st.markdown(res.get("summary", ""))

            pages_covered = res.get("pages", [])
            chunks_count = res.get("chunks_processed", 0)
            if pages_covered:
                page_str = ", ".join(str(p) for p in pages_covered)
                st.info(f"📄 **Coverage**: Chunks processed: {chunks_count} | Source pages covered: {page_str}")

# ===========================================================================
# TAB 3: Indexed Documents
# ===========================================================================
with tab_docs:
    st.markdown("### 📁 Currently Indexed Documents")
    documents = services.list_documents()
    if not documents:
        st.info("ℹ️ No documents currently stored in the vector database.")
    else:
        st.write(f"Total documents indexed: **{len(documents)}**")
        table_rows = []
        for d in documents:
            table_rows.append(
                {
                    "Filename": d["filename"],
                    "Pages": d.get("page_count", "N/A"),
                    "Total Chunks": d.get("chunks", 0),
                    "SHA-256 Hash": d.get("file_hash", "")[:12] + "...",
                    "Indexed At": d.get("indexed_at", "")[:19].replace("T", " "),
                }
            )
        st.dataframe(table_rows, use_container_width=True)

# ===========================================================================
# TAB 4: System Logs
# ===========================================================================
with tab_logs:
    st.markdown("### 📜 System Execution Logs")
    col_l1, col_l2 = st.columns([2, 1])
    with col_l1:
        log_level = st.selectbox("Log Level Filter", ["ALL", "INFO", "WARNING", "ERROR", "DEBUG"], index=0)
    with col_l2:
        st.write("")
        st.write("")
        if st.button("🔄 Refresh Logs"):
            st.rerun()

    logs = services.get_logs(level=log_level, max_lines=100)
    if not logs:
        st.caption("No logs recorded matching filter.")
    else:
        st.code("\n".join(logs), language="log")
