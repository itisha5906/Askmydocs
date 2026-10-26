"""
test_services.py — Mocked tests for core/services.py facade.
"""

from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from core.services import (
    ask,
    clear_index,
    get_models,
    ingest_files,
    list_documents,
    provider_status,
    summarize,
)


class TestServices:
    def test_ingest_files_with_mocked_pipeline(self, tmp_path):
        dummy_pdf = tmp_path / "paper.pdf"
        dummy_pdf.write_bytes(b"%PDF-1.4 dummy content")

        fake_docs = [
            Document(page_content="Clean page content", metadata={"source": "paper.pdf", "page": 1, "ocr": False})
        ]

        with patch("core.services.load_pdf", return_value=fake_docs), \
             patch("core.services.chunk_documents", return_value=fake_docs), \
             patch("core.services.add_documents") as mock_add_docs, \
             patch("core.services.is_file_indexed", return_value=False):

            result = ingest_files([dummy_pdf])

            assert "paper.pdf" in result
            assert result["paper.pdf"]["status"] == "indexed"
            assert result["paper.pdf"]["pages"] == 1
            mock_add_docs.assert_called_once()

    def test_ask_facade_delegates_to_rag_chain(self):
        with patch("core.services.answer_question") as mock_ans:
            mock_ans.return_value = {"answer": "42", "sources": [], "latency": 0.5}

            res = ask("What is the answer?", top_k=5)
            assert res["answer"] == "42"
            mock_ans.assert_called_once_with(
                query="What is the answer?",
                history=None,
                provider="Groq",
                model=None,
                top_k=5,
                use_mmr=False,
                doc_filter=None,
            )

    def test_summarize_facade_with_docstore(self):
        mock_store = MagicMock()
        mock_doc = Document(page_content="Text", metadata={"source": "docA.pdf", "page": 1})
        mock_store.docstore._dict = {"id1": mock_doc}

        with patch("core.services.load_vectorstore", return_value=mock_store), \
             patch("core.services.summarize_documents") as mock_sum:
            mock_sum.return_value = {"summary": "Short sum", "pages": [1]}

            res = summarize(doc_name_or_all="docA.pdf", style="short")
            assert res["summary"] == "Short sum"
            mock_sum.assert_called_once()

    def test_list_and_clear_documents_facade(self):
        with patch("core.services.get_indexed_files", return_value={"doc1.pdf": {"chunks": 5, "page_count": 2}}):
            docs = list_documents()
            assert len(docs) == 1
            assert docs[0]["filename"] == "doc1.pdf"
            assert docs[0]["chunks"] == 5

        with patch("core.services.clear_vectorstore", return_value=True):
            assert clear_index() is True

    def test_get_models_and_provider_status_facade(self):
        with patch("core.services.get_groq_models", return_value=["llama-3.1-8b-instant"]), \
             patch("core.services.check_provider_status", return_value={"Groq": {"configured": True}}):

            models = get_models()
            assert "llama-3.1-8b-instant" in models["Groq"]

            status = provider_status()
            assert status["Groq"]["configured"] is True
