"""
test_summarizer.py — Mocked tests for core/summarizer.py.
"""

from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from core.summarizer import summarize_documents


class TestSummarizer:
    def test_empty_chunks_returns_default_message(self):
        res = summarize_documents([])
        assert "No document content" in res["summary"]
        assert res["pages"] == []

    def test_small_chunk_count_uses_direct_reduce(self):
        chunks = [
            Document(page_content="Introduction to AI.", metadata={"page": 1}),
            Document(page_content="Deep learning basics.", metadata={"page": 2}),
        ]
        with patch("core.summarizer.invoke_with_fallback", return_value=("Executive Summary.", "Groq")) as mock_invoke:
            res = summarize_documents(chunks, style="short")

            assert res["summary"] == "Executive Summary."
            assert res["pages"] == [1, 2]
            assert res["chunks_processed"] == 2
            mock_invoke.assert_called_once()

    def test_map_reduce_batches_for_many_chunks(self):
        chunks = [
            Document(page_content=f"Chunk paragraph {i}.", metadata={"page": i})
            for i in range(1, 10)
        ]
        with patch("core.summarizer.invoke_with_fallback") as mock_invoke:
            # 9 chunks with batch size 4 -> 3 map calls + 1 reduce call = 4 calls total
            mock_invoke.side_effect = [
                ("Batch 1 summary", "Groq"),
                ("Batch 2 summary", "Groq"),
                ("Batch 3 summary", "Groq"),
                ("Final unified summary", "Groq"),
            ]

            res = summarize_documents(chunks, style="bullets")

            assert res["summary"] == "Final unified summary"
            assert len(res["pages"]) == 9
            assert mock_invoke.call_count == 4
