"""
test_retriever.py — Mocked tests for core/retriever.py.
"""

from unittest.mock import MagicMock

import pytest

from core.retriever import get_retriever, retrieve


class TestRetriever:
    def test_get_retriever_similarity_config(self):
        mock_store = MagicMock()
        mock_store.as_retriever.return_value = MagicMock()

        get_retriever(vectorstore=mock_store, top_k=5, use_mmr=False, doc_filter=None)

        mock_store.as_retriever.assert_called_once_with(
            search_type="similarity",
            search_kwargs={"k": 5},
        )

    def test_get_retriever_mmr_and_doc_filter(self):
        mock_store = MagicMock()
        mock_store.as_retriever.return_value = MagicMock()

        get_retriever(vectorstore=mock_store, top_k=4, use_mmr=True, doc_filter="docA.pdf")

        mock_store.as_retriever.assert_called_once()
        _, kwargs = mock_store.as_retriever.call_args
        assert kwargs["search_type"] == "mmr"
        assert kwargs["search_kwargs"]["k"] == 4
        assert kwargs["search_kwargs"]["fetch_k"] == 12  # 4 * MMR_FETCH_K_MULTIPLIER (3)
        assert kwargs["search_kwargs"]["filter"] == {"source": "docA.pdf"}

    def test_retrieve_invokes_retriever(self):
        mock_store = MagicMock()
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = ["doc1", "doc2"]
        mock_store.as_retriever.return_value = mock_retriever

        results = retrieve("test query", vectorstore=mock_store, top_k=3)
        assert results == ["doc1", "doc2"]
        mock_retriever.invoke.assert_called_once_with("test query")
