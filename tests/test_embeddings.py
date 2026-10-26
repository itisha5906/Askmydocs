"""
test_embeddings.py — Mocked tests for core/embeddings.py.
"""

from unittest.mock import MagicMock, patch

from core.embeddings import get_embeddings, reset_embeddings_cache


class TestEmbeddings:
    def setup_method(self):
        reset_embeddings_cache()

    def teardown_method(self):
        reset_embeddings_cache()

    def test_get_embeddings_returns_cached_instance(self):
        with patch("langchain_huggingface.HuggingFaceEmbeddings") as mock_hf:
            mock_inst = MagicMock()
            mock_hf.return_value = mock_inst

            emb1 = get_embeddings()
            emb2 = get_embeddings()

            assert emb1 is emb2
            mock_hf.assert_called_once()

    def test_get_embeddings_normalizes_embeddings(self):
        with patch("langchain_huggingface.HuggingFaceEmbeddings") as mock_hf:
            get_embeddings("custom/model")
            _, kwargs = mock_hf.call_args
            assert kwargs.get("encode_kwargs") == {"normalize_embeddings": True}
            assert kwargs.get("model_name") == "custom/model"

    def test_reset_cache_forces_recreation(self):
        with patch("langchain_huggingface.HuggingFaceEmbeddings") as mock_hf:
            mock_hf.side_effect = [MagicMock(), MagicMock()]
            emb1 = get_embeddings()
            reset_embeddings_cache()
            emb2 = get_embeddings()
            assert emb1 is not emb2
            assert mock_hf.call_count == 2
