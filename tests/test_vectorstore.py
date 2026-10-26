"""
test_vectorstore.py — Mocked tests for core/vectorstore.py.
"""

from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from core.vectorstore import (
    add_documents,
    build_vectorstore,
    clear_vectorstore,
    get_indexed_files,
    is_file_indexed,
    load_vectorstore,
)


class TestVectorStore:
    def test_build_vectorstore_saves_local_and_manifest(self, tmp_path):
        fake_chunks = [
            Document(page_content="Text chunk 1", metadata={"source": "test.pdf", "page": 1, "file_hash": "abc"}),
            Document(page_content="Text chunk 2", metadata={"source": "test.pdf", "page": 2, "file_hash": "abc"}),
        ]
        mock_faiss_store = MagicMock()
        mock_embed = MagicMock()

        with patch("core.vectorstore.FAISS.from_documents", return_value=mock_faiss_store) as mock_from_docs:
            store = build_vectorstore(fake_chunks, embeddings=mock_embed, index_path=tmp_path)

            assert store is mock_faiss_store
            mock_from_docs.assert_called_once_with(fake_chunks, mock_embed)
            mock_faiss_store.save_local.assert_called_once_with(str(tmp_path))

        manifest = get_indexed_files(tmp_path)
        assert "test.pdf" in manifest
        assert manifest["test.pdf"]["chunks"] == 2
        assert manifest["test.pdf"]["file_hash"] == "abc"

    def test_is_file_indexed_returns_correct_boolean(self, tmp_path):
        fake_chunks = [
            Document(page_content="Text chunk", metadata={"source": "doc1.pdf", "page": 1, "file_hash": "hash123"}),
        ]
        with patch("core.vectorstore.FAISS.from_documents", return_value=MagicMock()):
            build_vectorstore(fake_chunks, embeddings=MagicMock(), index_path=tmp_path)

        assert is_file_indexed("doc1.pdf", "hash123", index_path=tmp_path) is True
        assert is_file_indexed("doc1.pdf", "wronghash", index_path=tmp_path) is False
        assert is_file_indexed("nonexistent.pdf", "hash123", index_path=tmp_path) is False

    def test_load_vectorstore_returns_none_if_missing(self, tmp_path):
        store = load_vectorstore(embeddings=MagicMock(), index_path=tmp_path)
        assert store is None

    def test_clear_vectorstore_removes_files(self, tmp_path):
        dummy_file = tmp_path / "dummy.txt"
        dummy_file.write_text("hello")
        assert dummy_file.exists()

        result = clear_vectorstore(index_path=tmp_path)
        assert result is True
        assert not dummy_file.exists()
