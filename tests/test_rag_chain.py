"""
test_rag_chain.py — Mocked tests for core/rag_chain.py.
"""

from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from core.rag_chain import NOT_FOUND_MESSAGE, answer_question


class TestRAGChain:
    def test_answer_question_with_retrieved_docs(self):
        fake_docs = [
            Document(page_content="RAG combines search with generative LLMs.", metadata={"source": "report.pdf", "page": 1}),
        ]
        with patch("core.rag_chain.retrieve", return_value=fake_docs), \
             patch("core.rag_chain.invoke_with_fallback", return_value=("RAG is retrieval augmented generation.", "Groq")):

            res = answer_question(
                query="What is RAG?",
                history=[{"role": "user", "content": "Hi"}],
                provider="Groq",
            )

            assert res["answer"] == "RAG is retrieval augmented generation."
            assert len(res["sources"]) == 1
            assert res["sources"][0]["source"] == "report.pdf"
            assert res["sources"][0]["page"] == 1
            assert "latency" in res
            assert res["provider_used"] == "Groq"

    def test_answer_question_returns_not_found_when_no_docs(self):
        with patch("core.rag_chain.retrieve", return_value=[]):
            res = answer_question(query="Unknown topic")
            assert res["answer"] == NOT_FOUND_MESSAGE
            assert res["sources"] == []
            assert "latency" in res

    def test_history_passed_to_invoke(self):
        fake_docs = [Document(page_content="Content", metadata={"source": "a.pdf", "page": 2})]
        with patch("core.rag_chain.retrieve", return_value=fake_docs), \
             patch("core.rag_chain.condense_question", return_value="Follow up?"), \
             patch("core.rag_chain.invoke_with_fallback", return_value=("Answer", "Groq")) as mock_invoke:

            history = [
                {"role": "user", "content": "Question 1"},
                {"role": "assistant", "content": "Answer 1"},
            ]
            answer_question(query="Follow up?", history=history)

            mock_invoke.assert_called_once()
            call_messages = mock_invoke.call_args[1].get("messages")
            assert len(call_messages) == 4

    def test_condense_question_rewrites_with_history(self):
        from core.rag_chain import condense_question
        with patch("core.rag_chain.invoke_with_fallback", return_value=("What is machine learning?", "Groq")):
            res = condense_question(
                query="How does it work?",
                history=[{"role": "user", "content": "Tell me about machine learning."}],
            )
            assert res == "What is machine learning?"

    def test_condense_question_returns_query_when_history_empty(self):
        from core.rag_chain import condense_question
        res = condense_question("What is AI?", history=[])
        assert res == "What is AI?"
