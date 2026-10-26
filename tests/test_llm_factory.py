"""
test_llm_factory.py — Mocked tests for core/llm_factory.py.
"""

from unittest.mock import MagicMock, patch

from core.llm_factory import (
    check_provider_status,
    get_groq_models,
    get_llm,
    invoke_with_fallback,
)


class TestLLMFactory:
    def test_get_groq_models_live_fetch(self):
        fake_m1 = MagicMock(id="llama-3.3-70b-versatile", active=True)
        fake_m2 = MagicMock(id="whisper-large-v3", active=True)
        fake_response = MagicMock(data=[fake_m1, fake_m2])

        with patch("config.GROQ_API_KEY", "gsk_fake"), \
             patch("groq.Groq") as mock_groq_cls:
            mock_client = MagicMock()
            mock_client.models.list.return_value = fake_response
            mock_groq_cls.return_value = mock_client

            models = get_groq_models(force_refresh=True)

            assert "llama-3.3-70b-versatile" in models
            assert "whisper-large-v3" not in models  # audio model excluded

    def test_get_groq_models_fallback_on_failure(self):
        with patch("config.GROQ_API_KEY", "gsk_fake"), \
             patch("groq.Groq", side_effect=Exception("API offline")):
            models = get_groq_models(force_refresh=True)
            assert len(models) > 0
            assert "llama-3.3-70b-versatile" in models

    def test_get_llm_creates_chatgroq_and_chatopenai(self):
        with patch("langchain_groq.ChatGroq") as mock_groq:
            get_llm("Groq", model="llama-3.1-8b-instant")
            mock_groq.assert_called_once()

        with patch("langchain_openai.ChatOpenAI") as mock_openai:
            get_llm("OpenAI-compatible", model="custom-model")
            mock_openai.assert_called_once()

    def test_invoke_with_fallback_uses_secondary_on_primary_error(self):
        mock_primary_llm = MagicMock()
        mock_primary_llm.invoke.side_effect = RuntimeError("Primary rate limited")

        mock_fallback_llm = MagicMock()
        mock_fallback_llm.invoke.return_value = MagicMock(content="Fallback answer")

        with patch("core.llm_factory.get_llm") as mock_get_llm:
            mock_get_llm.side_effect = [mock_primary_llm, mock_fallback_llm]

            content, provider = invoke_with_fallback(
                messages="Hello",
                primary_provider="Groq",
                secondary_provider="OpenAI-compatible",
            )

            assert content == "Fallback answer"
            assert "Fallback" in provider

    def test_check_provider_status(self):
        with patch("config.GROQ_API_KEY", "key123"), patch("config.OPENAI_API_KEY", ""):
            status = check_provider_status()
            assert status["Groq"]["configured"] is True
            assert status["OpenAI-compatible"]["configured"] is False
