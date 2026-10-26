"""
llm_factory.py — Multi-LLM provider factory and dynamic model loader.

Supports Groq (Llama 3.x, etc.) and any OpenAI-compatible API endpoint.
Queries the live Groq API for currently available models to prevent using
deprecated/decommissioned endpoints, and provides seamless fallback switching.

Public API
----------
    get_groq_models() -> list[str]
    get_llm(provider, model, temperature, max_tokens) -> BaseChatModel
    invoke_with_fallback(messages, primary_provider, primary_model, ...) -> tuple[str, str]
    check_provider_status() -> dict[str, dict]
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

import config
from core.logger import get_logger

log = get_logger(__name__)

_cached_groq_models: list[str] | None = None


def get_groq_models(force_refresh: bool = False) -> list[str]:
    """Fetch active chat models from Groq API or return fallback list.

    Returns
    -------
    list[str]
        List of available model IDs.
    """
    global _cached_groq_models  # noqa: PLW0603

    if _cached_groq_models is not None and not force_refresh:
        return _cached_groq_models

    if not config.GROQ_API_KEY:
        log.warning("GROQ_API_KEY is not configured; using fallback model list.")
        return list(config.GROQ_FALLBACK_MODELS)

    try:
        from groq import Groq

        client = Groq(api_key=config.GROQ_API_KEY)
        response = client.models.list()

        # Filter for text/chat models, excluding audio/whisper models
        chat_models: list[str] = []
        for m in response.data:
            model_id = getattr(m, "id", "")
            active = getattr(m, "active", True)
            if active and not ("whisper" in model_id.lower() or "guard" in model_id.lower()):
                chat_models.append(model_id)

        # Sort so llama models are easily found first
        chat_models.sort(key=lambda name: (0 if "llama-3" in name else 1, name))

        if chat_models:
            _cached_groq_models = chat_models
            log.info("Loaded %d live models from Groq API.", len(chat_models))
            return chat_models

    except Exception as exc:  # noqa: BLE001
        log.warning("Failed to fetch live Groq models (%s); using fallback list.", exc)

    return list(config.GROQ_FALLBACK_MODELS)


def get_llm(
    provider: str = "Groq",
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    streaming: bool = False,
) -> BaseChatModel:
    """Instantiate a ChatModel for the given provider.

    Parameters
    ----------
    provider:
        'Groq' or 'OpenAI-compatible' (case-insensitive).
    model:
        Model name/identifier. If None, uses provider default.
    temperature:
        Sampling temperature (0.0 for deterministic RAG answers).
    max_tokens:
        Maximum generation tokens.
    streaming:
        Whether to enable token streaming.

    Returns
    -------
    BaseChatModel
    """
    prov = (provider or "Groq").strip().lower()

    if prov == "groq":
        try:
            from langchain_groq import ChatGroq
        except ImportError as exc:
            raise ImportError(
                "langchain-groq is not installed. Run: pip install langchain-groq"
            ) from exc

        target_model = model or (get_groq_models() or config.GROQ_FALLBACK_MODELS)[0]
        return ChatGroq(
            model=target_model,
            groq_api_key=config.GROQ_API_KEY,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=streaming,
        )

    if prov in ("openai", "openai-compatible"):
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise ImportError(
                "langchain-openai is not installed. Run: pip install langchain-openai"
            ) from exc

        target_model = model or config.OPENAI_MODEL
        kwargs: dict[str, Any] = {
            "model": target_model,
            "api_key": config.OPENAI_API_KEY or "dummy-key",
            "temperature": temperature,
            "max_tokens": max_tokens,
            "streaming": streaming,
        }
        if config.OPENAI_BASE_URL:
            kwargs["base_url"] = config.OPENAI_BASE_URL

        return ChatOpenAI(**kwargs)

    raise ValueError(f"Unsupported LLM provider: '{provider}'. Supported: Groq, OpenAI-compatible.")


def invoke_with_fallback(
    messages: Any,
    primary_provider: str = "Groq",
    primary_model: str | None = None,
    secondary_provider: str = "OpenAI-compatible",
    secondary_model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
) -> tuple[str, str]:
    """Invoke the primary LLM, automatically falling back to secondary on failure.

    Returns
    -------
    tuple[str, str]
        (content_string, provider_actually_used)
    """
    try:
        llm = get_llm(
            provider=primary_provider,
            model=primary_model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        response = llm.invoke(messages)
        content = response.content if hasattr(response, "content") else str(response)
        return str(content), primary_provider

    except Exception as exc:  # noqa: BLE001
        log.warning(
            "Primary LLM (%s/%s) failed: %s. Attempting fallback to %s...",
            primary_provider,
            primary_model,
            exc,
            secondary_provider,
        )

        try:
            fallback_llm = get_llm(
                provider=secondary_provider,
                model=secondary_model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            response = fallback_llm.invoke(messages)
            content = response.content if hasattr(response, "content") else str(response)
            return str(content), f"{secondary_provider} (Fallback)"
        except Exception as fallback_exc:
            log.error("Fallback LLM (%s) also failed: %s", secondary_provider, fallback_exc)
            raise RuntimeError(
                f"Both primary ({primary_provider}) and fallback ({secondary_provider}) LLM calls failed: {fallback_exc}"
            ) from fallback_exc


def check_provider_status() -> dict[str, dict]:
    """Check whether providers are configured with API keys."""
    groq_configured = bool(config.GROQ_API_KEY and config.GROQ_API_KEY.strip())
    openai_configured = bool(config.OPENAI_API_KEY and config.OPENAI_API_KEY.strip())

    return {
        "Groq": {
            "configured": groq_configured,
            "message": "API key present" if groq_configured else "GROQ_API_KEY missing in .env",
        },
        "OpenAI-compatible": {
            "configured": openai_configured,
            "message": (
                f"API key present ({config.OPENAI_BASE_URL})"
                if openai_configured
                else "OPENAI_API_KEY missing in .env"
            ),
        },
    }
