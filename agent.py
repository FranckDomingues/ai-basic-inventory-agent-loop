"""Configuration and client initialization for the inventory AI agent."""

import os
from dataclasses import dataclass
from typing import Mapping

from openai import OpenAI


_GROQ_BASE_URL = "https://api.groq.com/openai/v1"
_OLLAMA_BASE_URL = "http://localhost:11434/v1"


@dataclass(frozen=True)
class LLMConfig:
    """Resolved settings for an OpenAI-compatible language model service.

    Attributes:
        provider: Selected service, either ``groq`` or ``ollama``.
        model: Model identifier sent to the service.
        base_url: OpenAI-compatible API root.
        api_key: Configured provider credential; Ollama uses a local placeholder.
    """

    provider: str
    model: str
    base_url: str
    api_key: str | None

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "LLMConfig":
        """Build configuration from environment-style values.

        Args:
            environ: Optional mapping for tests or explicit configuration. When
                omitted, the current process environment is used.

        Returns:
            Resolved provider, model, endpoint, and available API key.

        Raises:
            ValueError: If ``LLM_PROVIDER`` selects an unsupported provider.
        """
        values = os.environ if environ is None else environ
        provider = values.get("LLM_PROVIDER", "groq").strip().lower()

        if provider == "groq":
            default_model = "llama-3.3-70b-versatile"
            default_base_url = _GROQ_BASE_URL
            api_key = values.get("GROQ_API_KEY")
        elif provider == "ollama":
            default_model = "qwen2.5:7b"
            default_base_url = _OLLAMA_BASE_URL
            # Ollama ignores the SDK key, but the official client requires one.
            api_key = values.get("OLLAMA_API_KEY")
        else:
            raise ValueError("LLM_PROVIDER must be either 'groq' or 'ollama'")

        return cls(
            provider=provider,
            model=values.get("LLM_MODEL", default_model),
            base_url=values.get("LLM_BASE_URL", default_base_url),
            api_key=api_key,
        )


def create_llm_client(config: LLMConfig | None = None) -> OpenAI:
    """Initialize the official OpenAI SDK for the configured compatible API.

    Args:
        config: Optional resolved settings; environment configuration is used
            when omitted.

    Returns:
        An ``OpenAI`` client configured with the selected model service.

    Raises:
        ValueError: If the selected provider has no configured SDK API key.
    """
    settings = config or LLMConfig.from_env()
    if not settings.api_key:
        key_variable = "GROQ_API_KEY" if settings.provider == "groq" else "OLLAMA_API_KEY"
        raise ValueError(f"{key_variable} must be configured for the selected provider")

    return OpenAI(
        api_key=settings.api_key,
        base_url=settings.base_url,
    )
