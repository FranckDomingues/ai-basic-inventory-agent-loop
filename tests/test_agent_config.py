"""Tests for LLM provider configuration and SDK client initialization."""

import pytest
from openai import OpenAI

from agent import LLMConfig, create_llm_client


def test_groq_configuration_defaults_without_exposing_credentials() -> None:
    config = LLMConfig.from_env({})

    assert config.provider == "groq"
    assert config.model == "llama-3.3-70b-versatile"
    assert config.base_url == "https://api.groq.com/openai/v1"
    assert config.api_key is None


def test_groq_configuration_reads_environment_values() -> None:
    config = LLMConfig.from_env(
        {
            "LLM_PROVIDER": "groq",
            "GROQ_API_KEY": "test-secret",
            "LLM_MODEL": "test-model",
            "LLM_BASE_URL": "https://example.test/v1",
        }
    )

    assert config == LLMConfig(
        provider="groq",
        model="test-model",
        base_url="https://example.test/v1",
        api_key="test-secret",
    )


def test_ollama_configuration_uses_local_openai_compatible_endpoint() -> None:
    config = LLMConfig.from_env(
        {"LLM_PROVIDER": "ollama", "OLLAMA_API_KEY": "local-placeholder"}
    )

    assert config.provider == "ollama"
    assert config.model == "qwen2.5:7b"
    assert config.base_url == "http://localhost:11434/v1"
    assert config.api_key == "local-placeholder"


def test_provider_name_is_case_insensitive() -> None:
    config = LLMConfig.from_env(
        {"LLM_PROVIDER": "OLLAMA", "OLLAMA_API_KEY": "local-placeholder"}
    )

    assert config.provider == "ollama"


def test_unsupported_provider_is_rejected() -> None:
    with pytest.raises(ValueError, match="LLM_PROVIDER"):
        LLMConfig.from_env({"LLM_PROVIDER": "unknown"})


def test_groq_client_requires_api_key_before_client_initialization() -> None:
    config = LLMConfig.from_env({})

    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        create_llm_client(config)


def test_groq_client_initialization_uses_configured_compatible_endpoint() -> None:
    config = LLMConfig.from_env({"GROQ_API_KEY": "test-only-key"})

    client = create_llm_client(config)

    assert isinstance(client, OpenAI)
    assert str(client.base_url) == "https://api.groq.com/openai/v1/"
    client.close()


def test_client_initialization_is_local_and_uses_configured_endpoint() -> None:
    config = LLMConfig.from_env(
        {"LLM_PROVIDER": "ollama", "OLLAMA_API_KEY": "local-placeholder"}
    )

    client = create_llm_client(config)

    assert isinstance(client, OpenAI)
    assert str(client.base_url) == "http://localhost:11434/v1/"
    client.close()


def test_ollama_client_requires_configured_sdk_api_key() -> None:
    config = LLMConfig.from_env({"LLM_PROVIDER": "ollama"})

    with pytest.raises(ValueError, match="OLLAMA_API_KEY"):
        create_llm_client(config)
