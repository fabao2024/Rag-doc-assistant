"""
Multi-Provider LLM Configuration Module

Supports: OpenAI, Anthropic, Google, ZhipuAI, Azure OpenAI, Ollama
"""
import os
from typing import Optional, Any

PROVIDER_OPENAI = "openai"
PROVIDER_ANTHROPIC = "anthropic"
PROVIDER_GOOGLE = "google"
PROVIDER_ZHIPUAI = "zhipuai"
PROVIDER_AZURE = "azure"
PROVIDER_OLLAMA = "ollama"

ALL_PROVIDERS = [
    PROVIDER_OPENAI,
    PROVIDER_ANTHROPIC,
    PROVIDER_GOOGLE,
    PROVIDER_ZHIPUAI,
    PROVIDER_AZURE,
    PROVIDER_OLLAMA,
]


def get_provider() -> str:
    """Get the configured LLM provider."""
    return os.getenv("LLM_PROVIDER", PROVIDER_OPENAI).lower()


def validate_provider(provider: str) -> None:
    """Validate that the provider is supported."""
    if provider not in ALL_PROVIDERS:
        raise ValueError(
            f"Unknown provider: {provider}. "
            f"Supported providers: {', '.join(ALL_PROVIDERS)}"
        )


def get_llm(provider: Optional[str] = None, **kwargs) -> Any:
    """
    Get a language model instance based on the configured provider.
    """
    provider = (provider or get_provider()).lower()
    validate_provider(provider)

    try:
        if provider == PROVIDER_OPENAI:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
                temperature=float(os.getenv("OPENAI_TEMPERATURE", "0")),
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                **kwargs
            )

        elif provider == PROVIDER_ANTHROPIC:
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                model=os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307"),
                temperature=float(os.getenv("ANTHROPIC_TEMPERATURE", "0")),
                anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
                **kwargs
            )

        elif provider == PROVIDER_GOOGLE:
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model=os.getenv("GOOGLE_MODEL", "gemini-pro"),
                temperature=float(os.getenv("GOOGLE_TEMPERATURE", "0")),
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                **kwargs
            )

        elif provider == PROVIDER_ZHIPUAI:
            from langchain_community.chat_models import ChatZhipuAI
            return ChatZhipuAI(
                model=os.getenv("ZHIPUAI_MODEL", "glm-4"),
                temperature=float(os.getenv("ZHIPUAI_TEMPERATURE", "0")),
                zhipuai_api_key=os.getenv("ZHIPUAI_API_KEY"),
                **kwargs
            )

        elif provider == PROVIDER_AZURE:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-35-turbo"),
                temperature=float(os.getenv("OPENAI_TEMPERATURE", "0")),
                openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
                **kwargs
            )

        elif provider == PROVIDER_OLLAMA:
            from langchain_community.chat_models import ChatOllama
            return ChatOllama(
                model=os.getenv("OLLAMA_MODEL", "llama2"),
                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                **kwargs
            )

        raise ValueError(f"Provider {provider} not implemented")

    except ImportError as e:
        raise ImportError(
            f"Missing package for provider '{provider}'. "
            f"Please install the required package: {str(e)}"
        ) from e


def get_embeddings(provider: Optional[str] = None, **kwargs) -> Any:
    """
    Get an embeddings instance.

    Note: For simplicity, we always use OpenAI embeddings since the vector store
    was created with OpenAI embeddings. If you want Ollama embeddings, you need to:
    1. Set OLLAMA_EMBEDDINGS_MODEL env var
    2. Pull the model: ollama pull <model>
    """
    # Always use OpenAI embeddings (the vector store was created with these)
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(
        model=os.getenv("OPENAI_EMBEDDINGS_MODEL", "text-embedding-ada-002"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        **kwargs
    )


def get_required_api_key(provider: str) -> str:
    """Get the required API key environment variable name for a provider."""
    provider = provider.lower()

    key_mapping = {
        PROVIDER_OPENAI: "OPENAI_API_KEY",
        PROVIDER_ANTHROPIC: "ANTHROPIC_API_KEY",
        PROVIDER_GOOGLE: "GOOGLE_API_KEY",
        PROVIDER_ZHIPUAI: "ZHIPUAI_API_KEY",
        PROVIDER_AZURE: "AZURE_OPENAI_API_KEY",
        PROVIDER_OLLAMA: None,  # No API key required
    }

    return key_mapping.get(provider)


def check_api_key(provider: str) -> bool:
    """Check if the required API key is set for a provider."""
    key = get_required_api_key(provider)
    if key is None:
        return True  # Ollama doesn't need an API key
    return bool(os.getenv(key))


def get_provider_info() -> dict:
    """Get information about all supported providers and their status."""
    provider = get_provider()
    info = {}

    for p in ALL_PROVIDERS:
        key = get_required_api_key(p)
        info[p] = {
            "api_key_env": key,
            "configured": check_api_key(p),
            "current": p == provider,
        }

    return info
