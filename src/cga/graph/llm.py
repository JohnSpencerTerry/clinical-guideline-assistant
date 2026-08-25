"""Single shared chat-model instance, pointed at OpenRouter (OpenAI-compatible API)."""

from langchain_openai import ChatOpenAI

from cga.config import settings


def get_chat_model(*, temperature: float = 0.0, timeout: float = 60.0, max_retries: int = 2) -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.openrouter_model,
        base_url=settings.openrouter_base_url,
        api_key=settings.openrouter_api_key,
        temperature=temperature,
        timeout=timeout,
        max_retries=max_retries,
    )
