"""TokenRouter LLM client — wraps OpenAIClient with TokenRouter-specific defaults.

Provides a one-liner to use TokenRouter (or any OpenAI-compatible endpoint)
with API key from the environment. Recommended for local development and
benchmarks where you don't want to hard-code base URLs.

Usage:
    import os
    os.environ['OPENAI_API_KEY'] = 'sk-...'
    os.environ['OPENAI_BASE_URL'] = 'https://api.tokenrouter.com/v1'

    from recall.llm_router import RouterClient
    llm = RouterClient(model='openai/gpt-4o-mini')

The RouterClient exposes the same protocol as LLMClient.
"""
from __future__ import annotations

import os
from typing import Any

from recall.llm import OpenAIClient


class RouterClient(OpenAIClient):
    """OpenAI-compatible client preconfigured for TokenRouter / OpenRouter / etc.

    Picks `OPENAI_BASE_URL` and `OPENAI_API_KEY` from the environment unless
    overridden.
    """

    def __init__(
        self,
        model: str = "openai/gpt-4o-mini",
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        super().__init__(
            model=model,
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
        )


# Recommended cheap models for the typical Recall LLM operations.
SUGGESTED_QUALITY_MODELS = (
    "openai/gpt-4o-mini",          # OpenAI mini
    "anthropic/claude-haiku-4.5",  # Anthropic mini
    "deepseek/deepseek-v3.2",      # DeepSeek
    "google/gemini-3-flash-preview",  # Google flash
)

SUGGESTED_GENERATION_MODELS = (
    "anthropic/claude-sonnet-4.6",
    "openai/gpt-4o-mini",
    "deepseek/deepseek-v4-pro",
)
