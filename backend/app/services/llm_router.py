"""
Multi-Provider LLM Router for Political Discourse Analysis
Supports OpenAI, Anthropic, and Groq APIs

LOCATION: backend/app/services/llm_router.py
"""

import logging
from typing import Optional
from functools import lru_cache

from app.config import settings

logger = logging.getLogger(__name__)


class LLMRouter:
    """
    Unified interface for multiple LLM providers.
    """

    SUPPORTED_PROVIDERS = {
        "openai": ["gpt-4o-mini", "gpt-4o", "gpt-4", "gpt-3.5-turbo"],
        "anthropic": [
            "claude-3-5-sonnet-20241022",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
        ],
        "groq": [
            "mixtral-8x7b-32768",
            "llama-3.1-70b-versatile",
            "llama-3.1-8b-instant",
        ],
    }

    DEFAULT_MODEL = {
        "openai": "gpt-4o-mini",
        "anthropic": "claude-3-5-sonnet-20241022",
        "groq": "llama-3.1-70b-versatile",
    }

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4o-mini",
        temperature: float = 0.3,
        max_tokens: int = 2000,
    ):
        self.provider = (provider or "openai").lower().strip()
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)

        if self.provider not in self.SUPPORTED_PROVIDERS:
            logger.warning(f"Unsupported provider '{provider}', falling back to openai")
            self.provider = "openai"

        allowed_models = self.SUPPORTED_PROVIDERS[self.provider]
        if model not in allowed_models:
            fallback = self.DEFAULT_MODEL[self.provider]
            logger.warning(
                f"Model '{model}' not in allowed list for provider '{self.provider}'. "
                f"Falling back to '{fallback}'."
            )
            self.model = fallback
        else:
            self.model = model

        self.client = self._get_client()
        logger.info(f"LLMRouter initialized: {self.provider} / {self.model}")

    def _get_client(self):
        try:
            if self.provider == "openai":
                from openai import OpenAI
                if not settings.OPENAI_API_KEY:
                    raise ValueError("OPENAI_API_KEY not set")
                return OpenAI(api_key=settings.OPENAI_API_KEY)

            if self.provider == "anthropic":
                from anthropic import Anthropic
                if not getattr(settings, "ANTHROPIC_API_KEY", None):
                    raise ValueError("ANTHROPIC_API_KEY not set")
                return Anthropic(api_key=settings.ANTHROPIC_API_KEY)

            if self.provider == "groq":
                from groq import Groq
                if not getattr(settings, "GROQ_API_KEY", None):
                    raise ValueError("GROQ_API_KEY not set")
                return Groq(api_key=settings.GROQ_API_KEY)

            raise ValueError(f"Unsupported provider '{self.provider}'")

        except Exception as e:
            logger.error(f"Failed to initialize {self.provider} client: {e}")
            raise

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Generate text. Raises on failure (do not silently return empty).
        """
        if not prompt or not str(prompt).strip():
            return ""

        if self.provider == "openai":
            return self._call_openai(prompt, system_prompt)
        if self.provider == "anthropic":
            return self._call_anthropic(prompt, system_prompt)
        if self.provider == "groq":
            return self._call_groq(prompt, system_prompt)

        raise ValueError(f"Unsupported provider: {self.provider}")

    def _call_openai(self, prompt: str, system_prompt: Optional[str]) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()

    def _call_anthropic(self, prompt: str, system_prompt: Optional[str]) -> str:
        system = system_prompt or "You are a helpful assistant."
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        # Anthropic returns list blocks
        content0 = resp.content[0].text if resp.content else ""
        return (content0 or "").strip()

    def _call_groq(self, prompt: str, system_prompt: Optional[str]) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()


@lru_cache(maxsize=32)
def get_llm_router(
    provider: str,
    model: str,
    temperature: float = 0.3,
    max_tokens: int = 2000,
) -> LLMRouter:
    """
    Cached router. Cache key includes max_tokens so different callers don't conflict.
    """
    return LLMRouter(
        provider=provider,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
