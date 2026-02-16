from __future__ import annotations

import logging
from functools import lru_cache
from typing import Optional

from app.config import settings

logger = logging.getLogger(__name__)


class LLMRouter:
    SUPPORTED_PROVIDERS = {
        "openai": ["gpt-4o-mini", "gpt-4o", "gpt-4", "gpt-3.5-turbo", "gpt-4-turbo"],
        "anthropic": [
            "claude-3-5-sonnet-20241022",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "claude-sonnet-4-20250514",  # Added for research analysis
        ],
        "groq": [
            "mixtral-8x7b-32768",
            "llama-3.1-70b-versatile",
            "llama-3.1-8b-instant",
            "llama3-70b-8192",
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
            logger.warning("Unsupported provider '%s', falling back to 'openai'", provider)
            self.provider = "openai"

        allowed_models = self.SUPPORTED_PROVIDERS[self.provider]
        if model not in allowed_models:
            fallback = self.DEFAULT_MODEL[self.provider]
            logger.warning(
                "Model '%s' not allowed for provider '%s'. Falling back to '%s'.",
                model,
                self.provider,
                fallback,
            )
            self.model = fallback
        else:
            self.model = model

        self.client = self._get_client()
        logger.info("LLMRouter initialized: %s / %s (temp=%.2f, max_tokens=%d)", 
                   self.provider, self.model, self.temperature, self.max_tokens)

    def _get_client(self):
        if self.provider == "openai":
            try:
                from openai import OpenAI  # type: ignore
            except ImportError:
                raise ImportError("OpenAI package not installed. Run: pip install openai")

            api_key = getattr(settings, "OPENAI_API_KEY", None)
            if not api_key:
                raise ValueError("OPENAI_API_KEY not set in environment")
            return OpenAI(api_key=api_key)

        if self.provider == "anthropic":
            try:
                from anthropic import Anthropic  # type: ignore
            except ImportError:
                raise ImportError("Anthropic package not installed. Run: pip install anthropic")

            api_key = getattr(settings, "ANTHROPIC_API_KEY", None)
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not set in environment")
            return Anthropic(api_key=api_key)

        if self.provider == "groq":
            try:
                from groq import Groq  # type: ignore
            except ImportError:
                raise ImportError("Groq package not installed. Run: pip install groq")

            api_key = getattr(settings, "GROQ_API_KEY", None)
            if not api_key:
                raise ValueError("GROQ_API_KEY not set in environment")
            return Groq(api_key=api_key)

        raise ValueError(f"Unsupported provider '{self.provider}'")

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate completion from LLM."""
        if not prompt or not str(prompt).strip():
            logger.warning("Empty prompt provided to generate()")
            return ""

        try:
            if self.provider == "openai":
                return self._call_openai(prompt, system_prompt)
            if self.provider == "anthropic":
                return self._call_anthropic(prompt, system_prompt)
            if self.provider == "groq":
                return self._call_groq(prompt, system_prompt)

            raise ValueError(f"Unsupported provider: {self.provider}")
        except Exception as e:
            logger.error("LLM generation failed (%s/%s): %s", self.provider, self.model, e, exc_info=True)
            raise

    def _call_openai(self, prompt: str, system_prompt: Optional[str]) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            logger.error("OpenAI API call failed: %s", e)
            raise

    def _call_anthropic(self, prompt: str, system_prompt: Optional[str]) -> str:
        system = system_prompt or "You are a helpful assistant."
        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )
            # Handle both TextBlock and string content
            if hasattr(resp, "content") and resp.content:
                if isinstance(resp.content, list) and len(resp.content) > 0:
                    content_block = resp.content[0]
                    if hasattr(content_block, "text"):
                        return content_block.text.strip()
                    return str(content_block).strip()
                return str(resp.content).strip()
            return ""
        except Exception as e:
            logger.error("Anthropic API call failed: %s", e)
            raise

    def _call_groq(self, prompt: str, system_prompt: Optional[str]) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            logger.error("Groq API call failed: %s", e)
            raise


@lru_cache(maxsize=32)
def get_llm_router(
    provider: str,
    model: str,
    temperature: float = 0.3,
    max_tokens: int = 2000,
) -> LLMRouter:
    """Get cached LLM router instance."""
    return LLMRouter(
        provider=provider,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def clear_router_cache() -> None:
    """Clear LRU cache for router instances."""
    get_llm_router.cache_clear()


__all__ = ["LLMRouter", "get_llm_router", "clear_router_cache"]