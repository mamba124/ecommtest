"""key logic that is necessary for calling LLMs and LLM based tools.
Supports Anthropic and Gemini providers.
Provider is auto-detected from env keys: ANTHROPIC_API_KEY takes priority,
then GEMINI_API_KEY. Falls back to Gemini if no explicit config override.
Also supports Ollama for local inference.
"""
import logging
import os
from abc import ABC, abstractmethod
from typing import Iterator

from app.core.config import Config

logger = logging.getLogger("llm")


class BaseLLM(ABC):
    @abstractmethod
    def generate(self, prompt: str, system: str = "") -> str: ...

    @abstractmethod
    def stream(self, prompt: str, system: str = "") -> Iterator[str]: ...


# ── Ollama ────────────────────────────────────────────────────────────────────

class OllamaLLM(BaseLLM):
    def __init__(self, config) -> None:
        import ollama
        self._client = ollama.Client(host=config.base_url)
        self._model = config.model
        self._options = {
            "temperature": config.temperature,
            "num_predict": config.max_tokens,
        }

    def generate(self, prompt: str, system: str = "") -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = self._client.chat(
            model=self._model,
            messages=messages,
            options=self._options,
        )
        return response.message.content

    def stream(self, prompt: str, system: str = "") -> Iterator[str]:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        for chunk in self._client.chat(
            model=self._model,
            messages=messages,
            stream=True,
            options=self._options,
        ):
            yield chunk.message.content


# ── Anthropic ─────────────────────────────────────────────────────────────────

class AnthropicLLM(BaseLLM):
    def __init__(self, config) -> None:
        import anthropic
        self._client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self._model = config.model if config.model != "llama3.2" else "claude-sonnet-4-6"
        self._temperature = config.temperature
        self._max_tokens = config.max_tokens

    def generate(self, prompt: str, system: str = "") -> str:
        kwargs: dict = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        response = self._client.messages.create(**kwargs)
        return response.content[0].text

    def stream(self, prompt: str, system: str = "") -> Iterator[str]:
        kwargs: dict = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        with self._client.messages.stream(**kwargs) as stream:
            for text in stream.text_stream:
                yield text


# ── Gemini ────────────────────────────────────────────────────────────────────

class GeminiLLM(BaseLLM):
    def __init__(self, config) -> None:
        import google.generativeai as genai
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        model_name = config.model if config.model != "llama3.2" else "gemini-1.5-flash"
        self._model = genai.GenerativeModel(
            model_name=model_name,
            generation_config={
                "temperature": config.temperature,
                "max_output_tokens": config.max_tokens,
            },
        )

    def generate(self, prompt: str, system: str = "") -> str:
        full = f"{system}\n\n{prompt}" if system else prompt
        response = self._model.generate_content(full)
        return response.text

    def stream(self, prompt: str, system: str = "") -> Iterator[str]:
        full = f"{system}\n\n{prompt}" if system else prompt
        for chunk in self._model.generate_content(full, stream=True):
            if chunk.text:
                yield chunk.text


# ── Factory ───────────────────────────────────────────────────────────────────

class LLMFactory:
    @staticmethod
    def create(config: Config) -> BaseLLM:
        """
        Provider resolution order:
        1. config.llm.provider if explicitly set to "ollama"
        2. ANTHROPIC_API_KEY present → AnthropicLLM
        3. GEMINI_API_KEY present → GeminiLLM
        4. Default → GeminiLLM (requires GEMINI_API_KEY)
        """
        provider = config.llm.provider

        if provider == "ollama":
            logger.info("LLM provider: Ollama")
            return OllamaLLM(config.llm)

        if provider == "anthropic" or os.getenv("ANTHROPIC_API_KEY"):
            logger.info("LLM provider: Anthropic")
            return AnthropicLLM(config.llm)

        if provider == "gemini" or os.getenv("GEMINI_API_KEY"):
            logger.info("LLM provider: Gemini")
            return GeminiLLM(config.llm)

        logger.warning("No LLM API key found; defaulting to Gemini (will fail without GEMINI_API_KEY)")
        return GeminiLLM(config.llm)
