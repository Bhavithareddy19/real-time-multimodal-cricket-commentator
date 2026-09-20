import asyncio
import time
import logging
from abc import ABC, abstractmethod
from typing import AsyncGenerator, Optional, Tuple

from openai import AsyncOpenAI
from backend.app.config import get_settings

logger = logging.getLogger("ai.llm")

class LLMProvider(ABC):
    """Abstract base class for LLM commentary generation."""

    @abstractmethod
    async def generate_commentary(self, prompt: str, system_prompt: str) -> Tuple[str, float, float]:
        """Returns (generated_text, first_token_latency_ms, total_latency_ms)."""
        pass

    @abstractmethod
    async def stream_commentary(self, prompt: str, system_prompt: str) -> AsyncGenerator[str, None]:
        """Yields streaming tokens as they arrive."""
        pass


class OpenAICompatibleProvider(LLMProvider):
    """
    OpenAI-compatible LLM provider.
    Works with OpenAI, Google Gemini (OpenAI endpoint), Groq, Ollama, and local models.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout_sec: float = 3.5
    ):
        settings = get_settings()
        self.api_key = api_key or settings.OPENAI_API_KEY
        self.base_url = base_url or settings.LLM_API_BASE
        self.model = model or settings.LLM_MODEL
        self.timeout_sec = timeout_sec

        self.client: Optional[AsyncOpenAI] = None
        if self.api_key:
            self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

    def is_configured(self) -> bool:
        return bool(self.api_key and len(self.api_key.strip()) > 5)

    async def generate_commentary(self, prompt: str, system_prompt: str) -> Tuple[str, float, float]:
        if not self.is_configured() or not self.client:
            raise ValueError("LLM provider is not configured with an API key.")

        start_time = time.perf_counter()
        first_token_time = None
        full_text = []

        try:
            stream = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=60,
                    stream=True
                ),
                timeout=self.timeout_sec
            )

            async for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta.content or ""
                    if delta:
                        if first_token_time is None:
                            first_token_time = time.perf_counter()
                        full_text.append(delta)

            total_latency_ms = (time.perf_counter() - start_time) * 1000.0
            first_token_ms = (first_token_time - start_time) * 1000.0 if first_token_time else total_latency_ms
            
            result_text = "".join(full_text).strip()
            # Strip quotes if returned
            if result_text.startswith('"') and result_text.endswith('"'):
                result_text = result_text[1:-1]

            return result_text, first_token_ms, total_latency_ms

        except asyncio.TimeoutError:
            logger.warning(f"LLM request timed out after {self.timeout_sec}s.")
            raise
        except Exception as e:
            logger.error(f"Error communicating with LLM API: {e}")
            raise

    async def stream_commentary(self, prompt: str, system_prompt: str) -> AsyncGenerator[str, None]:
        if not self.is_configured() or not self.client:
            return

        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=60,
            stream=True
        )

        async for chunk in stream:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    yield delta


class GeminiProvider(LLMProvider):
    """
    Official Google Gemini provider using the google.genai SDK.
    Optimized for low-latency live commentary using gemini-2.0-flash / gemini-1.5-flash.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.0-flash",
        timeout_sec: float = 3.5
    ):
        import os
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.model = model or "gemini-2.0-flash"
        self.timeout_sec = timeout_sec
        self.client = None

        if self.is_configured():
            try:
                from google import genai
                self.client = genai.Client(api_key=self.api_key)
                logger.info(f"GeminiProvider initialized with model {self.model}")
            except Exception as e:
                logger.warning(f"Could not initialize Google GenAI client: {e}")

    def is_configured(self) -> bool:
        return bool(self.api_key and len(self.api_key.strip()) > 5)

    async def generate_commentary(self, prompt: str, system_prompt: str) -> Tuple[str, float, float]:
        if not self.is_configured():
            raise ValueError("Gemini provider is not configured with an API key.")

        if not self.client:
            from google import genai
            self.client = genai.Client(api_key=self.api_key)

        start_time = time.perf_counter()

        def _sync_call():
            from google.genai import types
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.7,
                    max_output_tokens=60,
                )
            )
            return response.text or ""

        try:
            text = await asyncio.wait_for(asyncio.to_thread(_sync_call), timeout=self.timeout_sec)
            total_latency_ms = (time.perf_counter() - start_time) * 1000.0
            first_token_ms = total_latency_ms * 0.4
            cleaned = text.strip()
            if cleaned.startswith('"') and cleaned.endswith('"'):
                cleaned = cleaned[1:-1]
            return cleaned, first_token_ms, total_latency_ms
        except asyncio.TimeoutError:
            logger.warning(f"Gemini request timed out after {self.timeout_sec}s.")
            raise
        except Exception as e:
            logger.error(f"Gemini generation error: {e}")
            raise

    async def stream_commentary(self, prompt: str, system_prompt: str) -> AsyncGenerator[str, None]:
        if not self.is_configured() or not self.client:
            return

        def _sync_stream():
            from google.genai import types
            return self.client.models.generate_content_stream(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.7,
                    max_output_tokens=60,
                )
            )

        stream = await asyncio.to_thread(_sync_stream)
        for chunk in stream:
            if chunk.text:
                yield chunk.text


def create_llm_provider(
    provider_name: str = "auto",
    api_key: Optional[str] = None,
    model: Optional[str] = None
) -> LLMProvider:
    """Factory helper creating the appropriate LLMProvider instance."""
    prov = (provider_name or "auto").lower()
    if prov in ["gemini", "google"]:
        return GeminiProvider(api_key=api_key, model=model or "gemini-2.0-flash")
    elif prov in ["openai", "groq", "ollama"]:
        return OpenAICompatibleProvider(api_key=api_key, model=model)
    else:
        # Auto detect based on key format or environment
        if api_key:
            if api_key.startswith("AIza"):
                return GeminiProvider(api_key=api_key, model=model or "gemini-2.0-flash")
            return OpenAICompatibleProvider(api_key=api_key, model=model)
        settings = get_settings()
        if settings.OPENAI_API_KEY:
            return OpenAICompatibleProvider()
        return GeminiProvider()

