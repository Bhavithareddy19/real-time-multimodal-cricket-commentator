import asyncio
import time
import logging
from abc import ABC, abstractmethod
from typing import AsyncGenerator, Tuple, Optional
import edge_tts

from backend.app.config import get_settings

logger = logging.getLogger("speech.tts")

class TTSProvider(ABC):
    """Abstract base class for streaming Text-To-Speech providers."""

    @abstractmethod
    async def synthesize_stream(self, text: str) -> AsyncGenerator[bytes, None]:
        """Yields audio byte chunks (MP3/PCM) as they become available."""
        pass

    @abstractmethod
    async def synthesize(self, text: str) -> Tuple[bytes, float, float]:
        """Synthesizes text and returns (full_audio_bytes, first_audio_ms, total_ms)."""
        pass


class EdgeTTSProvider(TTSProvider):
    """
    High-performance streaming TTS provider using Microsoft Edge TTS.
    Offers low-latency broadcast-quality voices without subscription keys.
    """

    def __init__(self, voice: Optional[str] = None, rate: str = "+12%"):
        settings = get_settings()
        self.voice = voice or settings.TTS_VOICE
        self.rate = rate

    async def synthesize_stream(self, text: str) -> AsyncGenerator[bytes, None]:
        if not text or not text.strip():
            return

        communicate = edge_tts.Communicate(text.strip(), self.voice, rate=self.rate)
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                yield chunk["data"]

    async def synthesize(self, text: str) -> Tuple[bytes, float, float]:
        if not text or not text.strip():
            return b"", 0.0, 0.0

        start_time = time.perf_counter()
        first_audio_time = None
        chunks = []

        try:
            communicate = edge_tts.Communicate(text.strip(), self.voice, rate=self.rate)
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    if first_audio_time is None:
                        first_audio_time = time.perf_counter()
                    chunks.append(chunk["data"])

            total_ms = (time.perf_counter() - start_time) * 1000.0
            first_ms = (first_audio_time - start_time) * 1000.0 if first_audio_time else total_ms
            full_audio = b"".join(chunks)

            logger.info(f"Synthesized {len(full_audio)} bytes in {total_ms:.1f}ms (First audio: {first_ms:.1f}ms)")
            return full_audio, first_ms, total_ms

        except Exception as e:
            logger.error(f"EdgeTTS synthesis failed: {e}")
            raise


class MockTTSProvider(TTSProvider):
    """Mock TTS provider generating dummy MP3 header bytes for fast testing."""

    async def synthesize_stream(self, text: str) -> AsyncGenerator[bytes, None]:
        yield b"\xff\xfb\x90\x44" + b"\x00" * 100

    async def synthesize(self, text: str) -> Tuple[bytes, float, float]:
        return b"\xff\xfb\x90\x44" + b"\x00" * 200, 15.0, 30.0
