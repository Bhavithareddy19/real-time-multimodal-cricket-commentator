import logging
from typing import Optional, Callable, Dict, Any
from backend.app.models.schemas import CommentaryItem
from backend.app.speech.audio_stream import AudioPriorityManager

logger = logging.getLogger("streaming.audio_pipeline")

class AudioPipeline:
    """
    Coordinates commentary-to-audio synthesis and dispatch with priority preemption.
    """

    def __init__(self, audio_manager: Optional[AudioPriorityManager] = None):
        self.manager = audio_manager or AudioPriorityManager()
        self.latest_tts_latency_ms: float = 0.0
        self.latest_first_audio_ms: float = 0.0

    async def process_commentary(
        self,
        item: CommentaryItem,
        dispatch_cb: Callable[[Dict[str, Any]], None]
    ):
        """Processes commentary item into audio with priority preemption."""
        def wrapped_dispatch(payload: Dict[str, Any]):
            if payload.get("type") == "audio":
                self.latest_first_audio_ms = payload.get("first_audio_ms", 0.0)
                self.latest_tts_latency_ms = payload.get("tts_latency_ms", 0.0)
            dispatch_cb(payload)

        await self.manager.synthesize_and_dispatch(item, wrapped_dispatch)
