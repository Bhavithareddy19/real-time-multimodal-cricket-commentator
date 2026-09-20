import asyncio
import base64
import logging
from typing import Dict, Optional, Any, Callable
from backend.app.models.schemas import CommentaryItem
from backend.app.speech.tts import TTSProvider, EdgeTTSProvider

logger = logging.getLogger("speech.audio_stream")

class AudioPriorityManager:
    """
    Manages audio streaming with configurable event priorities and cancellation.
    Ensures critical events (WICKET, SIX, FOUR) preempt routine commentary.
    """

    DEFAULT_PRIORITIES = {
        "WICKET": 100,
        "SIX": 80,
        "FOUR": 60,
        "SHOT_PLAYED": 40,
        "SINGLE": 30,
        "DOUBLE": 30,
        "DOT_BALL": 20,
        "DELIVERY": 10
    }

    def __init__(
        self,
        tts_provider: Optional[TTSProvider] = None,
        custom_priorities: Optional[Dict[str, int]] = None
    ):
        self.tts = tts_provider or EdgeTTSProvider()
        self.priorities = custom_priorities or dict(self.DEFAULT_PRIORITIES)
        self.current_task: Optional[asyncio.Task] = None
        self.current_priority: int = 0
        self.is_active: bool = False

    def get_priority(self, event_type: str) -> int:
        return self.priorities.get(event_type.upper(), 25)

    def set_priority(self, event_type: str, priority: int):
        self.priorities[event_type.upper()] = priority

    async def synthesize_and_dispatch(
        self,
        item: CommentaryItem,
        dispatch_cb: Callable[[Dict[str, Any]], None]
    ):
        """
        Synthesizes commentary audio with priority preemption.
        If a new high-priority event arrives, cancels lower priority task.
        """
        item_priority = self.get_priority(item.event_type)

        # Check if higher priority item should preempt existing task
        if self.current_task and not self.current_task.done():
            if item_priority > self.current_priority:
                logger.info(f"Preempting lower-priority audio ({self.current_priority}) for {item.event_type} ({item_priority})")
                self.current_task.cancel()
                try:
                    await self.current_task
                except (asyncio.CancelledError, Exception):
                    pass
                # Inform clients to immediately clear their playback queue
                dispatch_cb({
                    "type": "audio_control",
                    "action": "clear_queue",
                    "reason": f"Preempted by {item.event_type}"
                })
            else:
                logger.info(f"Skipping audio synthesis for {item.event_type}: active task has equal or higher priority ({self.current_priority} >= {item_priority})")
                return

        self.current_priority = item_priority
        self.current_task = asyncio.current_task()

        try:
            audio_bytes, first_ms, total_ms = await self.tts.synthesize(item.text)
            if audio_bytes:
                b64_audio = base64.b64encode(audio_bytes).decode("ascii")
                payload = {
                    "type": "audio",
                    "delivery_id": item.delivery_id,
                    "event_type": item.event_type,
                    "priority": item_priority,
                    "audio_base64": b64_audio,
                    "format": "mp3",
                    "first_audio_ms": round(first_ms, 1),
                    "tts_latency_ms": round(total_ms, 1)
                }
                dispatch_cb(payload)
        except asyncio.CancelledError:
            logger.info(f"Audio synthesis task cancelled for delivery {item.delivery_id}")
            raise
        except Exception as e:
            logger.error(f"Error synthesizing audio for {item.delivery_id}: {e}")
        finally:
            if self.current_task == asyncio.current_task():
                self.current_priority = 0
                self.current_task = None
