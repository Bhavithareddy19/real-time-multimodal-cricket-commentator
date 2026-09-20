import asyncio
import pytest
from backend.app.models.schemas import CommentaryItem
from backend.app.speech.tts import MockTTSProvider, EdgeTTSProvider
from backend.app.speech.audio_stream import AudioPriorityManager
from backend.app.streaming.audio_pipeline import AudioPipeline

@pytest.mark.asyncio
async def test_mock_tts_provider():
    mock_tts = MockTTSProvider()
    audio, first_ms, total_ms = await mock_tts.synthesize("Glorious cover drive!")
    assert len(audio) > 0
    assert first_ms > 0
    assert total_ms >= first_ms

@pytest.mark.asyncio
async def test_audio_priority_manager_priorities():
    manager = AudioPriorityManager()
    assert manager.get_priority("WICKET") == 100
    assert manager.get_priority("SIX") == 80
    assert manager.get_priority("FOUR") == 60
    assert manager.get_priority("SHOT_PLAYED") == 40
    assert manager.get_priority("DOT_BALL") == 20
    assert manager.get_priority("WICKET") > manager.get_priority("FOUR") > manager.get_priority("DOT_BALL")

@pytest.mark.asyncio
async def test_audio_priority_manager_dispatch():
    mock_tts = MockTTSProvider()
    manager = AudioPriorityManager(tts_provider=mock_tts)
    
    dispatched = []
    def callback(payload):
        dispatched.append(payload)

    item = CommentaryItem(
        id="item-1",
        delivery_id="1.1",
        event_type="FOUR",
        text="Cracking drive through extra cover for four!",
        style="PROFESSIONAL",
        is_fallback=False,
        llm_latency_ms=120.0,
        timestamp=100.0
    )

    await manager.synthesize_and_dispatch(item, callback)
    assert len(dispatched) == 1
    assert dispatched[0]["type"] == "audio"
    assert dispatched[0]["event_type"] == "FOUR"
    assert dispatched[0]["priority"] == 60
    assert "audio_base64" in dispatched[0]
    assert dispatched[0]["format"] == "mp3"

@pytest.mark.asyncio
async def test_audio_priority_manager_preemption():
    class SlowTTS:
        async def synthesize(self, text: str):
            await asyncio.sleep(0.5)
            return b"slow_audio", 10.0, 500.0

    manager = AudioPriorityManager(tts_provider=SlowTTS())
    dispatched = []
    def callback(payload):
        dispatched.append(payload)

    routine_item = CommentaryItem(
        id="item-dot",
        delivery_id="1.1",
        event_type="DOT_BALL",
        text="Defended gently back to the bowler.",
        style="PROFESSIONAL",
        is_fallback=False,
        llm_latency_ms=100.0,
        timestamp=100.0
    )

    wicket_item = CommentaryItem(
        id="item-wkt",
        delivery_id="1.2",
        event_type="WICKET",
        text="OUT! Clean bowled! Timber disturbed!",
        style="ENERGETIC",
        is_fallback=False,
        llm_latency_ms=110.0,
        timestamp=101.0
    )

    # Start routine dot ball in background
    task1 = asyncio.create_task(manager.synthesize_and_dispatch(routine_item, callback))
    await asyncio.sleep(0.05) # let task1 start running

    # Preempt with WICKET
    await manager.synthesize_and_dispatch(wicket_item, callback)

    # Wait for task1 (which should be cancelled)
    try:
        await task1
    except asyncio.CancelledError:
        pass

    # Verify clear_queue action was dispatched
    control_actions = [d for d in dispatched if d.get("type") == "audio_control"]
    assert len(control_actions) >= 1
    assert control_actions[0]["action"] == "clear_queue"
    assert "Preempted by WICKET" in control_actions[0]["reason"]

@pytest.mark.asyncio
async def test_audio_pipeline_integration():
    mock_tts = MockTTSProvider()
    manager = AudioPriorityManager(tts_provider=mock_tts)
    pipeline = AudioPipeline(audio_manager=manager)

    dispatched = []
    def callback(payload):
        dispatched.append(payload)

    item = CommentaryItem(
        id="item-six",
        delivery_id="2.3",
        event_type="SIX",
        text="Massive hit! That has gone all the way into the top tier!",
        style="ENERGETIC",
        is_fallback=False,
        llm_latency_ms=140.0,
        timestamp=102.0
    )

    await pipeline.process_commentary(item, callback)
    assert len(dispatched) == 1
    assert dispatched[0]["type"] == "audio"
    assert pipeline.latest_tts_latency_ms > 0
