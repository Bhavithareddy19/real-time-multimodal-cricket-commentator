import pytest
from backend.app.ai.commentary import CommentaryMemory, FallbackCommentaryGenerator, CommentaryEngine
from backend.app.ai.prompts import build_system_prompt, build_event_prompt
from backend.app.models.schemas import CricketEvent

def test_commentary_memory_window():
    mem = CommentaryMemory(max_history=5)
    
    # Add 7 events
    for i in range(7):
        ev = CricketEvent(
            match_id="match_001",
            delivery_id=f"12.{i+1}",
            frame_id=i * 30,
            event_type="FOUR" if i == 2 else "DOT_BALL",
            shot_type="COVER_DRIVE" if i == 2 else "DEFENSIVE",
            runs=4 if i == 2 else 0,
            confidence=0.9
        )
        mem.update_from_event(ev)

    ctx = mem.get_context()
    assert mem.runs == 87 + 4
    assert mem.boundary_count == 1
    assert len(mem.recent_deliveries) == 5  # Bounded to max_history

def test_fallback_commentary_generator_styles():
    gen = FallbackCommentaryGenerator()
    
    ev_four = CricketEvent(
        match_id="match_001",
        delivery_id="12.1",
        frame_id=100,
        event_type="FOUR",
        shot_type="COVER_DRIVE",
        runs=4,
        confidence=0.92
    )

    for style in ["PROFESSIONAL", "ENERGETIC", "ANALYTICAL", "MINIMAL"]:
        line = gen.generate(ev_four, style=style)
        assert isinstance(line, str)
        assert len(line) > 5
        words = line.split()
        assert len(words) < 30
        assert "four" in line.lower() or "boundary" in line.lower() or "fence" in line.lower()

    ev_wicket = CricketEvent(
        match_id="match_001",
        delivery_id="12.2",
        frame_id=150,
        event_type="WICKET",
        runs=0,
        confidence=0.95
    )
    wkt_line = gen.generate(ev_wicket, style="PROFESSIONAL")
    assert "wicket" in wkt_line.lower() or "got him" in wkt_line.lower() or "gone" in wkt_line.lower() or "cleaned up" in wkt_line.lower()

def test_prompt_builders():
    sys_prompt = build_system_prompt("ENERGETIC")
    assert "cricket commentator" in sys_prompt.lower()
    assert "energetic" in sys_prompt.lower()

    ev_data = {
        "event_type": "SIX",
        "shot_type": "PULL",
        "runs": 6,
        "ball_speed_kmh": 135.0,
        "confidence": 0.95
    }
    match_ctx = {"overs": "14.2", "runs": 110, "wickets": 3}
    ev_prompt = build_event_prompt(ev_data, match_ctx, "PROFESSIONAL")
    assert "SIX" in ev_prompt
    assert "PULL" in ev_prompt
    assert "135.0 km/h" in ev_prompt

@pytest.mark.asyncio
async def test_commentary_engine_fallback():
    engine = CommentaryEngine()
    ev = CricketEvent(
        match_id="match_001",
        delivery_id="12.3",
        frame_id=200,
        event_type="FOUR",
        shot_type="COVER_DRIVE",
        runs=4,
        confidence=0.92
    )
    
    item = await engine.generate(ev, style="PROFESSIONAL")
    assert item is not None
    assert item.delivery_id == "12.3"
    assert item.event_type == "FOUR"
    assert len(item.text) > 10
    # In test environment without OPENAI_API_KEY, gracefully falls back to deterministic generator
    assert item.is_fallback is True
    assert item.llm_latency_ms >= 0.0
