from typing import Dict, Any

STYLE_DESCRIPTIONS = {
    "PROFESSIONAL": "traditional, polished, and measured cricket broadcast commentary (like Richie Benaud or Michael Atherton).",
    "ENERGETIC": "energetic, electrifying, high-energy, passionate T20 commentary (like Ravi Shastri or Danny Morrison).",
    "ANALYTICAL": "technical, focused on ball trajectory, mechanics, field placement, and batsman footwork.",
    "MINIMAL": "ultra-concise, strictly stating the essential outcome in 10 words or fewer."
}

def build_system_prompt(style: str = "PROFESSIONAL") -> str:
    style_desc = STYLE_DESCRIPTIONS.get(style.upper(), STYLE_DESCRIPTIONS["PROFESSIONAL"])
    return f"""You are an elite live cricket commentator providing commentary in a {style_desc} style.

CRITICAL BROADCAST RULES:
1. Ground every sentence ONLY in the provided event and match context.
2. Do NOT invent player names (refer to 'the batsman', 'the bowler', 'the striker', or 'the fielder').
3. Do NOT invent scores or match situations not explicitly provided.
4. Do NOT claim a boundary or wicket unless the event explicitly states FOUR, SIX, or WICKET.
5. Keep your commentary strictly under 25 words.
6. Sound like an authentic, natural cricket commentator on live television."""

def build_event_prompt(event_data: Dict[str, Any], match_context: Dict[str, Any], style: str = "PROFESSIONAL") -> str:
    over = match_context.get("overs", "12.1")
    score = f"{match_context.get('runs', 87)}/{match_context.get('wickets', 2)}"
    recent_history = match_context.get("recent_deliveries", "DOT, FOUR")

    event_type = event_data.get("event_type", "DOT_BALL")
    shot_type = event_data.get("shot_type", "None")
    runs = event_data.get("runs", 0)
    speed = f"{event_data.get('ball_speed_kmh')} km/h" if event_data.get("ball_speed_kmh") else "Not calibrated"
    confidence = f"{int(event_data.get('confidence', 0.9) * 100)}%"

    return f"""Current Match Context:
- Over: {over}
- Current Score: {score}
- Recent Deliveries: {recent_history}

Observed Visual Event:
- Event: {event_type}
- Shot Played: {shot_type}
- Runs: {runs}
- Ball Speed: {speed}
- Detection Confidence: {confidence}

Generate one natural, single-sentence broadcast commentary line for this event in {style} style."""
