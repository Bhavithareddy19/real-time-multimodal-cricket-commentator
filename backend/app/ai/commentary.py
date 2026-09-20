import random
import time
import logging
from typing import Dict, Any, List, Optional

from backend.app.config import get_settings
from backend.app.models.schemas import CricketEvent, CommentaryItem
from backend.app.ai.prompts import build_system_prompt, build_event_prompt
from backend.app.ai.llm import LLMProvider, OpenAICompatibleProvider

logger = logging.getLogger("ai.commentary")

class CommentaryMemory:
    """Bounded contextual memory window for recent deliveries and match state."""

    def __init__(self, max_history: int = 8):
        self.max_history = max_history
        self.recent_deliveries: List[Dict[str, Any]] = []
        self.runs: int = 87
        self.wickets: int = 2
        self.overs: str = "12.1"
        self.boundary_count: int = 0

    def update_from_event(self, event: CricketEvent):
        runs_add = event.runs or 0
        wkt_add = 1 if event.event_type == "WICKET" else 0
        if event.event_type in ["FOUR", "SIX"]:
            self.boundary_count += 1

        self.runs += runs_add
        self.wickets += wkt_add
        self.overs = event.delivery_id

        self.recent_deliveries.append({
            "delivery_id": event.delivery_id,
            "event_type": event.event_type,
            "shot_type": event.shot_type,
            "runs": runs_add
        })
        if len(self.recent_deliveries) > self.max_history:
            self.recent_deliveries.pop(0)

    def get_context(self) -> Dict[str, Any]:
        recent_str = ", ".join([f"{d['delivery_id']} {d['event_type']}" for d in self.recent_deliveries[-4:]]) or "None"
        return {
            "runs": self.runs,
            "wickets": self.wickets,
            "overs": self.overs,
            "boundary_count": self.boundary_count,
            "recent_deliveries": recent_str
        }


class FallbackCommentaryGenerator:
    """
    Deterministic rule-based fallback generator.
    Guarantees rich, natural broadcast lines across all 4 personality styles
    without requiring external network or paid LLM API keys.
    """

    FALLBACK_TEMPLATES = {
        "FOUR": {
            "PROFESSIONAL": [
                "Exquisite timing through the covers, and the ball races away for four.",
                "Beautifully driven on the up, piercing the off-side gap to find the boundary.",
                "Pure placement and class, beating the sweeping fielder all the way for four.",
                "Superbly stroked off the back foot, that will race to the fence for four."
            ],
            "ENERGETIC": [
                "Cracking shot! He absolutely creams it through extra cover for four!",
                "Smacked away with authority! The ball bullets to the boundary for four!",
                "Oh, what a strike! That is blistering off the blade and gone for four!",
                "Hammered! That boundary was written all over it, four runs!"
            ],
            "ANALYTICAL": [
                "Flawless weight transfer and high elbow, finding the gap at speed for four.",
                "Capitalized on the width outside off, piercing the cover sweep for four.",
                "Sweet contact right off the middle of the bat, finding the fence."
            ],
            "MINIMAL": [
                "That's four runs to the boundary.",
                "Driven away for four.",
                "Finds the boundary. Four runs."
            ]
        },
        "SIX": {
            "PROFESSIONAL": [
                "Magnificent strike high into the evening sky, and that has gone all the way for six.",
                "Effortless swing of the blade, lofted cleanly into the stands for a maximum.",
                "Struck with superb elevation, clearing the deep boundary rope for six."
            ],
            "ENERGETIC": [
                "Huge hit! That has gone miles into the top tier for six!",
                "Out of the park! What a mammoth strike over long-on for six!",
                "Boom! Maximum! Clears the roof with ease!"
            ],
            "ANALYTICAL": [
                "High launch trajectory, generating tremendous bat speed to clear the rope.",
                "Full swing through the arc, lofting the ball deep over the boundary."
            ],
            "MINIMAL": [
                "All the way. Six runs.",
                "Cleared the boundary for six.",
                "Maximum. Six runs."
            ]
        },
        "WICKET": {
            "PROFESSIONAL": [
                "Got him! A crucial breakthrough, the batsman has to depart.",
                "Gone! A lapse in judgment, and the bowling side breaks the partnership.",
                "Cleaned up! Superb bowling from the bowler, that is a big wicket."
            ],
            "ENERGETIC": [
                "Out! Timber! The stumps are shattered, absolute pandemonium!",
                "Gone! Big wicket, massive celebration in the middle!",
                "Caught! He takes the catch cleanly, what a massive breakthrough!"
            ],
            "ANALYTICAL": [
                "Beaten for seam movement through the gate, and the stumps are disturbed.",
                "Tempted by the line outside off, yielding the vital dismissal."
            ],
            "MINIMAL": [
                "Wicket! The batsman is out.",
                "Bowled him. Wicket falls.",
                "Out. That is a wicket."
            ]
        },
        "DOT_BALL": {
            "PROFESSIONAL": [
                "Good length outside off, defended solidly into the covers. No run.",
                "Pushed towards mid-off with soft hands, no run to be taken.",
                "Beaten outside off stump as the ball carries through to the keeper."
            ],
            "ENERGETIC": [
                "Steamed in and beaten! Fierce bowling from the quick, no run!",
                "Solid defensive prod, shut down quickly by point. Dot ball!"
            ],
            "ANALYTICAL": [
                "Disciplined line just back of a length, safely defended for a dot.",
                "Good tight corridor outside off, no room offered to score."
            ],
            "MINIMAL": [
                "No run from that delivery.",
                "Dot ball.",
                "Defended, no run."
            ]
        },
        "SHOT_PLAYED": {
            "PROFESSIONAL": [
                "The batsman plays a measured shot out towards the field.",
                "Nicely presented full face of the bat, working it into the gap."
            ],
            "ENERGETIC": [
                "Lashes out with intent, ball travels into the outfield!",
                "Good aggressive stroke from the striker!"
            ],
            "ANALYTICAL": [
                "Quick hands through the line, ball traveling through the field.",
                "Calculated stroke played into vacant turf."
            ],
            "MINIMAL": [
                "Shot played into the outfield.",
                "Struck into the field."
            ]
        },
        "DELIVERY": {
            "PROFESSIONAL": [
                "The bowler runs in and delivers outside the off stump.",
                "Here comes the bowler into the crease, pitching on a good length."
            ],
            "ENERGETIC": [
                "Steaming in at pace, here comes the delivery!",
                "Bowler charges in with high energy, release is sharp!"
            ],
            "ANALYTICAL": [
                "Full upright seam presentation as the bowler releases from over the wicket.",
                "Bowler hits the deck hard around the corridor of uncertainty."
            ],
            "MINIMAL": [
                "Ball is bowled.",
                "Bowler delivers."
            ]
        }
    }

    @classmethod
    def generate(cls, event: CricketEvent, style: str = "PROFESSIONAL") -> str:
        style_key = style.upper() if style.upper() in ["PROFESSIONAL", "ENERGETIC", "ANALYTICAL", "MINIMAL"] else "PROFESSIONAL"
        event_key = event.event_type if event.event_type in cls.FALLBACK_TEMPLATES else "DOT_BALL"

        templates = cls.FALLBACK_TEMPLATES.get(event_key, {}).get(style_key, ["Play continues in the middle."])
        text = random.choice(templates)

        if event.shot_type and event.shot_type != "UNKNOWN" and "SHOT" in text:
            text = text.replace("shot", event.shot_type.lower().replace("_", " "))

        return text


class CommentaryEngine:
    """
    Commentary Engine integrating LLM Provider with graceful local fallback.
    Maintains match memory and enforces live broadcasting constraints.
    """

    def __init__(self, provider: Optional[LLMProvider] = None):
        self.settings = get_settings()
        self.provider: LLMProvider = provider or OpenAICompatibleProvider()
        self.memory = CommentaryMemory()
        self.fallback = FallbackCommentaryGenerator()

    def set_provider(self, provider: LLMProvider):
        self.provider = provider
        logger.info(f"CommentaryEngine provider set to {provider.__class__.__name__}")

    async def generate(self, event: CricketEvent, style: str = "PROFESSIONAL") -> CommentaryItem:
        """
        Generates commentary for a visual cricket event.
        Attempts LLM generation first; falls back immediately on error or missing credentials.
        """
        self.memory.update_from_event(event)
        start_time = time.perf_counter()

        # Check if LLM is enabled and configured
        if self.settings.ENABLE_LLM and self.provider and self.provider.is_configured():
            try:
                system_prompt = build_system_prompt(style)
                event_prompt = build_event_prompt(event.model_dump(), self.memory.get_context(), style)

                text, first_token_ms, total_ms = await self.provider.generate_commentary(
                    prompt=event_prompt,
                    system_prompt=system_prompt
                )

                logger.info(f"LLM commentary generated ({first_token_ms:.0f}ms): {text}")
                return CommentaryItem(
                    delivery_id=event.delivery_id,
                    event_type=event.event_type,
                    text=text,
                    style=style,
                    is_fallback=False,
                    llm_latency_ms=round(first_token_ms, 1),
                    timestamp=time.time()
                )
            except Exception as e:
                logger.warning(f"LLM generation failed ({e}). Gracefully falling back to local deterministic commentary.")

        # Fallback Generator
        fallback_text = self.fallback.generate(event, style=style)
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return CommentaryItem(
            delivery_id=event.delivery_id,
            event_type=event.event_type,
            text=fallback_text,
            style=style,
            is_fallback=True,
            llm_latency_ms=round(latency_ms, 1),
            timestamp=time.time()
        )
