import logging
from typing import List, Optional, Callable
from backend.app.models.schemas import CricketEvent, MatchScorecard

logger = logging.getLogger("cricket.events")

class EventDispatcher:
    """
    Central event aggregator and dispatcher.
    Updates match scorecard and fans out structured events to commentary and WebSockets.
    """

    def __init__(self):
        self.scorecard = MatchScorecard(match_id="match_001", overs=12.0, runs=87, wickets=2)
        self.event_history: List[CricketEvent] = []
        self._listeners: List[Callable[[CricketEvent], None]] = []

    def add_listener(self, listener: Callable[[CricketEvent], None]):
        self._listeners.append(listener)

    def record_event(self, event: CricketEvent):
        self.event_history.append(event)
        if len(self.event_history) > 100:
            self.event_history.pop(0)

        # Update Match Scorecard
        runs_add = event.runs or 0
        wkt_add = 1 if event.event_type == "WICKET" else 0

        self.scorecard.runs += runs_add
        self.scorecard.wickets += wkt_add
        self.scorecard.recent_events.append(f"{event.delivery_id}: {event.event_type}")
        if len(self.scorecard.recent_events) > 10:
            self.scorecard.recent_events.pop(0)

        logger.info(f"Cricket Event Recorded: {event.event_type} ({event.shot_type or 'None'}) -> Score: {self.scorecard.runs}/{self.scorecard.wickets}")

        # Dispatch to all listeners
        for l in self._listeners:
            try:
                l(event)
            except Exception as e:
                logger.error(f"Error in event listener callback: {e}")

    def get_recent_events(self, count: int = 10) -> List[CricketEvent]:
        return self.event_history[-count:]
