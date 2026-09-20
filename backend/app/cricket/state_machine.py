import time
import os
import json
import logging
import numpy as np
from typing import List, Tuple, Optional, Dict, Any

from backend.app.models.schemas import CricketEvent, BallState, PlayerPose, Detection
from backend.app.cricket.rules import is_point_inside_polygon, has_crossed_boundary, is_point_in_roi
from backend.app.cricket.shot_classifier import ShotClassifier, ShotClassification
from backend.app.vision.calibration import AutoPitchCalibrator

logger = logging.getLogger("cricket.state_machine")

class CricketEventStateMachine:
    """
    Finite State Machine governing cricket delivery lifecycle and visual event detection.
    Enforces strict debouncing and delivery tracking.
    """

    STATES = [
        "IDLE",
        "BOWLER_APPROACH",
        "DELIVERY",
        "BALL_IN_PLAY",
        "BAT_CONTACT",
        "BALL_TRAVEL",
        "BOUNDARY_DETECTED",
        "WICKET_DETECTED",
        "BALL_COMPLETE"
    ]

    def __init__(self, match_id: str = "match_001", calibration_file: str = "config/calibration.json"):
        self.match_id = match_id
        self.current_state = "IDLE"
        self.previous_state = "IDLE"
        
        # Delivery counter: (over, ball)
        self.current_over = 12
        self.current_ball = 1
        self.delivery_id = f"{self.current_over}.{self.current_ball}"

        # Internal state memory for current delivery
        self.shot_classifier = ShotClassifier()
        self.emitted_boundary = False
        self.emitted_wicket = False
        self.emitted_shot = False
        self.emitted_delivery = False
        self.current_shot: Optional[ShotClassification] = None
        self.current_speed_kmh: Optional[float] = None
        self.state_entry_time = time.time()
        self.ball_complete_frame_delay = 0

        # Load field calibration
        self.calibrator = AutoPitchCalibrator(calibration_file)
        self.boundary_polygon: List[List[int]] = self.calibrator.boundary_polygon
        self.stump_roi: List[int] = self.calibrator.stump_roi
        self.load_calibration(calibration_file)

    def load_calibration(self, filepath: str):
        self.calibrator.load_fallback_config()
        self.boundary_polygon = self.calibrator.boundary_polygon
        self.stump_roi = self.calibrator.stump_roi

    def update_dynamic_calibration(self, frame: np.ndarray):
        """Runs autonomous pitch and boundary calibration on keyframe."""
        is_cal, conf, details = self.calibrator.calibrate_from_frame(frame)
        if is_cal and conf >= 0.70:
            if "pitch_polygon" in details and details["pitch_polygon"]:
                self.pitch_polygon = details["pitch_polygon"]
            logger.info(f"Dynamic field calibration updated (conf={conf:.2f}).")

    def _next_delivery(self):
        """Advances delivery counter and resets delivery-level debounce guards."""
        self.current_ball += 1
        if self.current_ball > 6:
            self.current_ball = 1
            self.current_over += 1
        self.delivery_id = f"{self.current_over}.{self.current_ball}"
        self.emitted_boundary = False
        self.emitted_wicket = False
        self.emitted_shot = False
        self.emitted_delivery = False
        self.current_shot = None
        self.current_speed_kmh = None

    def _transition(self, new_state: str):
        if new_state != self.current_state:
            logger.info(f"[{self.delivery_id}] State Transition: {self.current_state} -> {new_state}")
            self.previous_state = self.current_state
            self.current_state = new_state
            self.state_entry_time = time.time()

    def process_frame(
        self,
        frame_id: int,
        ball_state: Optional[BallState],
        trajectory: List[Tuple[float, float]],
        poses: List[PlayerPose],
        detections: List[Detection],
        timestamp: float,
        video_timestamp: Optional[float] = None
    ) -> Optional[CricketEvent]:
        """
        Executes event state machine evaluation for a single frame.
        Returns a strongly-typed CricketEvent ONLY when a new distinct event occurs.
        """
        emitted_event: Optional[CricketEvent] = None
        v_ts = video_timestamp if video_timestamp is not None else timestamp

        if ball_state and ball_state.speed_kmh and ball_state.speed_estimation_available:
            self.current_speed_kmh = ball_state.speed_kmh

        # Extract batsman pose if present
        batsman_pose = next((p for p in poses if p.role == "batsman"), None)

        # STATE 1: IDLE
        if self.current_state == "IDLE":
            # If ball is detected with downward velocity or bowler is in runup
            if ball_state and (ball_state.vy > 3.0 or ball_state.y < 260):
                self._transition("DELIVERY")
                if not self.emitted_delivery:
                    self.emitted_delivery = True
                    emitted_event = CricketEvent(
                        match_id=self.match_id,
                        delivery_id=self.delivery_id,
                        frame_id=frame_id,
                        timestamp=timestamp,
                        video_timestamp=v_ts,
                        event_type="DELIVERY",
                        ball_speed_kmh=self.current_speed_kmh,
                        confidence=0.92,
                        metadata={"info": "Bowler releases delivery"}
                    )

        # STATE 2: DELIVERY
        elif self.current_state == "DELIVERY":
            self._transition("BALL_IN_PLAY")

        # STATE 3: BALL_IN_PLAY
        if self.current_state == "BALL_IN_PLAY":
            if ball_state:
                # Dynamically reference stumps and contact zone from batsman pose if detected
                active_stump_roi = self.stump_roi
                bat_contact_y = 260.0
                if batsman_pose and batsman_pose.keypoints:
                    kpts = batsman_pose.keypoints
                    ankles = [kpts[k].y for k in ["left_ankle", "right_ankle"] if k in kpts]
                    if ankles:
                        batsman_base_y = max(ankles)
                        bat_contact_y = max(180.0, batsman_base_y - 45.0)
                        ref_pt = kpts.get("left_ankle") or kpts.get("nose")
                        if ref_pt:
                            active_stump_roi = [int(ref_pt.x - 16), int(batsman_base_y - 30), int(ref_pt.x + 16), int(batsman_base_y + 10)]

                # Check for Wicket (Ball in stump ROI)
                ball_pt = (ball_state.x, ball_state.y)
                if is_point_in_roi(ball_pt, active_stump_roi) and not self.emitted_wicket:
                    self.emitted_wicket = True
                    self._transition("WICKET_DETECTED")
                    return CricketEvent(
                        match_id=self.match_id,
                        delivery_id=self.delivery_id,
                        frame_id=frame_id,
                        timestamp=timestamp,
                        video_timestamp=v_ts,
                        event_type="WICKET",
                        runs=0,
                        confidence=0.88,
                        metadata={"wicket_type": "BOWLED / LBW"}
                    )

                # Check for Bat Contact / Shot
                # Near batsman zone with trajectory inflection or horizontal velocity
                if len(trajectory) >= 6 and abs(ball_state.vx) > 3.5 and ball_state.y >= bat_contact_y:
                    self._transition("BAT_CONTACT")
                    if not self.emitted_shot:
                        self.emitted_shot = True
                        classification = self.shot_classifier.classify(
                            ball_trajectory=trajectory,
                            batsman_pose=batsman_pose
                        )
                        self.current_shot = classification
                        emitted_event = CricketEvent(
                            match_id=self.match_id,
                            delivery_id=self.delivery_id,
                            frame_id=frame_id,
                            timestamp=timestamp,
                            video_timestamp=v_ts,
                            event_type="SHOT_PLAYED",
                            shot_type=classification.shot_type,
                            ball_speed_kmh=self.current_speed_kmh,
                            confidence=classification.confidence,
                            metadata={"reasoning": classification.reasoning}
                        )

            # If ball disappeared or settled with no shot -> DOT_BALL
            if not ball_state and time.time() - self.state_entry_time > 1.5:
                self._transition("BALL_COMPLETE")
                emitted_event = CricketEvent(
                    match_id=self.match_id,
                    delivery_id=self.delivery_id,
                    frame_id=frame_id,
                    timestamp=timestamp,
                    video_timestamp=v_ts,
                    event_type="DOT_BALL",
                    runs=0,
                    confidence=0.90,
                    metadata={"reasoning": "Ball passed safely to keeper"}
                )

        # STATE 4: BAT_CONTACT
        elif self.current_state == "BAT_CONTACT":
            self._transition("BALL_TRAVEL")

        # STATE 5: BALL_TRAVEL
        elif self.current_state == "BALL_TRAVEL":
            if ball_state and len(trajectory) >= 2:
                curr_pt = (ball_state.x, ball_state.y)
                prev_pt = trajectory[-2]

                # Check boundary crossing
                if has_crossed_boundary(prev_pt, curr_pt, self.boundary_polygon) and not self.emitted_boundary:
                    self.emitted_boundary = True
                    self._transition("BOUNDARY_DETECTED")
                    
                    shot_type_str = self.current_shot.shot_type if self.current_shot else "PULL"
                    # If shot was lofted -> SIX, else ground stroke -> FOUR
                    is_six = (self.current_shot and self.current_shot.shot_type == "LOFTED_SHOT")
                    event_name = "SIX" if is_six else "FOUR"
                    runs_val = 6 if is_six else 4

                    return CricketEvent(
                        match_id=self.match_id,
                        delivery_id=self.delivery_id,
                        frame_id=frame_id,
                        timestamp=timestamp,
                        video_timestamp=v_ts,
                        event_type=event_name,
                        shot_type=shot_type_str,
                        runs=runs_val,
                        confidence=0.94,
                        metadata={"shot": shot_type_str, "exit_speed": self.current_speed_kmh}
                    )

            # If ball has stopped or stayed in outfield without reaching boundary (> 2.5s)
            if time.time() - self.state_entry_time > 2.5:
                self._transition("BALL_COMPLETE")
                if not self.emitted_boundary and not self.emitted_wicket:
                    # Normal field shot (Single / Double)
                    shot_str = self.current_shot.shot_type if self.current_shot else "DEFENSIVE"
                    runs = 0 if shot_str == "DEFENSIVE" else 1
                    event_name = "DOT_BALL" if runs == 0 else "SINGLE"
                    emitted_event = CricketEvent(
                        match_id=self.match_id,
                        delivery_id=self.delivery_id,
                        frame_id=frame_id,
                        timestamp=timestamp,
                        video_timestamp=v_ts,
                        event_type=event_name,
                        shot_type=shot_str,
                        runs=runs,
                        confidence=0.85
                    )

        # STATE 6: BOUNDARY_DETECTED
        elif self.current_state == "BOUNDARY_DETECTED":
            # Debounce hold: stay in boundary state for 1 second before wrapping ball
            if time.time() - self.state_entry_time > 1.2:
                self._transition("BALL_COMPLETE")

        # STATE 7: WICKET_DETECTED
        elif self.current_state == "WICKET_DETECTED":
            if time.time() - self.state_entry_time > 1.5:
                self._transition("BALL_COMPLETE")

        # STATE 8: BALL_COMPLETE
        elif self.current_state == "BALL_COMPLETE":
            self.ball_complete_frame_delay += 1
            if self.ball_complete_frame_delay > 20:  # ~0.6s cooldown between deliveries
                self.ball_complete_frame_delay = 0
                self._next_delivery()
                self._transition("IDLE")

        return emitted_event
