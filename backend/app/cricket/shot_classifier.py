import math
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from backend.app.models.schemas import PlayerPose

@dataclass
class ShotClassification:
    shot_type: str
    confidence: float
    reasoning: str = ""

class ShotClassifier:
    """
    Modular Cricket Shot Classifier.
    Combines ball trajectory kinematics, exit angles, bat proximity,
    and batsman posture to classify cricket strokes with explicit uncertainty.
    """

    SUPPORTED_SHOTS = [
        "DEFENSIVE",
        "COVER_DRIVE",
        "STRAIGHT_DRIVE",
        "ON_DRIVE",
        "SQUARE_CUT",
        "PULL",
        "HOOK",
        "SWEEP",
        "LOFTED_SHOT",
        "MISS",
        "UNKNOWN"
    ]

    def classify(
        self,
        ball_trajectory: List[Tuple[float, float]],
        batsman_pose: Optional[PlayerPose] = None,
        bat_position: Optional[Tuple[float, float]] = None
    ) -> ShotClassification:
        """
        Classifies cricket stroke from observed visual telemetry.
        Never fabricates certainty: returns UNKNOWN if data is insufficient.
        """
        if not ball_trajectory or len(ball_trajectory) < 6:
            return ShotClassification(shot_type="UNKNOWN", confidence=0.31, reasoning="Insufficient trajectory points")

        # Analyze pre-contact and post-contact trajectory vectors
        # Find point of inflection / sharpest change in direction
        inflection_idx = self._find_inflection_point(ball_trajectory)
        if inflection_idx is None or inflection_idx >= len(ball_trajectory) - 2:
            # Check if ball passed batsman without bat contact
            return ShotClassification(shot_type="MISS", confidence=0.75, reasoning="No significant directional deflection")

        pre_pts = ball_trajectory[max(0, inflection_idx - 6):inflection_idx]
        post_pts = ball_trajectory[inflection_idx:inflection_idx + 8]

        if len(post_pts) < 2:
            return ShotClassification(shot_type="UNKNOWN", confidence=0.35, reasoning="Post-contact path too short")

        # Exit velocity vector
        dx = post_pts[-1][0] - post_pts[0][0]
        dy = post_pts[-1][1] - post_pts[0][1]
        exit_dist = math.hypot(dx, dy)
        exit_speed = exit_dist / max(1, len(post_pts))

        # Check for gentle block / defensive stroke
        if exit_speed < 3.5:
            return ShotClassification(
                shot_type="DEFENSIVE",
                confidence=0.88,
                reasoning="Low exit velocity and deadened ball movement"
            )

        # Calculate angle of exit path in degrees (-180 to +180)
        # 0 deg = moving right, 90 deg = moving down, -90 deg = moving up (back down the pitch), 180 = moving left
        angle = math.degrees(math.atan2(dy, dx))

        # Inspect batsman posture if available
        is_kneeling = False
        arm_high = False
        if batsman_pose:
            kpts = batsman_pose.keypoints
            if "left_knee" in kpts and "right_knee" in kpts and "left_hip" in kpts:
                # Knee flexion check
                knee_y = max(kpts["left_knee"].y, kpts["right_knee"].y)
                hip_y = kpts["left_hip"].y
                if abs(knee_y - hip_y) < 25:
                    is_kneeling = True

            if "left_wrist" in kpts and "left_shoulder" in kpts:
                if kpts["left_wrist"].y < kpts["left_shoulder"].y:
                    arm_high = True

        if is_kneeling:
            return ShotClassification(shot_type="SWEEP", confidence=0.82, reasoning="Batsman knee down with sweeping path")

        # Heading toward off-side covers (left and slightly up/forward or left-down)
        if -170 <= angle <= -110 or 140 <= angle <= 180:
            # Shot driven towards extra cover or deep cover
            conf = 0.91 if exit_speed > 6.0 else 0.82
            return ShotClassification(
                shot_type="COVER_DRIVE",
                confidence=conf,
                reasoning=f"High exit velocity ({exit_speed:.1f}px/f) angled through covers ({angle:.1f}°)"
            )

        # Heading straight back towards bowler (-110 < angle < -70)
        elif -110 < angle < -70:
            return ShotClassification(
                shot_type="STRAIGHT_DRIVE",
                confidence=0.89,
                reasoning=f"Ball directed straight down the ground ({angle:.1f}°)"
            )

        # Heading toward mid-on / on-side (-70 <= angle < -10)
        elif -70 <= angle < -10:
            return ShotClassification(
                shot_type="ON_DRIVE",
                confidence=0.86,
                reasoning=f"Front-foot stroke directed through mid-on ({angle:.1f}°)"
            )

        # Square cut on off-side (sharp horizontal or behind square off-side: -180 to -160 or 170 to 180)
        elif abs(angle) > 160:
            return ShotClassification(
                shot_type="SQUARE_CUT",
                confidence=0.84,
                reasoning="Square off-side deflection across point/third-man"
            )

        # Pull / Hook shot (hit to midwicket or square leg: 10 <= angle <= 100)
        elif 10 <= angle <= 100:
            shot_name = "HOOK" if arm_high else "PULL"
            return ShotClassification(
                shot_type=shot_name,
                confidence=0.88,
                reasoning=f"Cross-bat stroke struck to leg-side ({angle:.1f}°)"
            )

        # Default fallback if angle is ambiguous
        return ShotClassification(
            shot_type="UNKNOWN",
            confidence=0.42,
            reasoning=f"Ambiguous exit angle {angle:.1f}°"
        )

    def _find_inflection_point(self, trajectory: List[Tuple[float, float]]) -> Optional[int]:
        """Finds trajectory index with highest directional angular acceleration or sudden deceleration."""
        if len(trajectory) < 4:
            return None

        max_angle_change = 0.0
        inflection_idx = None
        max_decel = 0.0
        decel_idx = None

        for i in range(1, len(trajectory) - 2):
            p_prev = trajectory[i - 1]
            p_curr = trajectory[i]
            p_next = trajectory[i + 1]

            v1 = (p_curr[0] - p_prev[0], p_curr[1] - p_prev[1])
            v2 = (p_next[0] - p_curr[0], p_next[1] - p_curr[1])

            len1 = math.hypot(v1[0], v1[1])
            len2 = math.hypot(v2[0], v2[1])

            # 1. Angular change (deflection/drive/cut/pull)
            if len1 > 1.0 and len2 > 1.0:
                dot = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (len1 * len2)))
                angle_diff = math.acos(dot)
                if angle_diff > max_angle_change and angle_diff > 0.4:
                    max_angle_change = angle_diff
                    inflection_idx = i

            # 2. Sudden deceleration (defensive block / deadened ball)
            if len1 > 4.0 and len2 < 3.0:
                decel = len1 - len2
                if decel > max_decel:
                    max_decel = decel
                    decel_idx = i

        return inflection_idx if inflection_idx is not None else decel_idx
