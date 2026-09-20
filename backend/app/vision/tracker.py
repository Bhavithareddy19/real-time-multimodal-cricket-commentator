import math
import time
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict

from backend.app.models.schemas import Detection, BallState
from backend.app.vision.trajectory import TrajectoryBuffer

class BallTracker:
    """
    Robust Cricket Ball Tracker using Kalman Filtering, Trajectory Smoothing,
    and Missing-Frame Occlusion Recovery.
    """

    def __init__(
        self,
        max_missed_frames: int = 6,
        max_association_dist: float = 85.0,
        pixels_per_meter: Optional[float] = 24.5,
        speed_estimation_available: bool = False
    ):
        self.trajectory_buffer = TrajectoryBuffer(max_length=45)
        self.max_missed_frames = max_missed_frames
        self.max_association_dist = max_association_dist
        self.pixels_per_meter = pixels_per_meter
        self.speed_estimation_available = speed_estimation_available

        self.missed_frames = 0
        self.current_state: Optional[BallState] = None
        self.is_tracking = False

    def reset(self):
        self.trajectory_buffer.reset()
        self.missed_frames = 0
        self.current_state = None
        self.is_tracking = False

    def update(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        timestamp: float
    ) -> Optional[BallState]:
        """
        Processes detections to identify and update the ball track.
        If YOLO misses the ball, utilizes Kalman prediction and motion/color heuristics.
        """
        # 1. Search for explicit ball detections from YOLO with physical size constraints
        h_frame, w_frame = frame.shape[:2]
        max_ball_dim = max(18.0, min(w_frame, h_frame) * 0.08)  # Ball should not exceed 8% of frame dimension

        ball_candidates = []
        for d in detections:
            if d.class_name == "sports ball" or "ball" in d.class_name:
                bw = d.bbox[2] - d.bbox[0]
                bh = d.bbox[3] - d.bbox[1]
                # Reject boxes that are too large (e.g. players' torsos, helmets misclassified by COCO)
                if bw <= max_ball_dim and bh <= max_ball_dim:
                    aspect = max(bw, bh) / max(1.0, min(bw, bh))
                    if aspect <= 2.8:  # Ball or motion-blurred streak
                        ball_candidates.append(d)

        best_candidate: Optional[Tuple[float, float, float]] = None

        if ball_candidates:
            # Pick candidate with highest confidence or closest to predicted position
            if self.is_tracking and self.current_state:
                min_d = float('inf')
                for c in ball_candidates:
                    cx = (c.bbox[0] + c.bbox[2]) / 2.0
                    cy = (c.bbox[1] + c.bbox[3]) / 2.0
                    d = math.hypot(cx - self.current_state.x, cy - self.current_state.y)
                    if d < min_d and d <= self.max_association_dist * 2.0:
                        min_d = d
                        best_candidate = (cx, cy, c.confidence)
            if not best_candidate:
                # Default to candidate with best confidence
                c = max(ball_candidates, key=lambda x: x.confidence)
                best_candidate = ((c.bbox[0] + c.bbox[2]) / 2.0, (c.bbox[1] + c.bbox[3]) / 2.0, c.confidence)

        # 2. Color/Motion Candidate Fallback when YOLO misses small ball on pitch
        if not best_candidate:
            if self.is_tracking and self.current_state:
                # Search a local ROI around predicted position
                pred_x = self.current_state.x + self.current_state.vx * 0.033
                pred_y = self.current_state.y + self.current_state.vy * 0.033
                rx1 = max(0, int(pred_x - 45))
                ry1 = max(0, int(pred_y - 45))
                rx2 = min(w_frame, int(pred_x + 45))
                ry2 = min(h_frame, int(pred_y + 45))
                max_allowed_dist = self.max_association_dist
            else:
                # Search central pitch/corridor area to acquire initial ball track
                rx1 = max(0, int(w_frame * 0.20))
                ry1 = max(0, int(h_frame * 0.15))
                rx2 = min(w_frame, int(w_frame * 0.80))
                ry2 = min(h_frame, int(h_frame * 0.85))
                pred_x = (rx1 + rx2) / 2.0
                pred_y = (ry1 + ry2) / 2.0
                max_allowed_dist = max(w_frame, h_frame)

            if rx2 > rx1 and ry2 > ry1:
                roi = frame[ry1:ry2, rx1:rx2]
                hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                
                # Cricket ball color filter:
                # A) Red leather ball (Test cricket):
                mask_red1 = cv2.inRange(hsv, np.array([0, 70, 40]), np.array([12, 255, 255]))
                mask_red2 = cv2.inRange(hsv, np.array([168, 70, 40]), np.array([180, 255, 255]))
                # B) White/Pink ball (Limited overs/Night cricket):
                mask_white = cv2.inRange(hsv, np.array([0, 0, 185]), np.array([180, 50, 255]))
                ball_mask = mask_red1 | mask_red2 | mask_white

                contours, _ = cv2.findContours(ball_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                valid_blobs = []
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if 4 <= area <= 200:  # Ball blob size range
                        (bx, by), radius = cv2.minEnclosingCircle(cnt)
                        global_x = rx1 + bx
                        global_y = ry1 + by
                        dist = math.hypot(global_x - pred_x, global_y - pred_y)
                        valid_blobs.append((dist, global_x, global_y))

                if valid_blobs:
                    valid_blobs.sort(key=lambda x: x[0])
                    if valid_blobs[0][0] <= max_allowed_dist:
                        best_candidate = (valid_blobs[0][1], valid_blobs[0][2], 0.65)

        # 3. Update or Predict
        if best_candidate:
            cx, cy, conf = best_candidate
            sx, sy, vx, vy = self.trajectory_buffer.add_measurement(cx, cy, timestamp, conf)
            self.missed_frames = 0
            self.is_tracking = True

            speed_kmh, speed_avail = self.trajectory_buffer.estimate_speed_kmh(
                self.pixels_per_meter,
                self.speed_estimation_available
            )

            self.current_state = BallState(
                x=round(sx, 1),
                y=round(sy, 1),
                vx=round(vx, 1),
                vy=round(vy, 1),
                confidence=round(conf, 2),
                timestamp=timestamp,
                speed_kmh=speed_kmh,
                speed_estimation_available=speed_avail
            )
        elif self.is_tracking:
            self.missed_frames += 1
            if self.missed_frames <= self.max_missed_frames:
                # Prediction during occlusion
                sx, sy, vx, vy = self.trajectory_buffer.add_prediction_only(timestamp)
                last_conf = self.current_state.confidence * 0.8 if self.current_state else 0.4
                speed_kmh, speed_avail = self.trajectory_buffer.estimate_speed_kmh(
                    self.pixels_per_meter,
                    self.speed_estimation_available
                )
                self.current_state = BallState(
                    x=round(sx, 1),
                    y=round(sy, 1),
                    vx=round(vx, 1),
                    vy=round(vy, 1),
                    confidence=round(last_conf, 2),
                    timestamp=timestamp,
                    speed_kmh=speed_kmh,
                    speed_estimation_available=speed_avail
                )
            else:
                # Lost track
                self.is_tracking = False
                self.current_state = None

        return self.current_state

    def get_trajectory_points(self) -> List[Tuple[float, float]]:
        return self.trajectory_buffer.get_smoothed_points()

    def draw_trajectory(self, frame: np.ndarray) -> np.ndarray:
        """Draws glowing comet-style ball trajectory trail onto the frame."""
        annotated = frame.copy()
        pts = self.trajectory_buffer.points
        n = len(pts)
        if n < 2:
            return annotated

        for i in range(1, n):
            pt1 = (int(pts[i - 1].x), int(pts[i - 1].y))
            pt2 = (int(pts[i].x), int(pts[i].y))
            progress = i / float(n)  # 0.0 (oldest) to 1.0 (newest)
            
            # Color gradient: Cyan/Yellow fading into bright Orange/Red
            b = int(255 * (1.0 - progress))
            g = int(180 * progress)
            r = int(255 * progress)
            thickness = max(1, int(1 + progress * 3))

            cv2.line(annotated, pt1, pt2, (b, g, r), thickness, cv2.LINE_AA)
            if i == n - 1 and self.current_state:
                # Draw leading ball glow
                curr_pt = (int(self.current_state.x), int(self.current_state.y))
                cv2.circle(annotated, curr_pt, 7, (0, 165, 255), 2, cv2.LINE_AA)
                cv2.circle(annotated, curr_pt, 4, (0, 0, 255), -1, cv2.LINE_AA)

        return annotated


class PlayerTracker:
    """
    Tracks player centroids and associates persistent IDs across frames.
    """

    def __init__(self, max_distance: float = 60.0):
        self.max_distance = max_distance
        self.tracks: Dict[int, Tuple[float, float, float]] = {}  # id -> (x, y, last_time)
        self.next_id = 1

    def update(self, detections: List[Detection], timestamp: float) -> List[Detection]:
        persons = [d for d in detections if d.class_name == "person"]
        updated: List[Detection] = []

        for p in persons:
            cx = (p.bbox[0] + p.bbox[2]) / 2.0
            cy = (p.bbox[1] + p.bbox[3]) / 2.0

            # Find matching track
            best_id = None
            min_dist = float('inf')
            for tid, (tx, ty, ttime) in self.tracks.items():
                d = math.hypot(cx - tx, cy - ty)
                if d < min_dist and d < self.max_distance:
                    min_dist = d
                    best_id = tid

            if best_id is None:
                best_id = self.next_id
                self.next_id += 1

            self.tracks[best_id] = (cx, cy, timestamp)
            p.track_id = best_id
            updated.append(p)

        # Cleanup old tracks (> 3.0 sec)
        to_delete = [tid for tid, (_, _, ttime) in self.tracks.items() if timestamp - ttime > 3.0]
        for tid in to_delete:
            del self.tracks[tid]

        return updated
