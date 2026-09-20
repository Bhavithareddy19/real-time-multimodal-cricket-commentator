import asyncio
import time
import logging
import cv2
import numpy as np
from typing import List, Tuple, Optional
from ultralytics import YOLO

from backend.app.config import get_settings
from backend.app.models.schemas import Detection, BallState, PlayerPose
from backend.app.vision.tracker import BallTracker, PlayerTracker
from backend.app.vision.pose import PoseEngine
from backend.app.vision.jersey import JerseyColorClassifier

logger = logging.getLogger("vision.detector")

class VisionEngine:
    """
    Unified Computer Vision Engine integrating:
    - YOLOv8 Object Detection (person, ball, bat)
    - 2D Kalman Filter Ball Tracking & Trajectory Estimation
    - Player Centroid/IoU Tracking
    - YOLOv8-pose Skeletal Landmark Estimation
    """

    CRICKET_CLASS_MAP = {
        0: "person",
        32: "sports ball",
        34: "bat",
    }

    def __init__(self, model_path: Optional[str] = None):
        self.settings = get_settings()
        self.device = self.settings.get_resolved_device()
        self.model_path = model_path or self.settings.YOLO_MODEL
        self.model: Optional[YOLO] = None
        self._is_loaded = False
        self._warmup_done = False

        # Phase 2 components
        self.ball_tracker = BallTracker()
        self.player_tracker = PlayerTracker()
        self.pose_engine = PoseEngine()

    def load_model(self):
        """Loads and warms up YOLO detection and pose models."""
        if self._is_loaded:
            return

        logger.info(f"Loading YOLO model '{self.model_path}' on device '{self.device}'...")
        try:
            self.model = YOLO(self.model_path)
            dummy_img = np.zeros((360, 640, 3), dtype=np.uint8)
            self.model(dummy_img, device=self.device, verbose=False)
            self._is_loaded = True
            self._warmup_done = True
            logger.info(f"YOLO model loaded and warmed up successfully on {self.device}.")
            
            # Also warmup pose engine
            self.pose_engine.load_model()
        except Exception as e:
            logger.error(f"Failed to load YOLO model {self.model_path}: {e}")
            raise

    def reset_tracking(self):
        self.ball_tracker.reset()

    def _sync_detect(self, frame: np.ndarray, conf_threshold: float = 0.25) -> Tuple[List[Detection], float]:
        if not self._is_loaded:
            self.load_model()

        start_time = time.perf_counter()
        
        # Classes: 0 (person), 32 (sports ball), 34 (bat)
        results = self.model.predict(
            source=frame,
            classes=[0, 32, 34],
            conf=conf_threshold,
            device=self.device,
            verbose=False,
            imgsz=480
        )

        inference_time_ms = (time.perf_counter() - start_time) * 1000.0

        detections: List[Detection] = []
        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None:
                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    xyxy = box.xyxy[0].tolist()
                    class_name = self.CRICKET_CLASS_MAP.get(cls_id, self.model.names.get(cls_id, f"obj_{cls_id}"))

                    detections.append(Detection(
                        class_id=cls_id,
                        class_name=class_name,
                        confidence=round(conf, 3),
                        bbox=[round(coord, 1) for coord in xyxy]
                    ))

        return detections, inference_time_ms

    async def detect(self, frame: np.ndarray, conf_threshold: float = 0.25) -> Tuple[List[Detection], float]:
        return await asyncio.to_thread(self._sync_detect, frame, conf_threshold)

    async def process_frame_tracking(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        timestamp: float,
        run_pose: bool = True
    ) -> Tuple[Optional[BallState], List[Tuple[float, float]], List[PlayerPose], float]:
        """
        Executes Ball Tracking, Player Tracking, and Pose Estimation concurrently.
        """
        start_t = time.perf_counter()

        # Update player tracks
        updated_dets = self.player_tracker.update(detections, timestamp)

        # Extract jersey colors and attribute teams for players
        for d in updated_dets:
            if d.class_name == "person":
                team, hex_col, _ = JerseyColorClassifier.extract_jersey_color(frame, d.bbox)
                d.team = team
                d.jersey_color = hex_col

        # Update ball Kalman filter and trajectory
        ball_state = self.ball_tracker.update(frame, updated_dets, timestamp)
        trajectory = self.ball_tracker.get_trajectory_points()

        # Run pose estimation
        poses: List[PlayerPose] = []
        if run_pose:
            # Downsample pose frequency or run asynchronously
            poses, _ = await self.pose_engine.estimate(frame, conf_threshold=0.35)

        total_latency_ms = (time.perf_counter() - start_t) * 1000.0
        return ball_state, trajectory, poses, total_latency_ms

    def draw_overlays(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        ball_state: Optional[BallState] = None,
        trajectory: Optional[List[Tuple[float, float]]] = None,
        poses: Optional[List[PlayerPose]] = None,
        fps: float = 0.0,
        latency_ms: float = 0.0,
        current_event: Optional[str] = None,
        boundary_polygon: Optional[List[List[int]]] = None
    ) -> np.ndarray:
        """
        Comprehensive sports broadcast overlay rendering:
        - Player & Equipment bounding boxes
        - Skeletons and keypoints
        - Ball trajectory trail and leading glow
        - Boundary perimeter
        - HUD Banner with measured FPS and CV latency
        """
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        # 1. Boundary Polygon
        if boundary_polygon and len(boundary_polygon) >= 3:
            pts = np.array(boundary_polygon, np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated, [pts], isClosed=True, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

        # 2. Pose Skeletons
        if poses:
            annotated = self.pose_engine.draw_skeletons(annotated, poses)

        # 3. Object Bounding Boxes
        color_map = {
            "person": (245, 147, 66),
            "sports ball": (0, 71, 255),
            "bat": (50, 205, 50),
        }

        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det.bbox]
            color = color_map.get(det.class_name, (200, 200, 200))
            thickness = 2 if det.class_name == "sports ball" else 1
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

            track_info = f" #{det.track_id}" if det.track_id else ""
            team_info = f" [{det.team}]" if det.team and det.team not in ["Unknown", "Club / Custom"] else ""
            label = f"{det.class_name.upper()}{track_info}{team_info} {int(det.confidence * 100)}%"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            cv2.rectangle(annotated, (x1, max(0, y1 - lh - 6)), (x1 + lw + 6, max(lh + 6, y1)), color, cv2.FILLED)
            cv2.putText(annotated, label, (x1 + 3, max(lh + 1, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)

        # 4. Ball Trajectory Comet Trail
        annotated = self.ball_tracker.draw_trajectory(annotated)

        # 5. Header Broadcast HUD Banner
        hud_h = 36
        sub_img = annotated[0:hud_h, 0:w]
        dark_rect = np.zeros(sub_img.shape, dtype=np.uint8)
        cv2.addWeighted(sub_img, 0.35, dark_rect, 0.65, 0, sub_img)

        hud_text = f"FPS: {fps:4.1f} | CV: {latency_ms:4.1f}ms"
        if ball_state:
            hud_text += f" | BALL: ({int(ball_state.x)},{int(ball_state.y)})"
            if ball_state.speed_kmh and ball_state.speed_estimation_available:
                hud_text += f" {ball_state.speed_kmh}km/h"
        cv2.putText(annotated, hud_text, (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 200), 1, cv2.LINE_AA)

        if current_event:
            ev_text = f"EVENT: {current_event}"
            cv2.putText(annotated, ev_text, (w - 230, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 215, 255), 2, cv2.LINE_AA)

        return annotated
