import asyncio
import time
import logging
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from ultralytics import YOLO

from backend.app.config import get_settings
from backend.app.models.schemas import PlayerPose, PoseKeypoint

logger = logging.getLogger("vision.pose")

class PoseEngine:
    """
    Player Pose Estimation Engine using YOLOv8-pose.
    Detects 17 skeletal landmarks and evaluates batsman/bowler posture.
    """

    KEYPOINT_NAMES = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]

    SKELETON_PAIRS = [
        ("left_shoulder", "right_shoulder"),
        ("left_shoulder", "left_hip"),
        ("right_shoulder", "right_hip"),
        ("left_hip", "right_hip"),
        ("left_shoulder", "left_elbow"),
        ("left_elbow", "left_wrist"),
        ("right_shoulder", "right_elbow"),
        ("right_elbow", "right_wrist"),
        ("left_hip", "left_knee"),
        ("left_knee", "left_ankle"),
        ("right_hip", "right_knee"),
        ("right_knee", "right_ankle")
    ]

    def __init__(self, model_path: Optional[str] = None):
        self.settings = get_settings()
        self.device = self.settings.get_resolved_device()
        self.model_path = model_path or self.settings.YOLO_POSE_MODEL
        self.model: Optional[YOLO] = None
        self._is_loaded = False

    def load_model(self):
        if self._is_loaded:
            return
        logger.info(f"Loading YOLO Pose model '{self.model_path}' on device '{self.device}'...")
        try:
            self.model = YOLO(self.model_path)
            dummy = np.zeros((360, 640, 3), dtype=np.uint8)
            self.model(dummy, device=self.device, verbose=False)
            self._is_loaded = True
            logger.info("YOLO Pose model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load YOLO Pose model: {e}")
            raise

    def _sync_estimate(self, frame: np.ndarray, conf_threshold: float = 0.3) -> Tuple[List[PlayerPose], float]:
        if not self._is_loaded:
            self.load_model()

        start_time = time.perf_counter()
        results = self.model.predict(
            source=frame,
            conf=conf_threshold,
            device=self.device,
            verbose=False,
            imgsz=384
        )
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        poses: List[PlayerPose] = []
        if results and len(results) > 0 and results[0].keypoints is not None:
            kpts_data = results[0].keypoints.data  # shape: (N, 17, 3) or (N, 17, 2)
            for idx, person_kpts in enumerate(kpts_data):
                kpts_dict: Dict[str, PoseKeypoint] = {}
                person_arr = person_kpts.cpu().numpy()
                total_conf = 0.0
                valid_count = 0

                for i, name in enumerate(self.KEYPOINT_NAMES):
                    if i < len(person_arr):
                        kx = float(person_arr[i][0])
                        ky = float(person_arr[i][1])
                        kc = float(person_arr[i][2]) if person_arr.shape[1] > 2 else 0.8
                        if kc > 0.2:
                            kpts_dict[name] = PoseKeypoint(x=round(kx, 1), y=round(ky, 1), confidence=round(kc, 2))
                            total_conf += kc
                            valid_count += 1

                avg_conf = (total_conf / valid_count) if valid_count > 0 else 0.0
                
                # Determine role based on vertical position or keypoint configuration
                role = "unknown"
                if "left_wrist" in kpts_dict and "right_wrist" in kpts_dict:
                    # In lower half of frame: likely batsman
                    if kpts_dict["left_wrist"].y > frame.shape[0] * 0.45:
                        role = "batsman"
                    else:
                        role = "bowler"

                poses.append(PlayerPose(
                    track_id=idx + 1,
                    role=role,
                    confidence=round(avg_conf, 2),
                    keypoints=kpts_dict
                ))

        return poses, latency_ms

    async def estimate(self, frame: np.ndarray, conf_threshold: float = 0.3) -> Tuple[List[PlayerPose], float]:
        return await asyncio.to_thread(self._sync_estimate, frame, conf_threshold)

    def draw_skeletons(self, frame: np.ndarray, poses: List[PlayerPose]) -> np.ndarray:
        """Renders pose skeletons and keypoints onto the frame."""
        annotated = frame.copy()

        for p in poses:
            color = (0, 255, 255) if p.role == "batsman" else (255, 105, 180)  # Yellow for batsman, Pink for bowler

            # Draw bones
            for pt1_name, pt2_name in self.SKELETON_PAIRS:
                if pt1_name in p.keypoints and pt2_name in p.keypoints:
                    pt1 = (int(p.keypoints[pt1_name].x), int(p.keypoints[pt1_name].y))
                    pt2 = (int(p.keypoints[pt2_name].x), int(p.keypoints[pt2_name].y))
                    cv2.line(annotated, pt1, pt2, color, 2, cv2.LINE_AA)

            # Draw keypoint joints
            for name, kp in p.keypoints.items():
                center = (int(kp.x), int(kp.y))
                cv2.circle(annotated, center, 3, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(annotated, center, 4, color, 1, cv2.LINE_AA)

            # Label player role
            if "nose" in p.keypoints:
                nx = int(p.keypoints["nose"].x)
                ny = int(p.keypoints["nose"].y) - 12
                label = f"{p.role.upper()} #{p.track_id}"
                cv2.putText(annotated, label, (nx - 20, ny), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

        return annotated
