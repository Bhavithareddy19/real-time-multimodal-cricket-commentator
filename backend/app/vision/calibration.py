import os
import json
import logging
import cv2
import numpy as np
from typing import Tuple, List, Optional, Dict, Any

logger = logging.getLogger("vision.calibration")

class AutoPitchCalibrator:
    """
    Autonomous Camera & Pitch Calibration using Computer Vision:
    - Outfield grass segmentation (HSV green mask)
    - Central turf pitch isolation
    - Popping & bowling crease line detection via HoughLinesP
    - Dynamic boundary polygon and stump ROI generation
    - Graceful fallback to static calibration when confidence is low
    """

    DEFAULT_BOUNDARY_POLYGON = [
        [40, 330], [180, 190], [460, 190], [600, 330], [480, 355], [160, 355]
    ]
    DEFAULT_STUMP_ROI = [310, 295, 330, 330]  # [x1, y1, x2, y2]

    def __init__(self, fallback_config_path: str = "config/calibration.json"):
        self.fallback_config_path = fallback_config_path
        self.boundary_polygon: List[List[int]] = list(self.DEFAULT_BOUNDARY_POLYGON)
        self.stump_roi: List[int] = list(self.DEFAULT_STUMP_ROI)
        self.pitch_polygon: Optional[List[List[int]]] = None
        self.calibration_confidence: float = 0.0
        self.is_calibrated: bool = False
        self.load_fallback_config()

    def load_fallback_config(self):
        if os.path.exists(self.fallback_config_path):
            try:
                with open(self.fallback_config_path, "r") as f:
                    data = json.load(f)
                    if "boundary_polygon" in data and len(data["boundary_polygon"]) >= 3:
                        self.boundary_polygon = data["boundary_polygon"]
                    if "stump_roi" in data and len(data["stump_roi"]) == 4:
                        self.stump_roi = data["stump_roi"]
            except Exception as e:
                logger.warning(f"Failed to load fallback calibration config: {e}")

    def calibrate_from_frame(self, frame: np.ndarray) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Analyzes a single keyframe or background image to auto-detect pitch and boundary geometry.
        Returns (is_success, confidence, calibration_details).
        """
        if frame is None or frame.size == 0:
            return False, 0.0, {}

        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 1. Detect Outfield Grass Mask (Green spectrum)
        lower_green = np.array([30, 35, 30])
        upper_green = np.array([88, 255, 255])
        grass_mask = cv2.inRange(hsv, lower_green, upper_green)

        # Morphological opening/closing to smooth grass regions
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        grass_clean = cv2.morphologyEx(grass_mask, cv2.MORPH_OPEN, kernel)
        grass_clean = cv2.morphologyEx(grass_clean, cv2.MORPH_CLOSE, kernel)

        grass_ratio = cv2.countNonZero(grass_clean) / float(w * h)

        # If image has insufficient grass (<15%), camera is not showing a cricket ground
        if grass_ratio < 0.15:
            logger.info(f"AutoPitchCalibrator: low grass ratio ({grass_ratio:.2f}), retaining fallback geometry.")
            return False, 0.20, {"reason": "Low turf coverage"}

        # 2. Extract Central Non-Grass Patch (The Turf Pitch)
        # Pitch is generally non-green or brownish inside the grass region
        pitch_candidate_mask = cv2.bitwise_not(grass_clean)
        # Limit to central vertical strip (x between 25% and 75% of frame width)
        central_strip = np.zeros_like(pitch_candidate_mask)
        x_min, x_max = int(w * 0.20), int(w * 0.80)
        y_min, y_max = int(h * 0.25), int(h * 0.95)
        central_strip[y_min:y_max, x_min:x_max] = 255
        pitch_mask = cv2.bitwise_and(pitch_candidate_mask, central_strip)

        contours, _ = cv2.findContours(pitch_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_pitch_contour = None
        max_area = 0
        min_pitch_area = (w * h) * 0.02
        max_pitch_area = (w * h) * 0.35

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_pitch_area <= area <= max_pitch_area:
                # Must be vertically elongated (pitch is longer than wide)
                bx, by, bw, bh = cv2.boundingRect(cnt)
                if bh > bw * 1.1:
                    if area > max_area:
                        max_area = area
                        best_pitch_contour = cnt

        # 3. Detect White Crease Lines
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40, minLineLength=int(w * 0.08), maxLineGap=15)

        detected_crease_y = None
        if lines is not None:
            # Find horizontal lines in the lower half of the pitch
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Horizontal line: |slope| < 0.2
                if abs(y2 - y1) < 8 and y1 > h * 0.50:
                    detected_crease_y = (y1 + y2) / 2.0
                    break

        # 4. Synthesize Geometry & Score Confidence
        conf = 0.50
        if best_pitch_contour is not None:
            conf += 0.30
            # Approximate pitch to a polygon
            epsilon = 0.04 * cv2.arcLength(best_pitch_contour, True)
            approx = cv2.approxPolyDP(best_pitch_contour, epsilon, True)
            self.pitch_polygon = [pt[0].tolist() for pt in approx]

            # Dynamic stump ROI placement based on detected pitch
            bx, by, bw, bh = cv2.boundingRect(best_pitch_contour)
            crease_y = detected_crease_y or (by + bh * 0.85)
            stump_w = max(16, int(bw * 0.15))
            stump_cx = bx + bw // 2
            self.stump_roi = [
                int(stump_cx - stump_w // 2),
                int(crease_y - 25),
                int(stump_cx + stump_w // 2),
                int(crease_y + 10)
            ]
        else:
            # Calibrate relative to frame size
            cx = w // 2
            self.stump_roi = [int(cx - 15), int(h * 0.65), int(cx + 15), int(h * 0.72)]

        if detected_crease_y is not None:
            conf += 0.15

        # 5. Extract Outer Field Boundary from Grass Hull
        # Cricket boundary rope is always inset inside the camera frame
        grass_contours, _ = cv2.findContours(grass_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if grass_contours:
            largest_grass = max(grass_contours, key=cv2.contourArea)
            hull = cv2.convexHull(largest_grass)
            epsilon = 0.035 * cv2.arcLength(hull, True)
            approx_hull = cv2.approxPolyDP(hull, epsilon, True)
            if len(approx_hull) >= 4:
                # Inset 12% towards frame center so ball crossing boundary can be detected
                cx, cy = w / 2.0, h / 2.0
                inset_poly = []
                for pt in approx_hull:
                    px, py = pt[0]
                    ix = int(px + (cx - px) * 0.12)
                    iy = int(py + (cy - py) * 0.12)
                    inset_poly.append([ix, iy])
                self.boundary_polygon = inset_poly
                conf += 0.05

        self.calibration_confidence = min(0.95, round(conf, 2))
        self.is_calibrated = (self.calibration_confidence >= 0.60)

        logger.info(f"AutoPitchCalibrator complete: confidence={self.calibration_confidence:.2f}, is_calibrated={self.is_calibrated}")
        return self.is_calibrated, self.calibration_confidence, {
            "boundary_polygon": self.boundary_polygon,
            "stump_roi": self.stump_roi,
            "pitch_polygon": self.pitch_polygon,
            "confidence": self.calibration_confidence
        }
