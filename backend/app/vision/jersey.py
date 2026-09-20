import cv2
import numpy as np
from typing import Optional, Tuple, Dict, Any, List
from sklearn.cluster import KMeans

class JerseyColorClassifier:
    """
    Player & Team Identification via Torso Jersey Color Extraction:
    - Extracts torso ROI from detected player bounding box or pose keypoints
    - Performs K-Means clustering to isolate dominant non-skin, non-grass jersey color
    - Maps color to international cricket team palettes
    """

    KNOWN_TEAM_PALETTES = [
        {"team": "Australia", "h_min": 18, "h_max": 38, "s_min": 80, "v_min": 120, "hex": "#FFD700"},
        {"team": "India", "h_min": 95, "h_max": 130, "s_min": 70, "v_min": 60, "hex": "#1E90FF"},
        {"team": "Pakistan / South Africa", "h_min": 35, "h_max": 85, "s_min": 60, "v_min": 40, "hex": "#006400"},
        {"team": "England", "h_min": 0, "h_max": 10, "s_min": 100, "v_min": 80, "hex": "#DC143C"},
        {"team": "West Indies", "h_min": 165, "h_max": 180, "s_min": 60, "v_min": 40, "hex": "#800000"},
        {"team": "Test Match (White)", "h_min": 0, "h_max": 180, "s_min": 0, "s_max": 40, "v_min": 170, "v_max": 255, "hex": "#F8F8FF"},
    ]

    @classmethod
    def extract_jersey_color(
        cls,
        frame: np.ndarray,
        bbox: List[float],
        keypoints: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, str, float]:
        """
        Extracts dominant jersey color from player torso.
        Returns (team_name, dominant_hex, confidence).
        """
        if frame is None or len(bbox) < 4:
            return "Unknown", "#808080", 0.0

        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = frame.shape[:2]
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w))
        y2 = max(y1 + 1, min(y2, h))

        # Sample torso: middle vertical third (y: 20% to 55%), middle horizontal 60%
        box_h = y2 - y1
        box_w = x2 - x1
        if box_h < 15 or box_w < 10:
            return "Unknown", "#808080", 0.0

        torso_y1 = y1 + int(box_h * 0.20)
        torso_y2 = y1 + int(box_h * 0.55)
        torso_x1 = x1 + int(box_w * 0.20)
        torso_x2 = x1 + int(box_w * 0.80)

        torso_crop = frame[torso_y1:torso_y2, torso_x1:torso_x2]
        if torso_crop.size == 0 or torso_crop.shape[0] < 4 or torso_crop.shape[1] < 4:
            return "Unknown", "#808080", 0.0

        # Convert to HSV
        hsv_crop = cv2.cvtColor(torso_crop, cv2.COLOR_BGR2HSV)
        pixels = hsv_crop.reshape(-1, 3)

        # Filter out background grass (H: 35-85, S > 40)
        valid_pixels = []
        for p in pixels:
            h_val, s_val, v_val = p[0], p[1], p[2]
            if 35 <= h_val <= 85 and s_val > 50:
                continue  # Background grass leak
            valid_pixels.append(p)

        if len(valid_pixels) < 16:
            valid_pixels = pixels

        valid_arr = np.array(valid_pixels, dtype=np.float32)

        try:
            # 2-cluster KMeans to separate jersey from shadows/collar
            kmeans = KMeans(n_clusters=min(2, len(valid_arr)), n_init=3, random_state=42)
            kmeans.fit(valid_arr)
            # Pick cluster with higher saturation or value
            centers = kmeans.cluster_centers_
            best_center = max(centers, key=lambda c: c[1] * 0.6 + c[2] * 0.4)
            h_dom, s_dom, v_dom = int(best_center[0]), int(best_center[1]), int(best_center[2])
        except Exception:
            median_hsv = np.median(valid_arr, axis=0)
            h_dom, s_dom, v_dom = int(median_hsv[0]), int(median_hsv[1]), int(median_hsv[2])

        # Convert dominant HSV to RGB Hex
        bgr_pixel = cv2.cvtColor(np.uint8([[[h_dom, s_dom, v_dom]]]), cv2.COLOR_HSV2BGR)[0][0]
        hex_color = f"#{int(bgr_pixel[2]):02x}{int(bgr_pixel[1]):02x}{int(bgr_pixel[0]):02x}"

        # Match against Known Team Palettes
        matched_team = "Club / Custom"
        matched_conf = 0.50

        # White / Test match check first
        if s_dom <= 45 and v_dom >= 140:
            return "Test Match (White)", hex_color, 0.88

        for pal in cls.KNOWN_TEAM_PALETTES:
            if "s_max" in pal and pal["team"].startswith("Test"):
                continue
            if pal["h_min"] <= h_dom <= pal["h_max"] and s_dom >= pal["s_min"] and v_dom >= pal["v_min"]:
                matched_team = pal["team"]
                matched_conf = 0.85
                break

        return matched_team, hex_color, matched_conf
