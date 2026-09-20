import math
import cv2
import numpy as np
from typing import List, Tuple, Optional

def is_point_inside_polygon(point: Tuple[float, float], polygon: List[List[int]]) -> bool:
    """
    Checks if point (x, y) is inside the polygon using pointPolygonTest.
    Returns True if inside or on boundary, False if outside.
    """
    if not polygon or len(polygon) < 3:
        return False
    pts = np.array(polygon, dtype=np.int32)
    # pointPolygonTest returns > 0 if inside, 0 if on edge, < 0 if outside
    res = cv2.pointPolygonTest(pts, (float(point[0]), float(point[1])), False)
    return res >= 0

def has_crossed_boundary(
    prev_point: Tuple[float, float],
    curr_point: Tuple[float, float],
    boundary_polygon: List[List[int]]
) -> bool:
    """
    Determines if the ball crossed from inside the boundary polygon to outside.
    """
    if not boundary_polygon or len(boundary_polygon) < 3:
        return False

    was_inside = is_point_inside_polygon(prev_point, boundary_polygon)
    is_inside = is_point_inside_polygon(curr_point, boundary_polygon)

    # Ball was inside the field and is now outside!
    return was_inside and not is_inside

def distance_to_polygon(point: Tuple[float, float], polygon: List[List[int]]) -> float:
    """
    Measures signed distance from point to polygon boundary.
    Positive inside, negative outside.
    """
    if not polygon or len(polygon) < 3:
        return 0.0
    pts = np.array(polygon, dtype=np.int32)
    return float(cv2.pointPolygonTest(pts, (float(point[0]), float(point[1])), True))

def is_point_in_roi(point: Tuple[float, float], roi: List[int]) -> bool:
    """
    Checks if point (x, y) is within a rectangular ROI [x1, y1, x2, y2].
    """
    if len(roi) < 4:
        return False
    x1, y1, x2, y2 = roi
    return x1 <= point[0] <= x2 and y1 <= point[1] <= y2

def calculate_angle_degrees(v1: Tuple[float, float], v2: Tuple[float, float]) -> float:
    """
    Calculates angle in degrees between two 2D vectors.
    """
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    det = v1[0] * v2[1] - v1[1] * v2[0]
    angle_rad = math.atan2(det, dot)
    return math.degrees(angle_rad)
