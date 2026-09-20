import math
import time
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class TrajectoryPoint:
    x: float
    y: float
    timestamp: float
    confidence: float
    is_interpolated: bool = False

class KalmanFilter2D:
    """
    Standard Linear 2D Kalman Filter for ball state estimation.
    State vector: [x, y, vx, vy]^T
    Measurement vector: [zx, zy]^T
    """

    def __init__(self, process_noise_std: float = 1.5, measurement_noise_std: float = 2.0):
        # State [x, y, vx, vy]^T
        self.state = np.zeros((4, 1), dtype=np.float64)
        
        # State covariance matrix P
        self.P = np.eye(4, dtype=np.float64) * 50.0
        
        # Measurement matrix H
        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ], dtype=np.float64)
        
        # Measurement noise covariance R
        self.R = np.eye(2, dtype=np.float64) * (measurement_noise_std ** 2)
        
        self.process_noise_std = process_noise_std
        self.last_timestamp: Optional[float] = None
        self.is_initialized = False

    def initialize(self, x: float, y: float, timestamp: float):
        self.state = np.array([[x], [y], [0.0], [0.0]], dtype=np.float64)
        self.P = np.eye(4, dtype=np.float64) * 10.0
        self.last_timestamp = timestamp
        self.is_initialized = True

    def predict(self, current_timestamp: float) -> Tuple[float, float, float, float]:
        """Runs the prediction step using dt = current_timestamp - last_timestamp."""
        if not self.is_initialized:
            return 0.0, 0.0, 0.0, 0.0

        dt = 0.033  # Default ~30 FPS
        if self.last_timestamp is not None and current_timestamp > self.last_timestamp:
            dt = min(current_timestamp - self.last_timestamp, 0.5)

        self.last_timestamp = current_timestamp

        # State transition matrix F
        F = np.array([
            [1.0, 0.0, dt,  0.0],
            [0.0, 1.0, 0.0, dt ],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float64)

        # Process noise covariance Q (discrete white noise model)
        q = self.process_noise_std ** 2
        dt4 = (dt ** 4) / 4.0
        dt3 = (dt ** 3) / 2.0
        dt2 = dt ** 2
        Q = np.array([
            [dt4 * q, 0.0,     dt3 * q, 0.0    ],
            [0.0,     dt4 * q, 0.0,     dt3 * q],
            [dt3 * q, 0.0,     dt2 * q, 0.0    ],
            [0.0,     dt3 * q, 0.0,     dt2 * q]
        ], dtype=np.float64)

        # Predict state and covariance
        self.state = F @ self.state
        self.P = (F @ self.P @ F.T) + Q

        return float(self.state[0, 0]), float(self.state[1, 0]), float(self.state[2, 0]), float(self.state[3, 0])

    def update(self, zx: float, zy: float) -> Tuple[float, float, float, float]:
        """Corrects state estimation with new sensor measurement."""
        if not self.is_initialized:
            self.initialize(zx, zy, time.time())
            return zx, zy, 0.0, 0.0

        z = np.array([[zx], [zy]], dtype=np.float64)
        
        # Innovation (measurement residual)
        y = z - (self.H @ self.state)
        
        # Innovation covariance
        S = (self.H @ self.P @ self.H.T) + self.R
        
        # Optimal Kalman Gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Updated state and covariance
        self.state = self.state + (K @ y)
        I = np.eye(4, dtype=np.float64)
        self.P = (I - (K @ self.H)) @ self.P

        return float(self.state[0, 0]), float(self.state[1, 0]), float(self.state[2, 0]), float(self.state[3, 0])


class TrajectoryBuffer:
    """
    Maintains a rolling window of ball trajectory points, provides smoothing,
    and calculates velocity / speed vectors with camera calibration.
    """

    def __init__(self, max_length: int = 40):
        self.max_length = max_length
        self.points: List[TrajectoryPoint] = []
        self.kalman = KalmanFilter2D()

    def reset(self):
        self.points.clear()
        self.kalman = KalmanFilter2D()

    def add_measurement(self, x: float, y: float, timestamp: float, confidence: float) -> Tuple[float, float, float, float]:
        if not self.kalman.is_initialized:
            self.kalman.initialize(x, y, timestamp)
            sx, sy, vx, vy = x, y, 0.0, 0.0
        else:
            self.kalman.predict(timestamp)
            sx, sy, vx, vy = self.kalman.update(x, y)

        point = TrajectoryPoint(x=sx, y=sy, timestamp=timestamp, confidence=confidence, is_interpolated=False)
        self.points.append(point)
        if len(self.points) > self.max_length:
            self.points.pop(0)

        return sx, sy, vx, vy

    def add_prediction_only(self, timestamp: float) -> Tuple[float, float, float, float]:
        """Used during temporary missed detections/occlusions."""
        if not self.kalman.is_initialized:
            return 0.0, 0.0, 0.0, 0.0

        sx, sy, vx, vy = self.kalman.predict(timestamp)
        # Lower confidence for interpolated predictions
        last_conf = self.points[-1].confidence if self.points else 0.5
        point = TrajectoryPoint(x=sx, y=sy, timestamp=timestamp, confidence=last_conf * 0.85, is_interpolated=True)
        self.points.append(point)
        if len(self.points) > self.max_length:
            self.points.pop(0)

        return sx, sy, vx, vy

    def get_smoothed_points(self) -> List[Tuple[float, float]]:
        return [(p.x, p.y) for p in self.points]

    def estimate_speed_kmh(self, pixels_per_meter: Optional[float] = None, speed_estimation_available: bool = False) -> Tuple[Optional[float], bool]:
        """
        Calculates speed in km/h based on trajectory and camera calibration.
        If camera calibration is not verified, returns (None, False) to prevent fabricating metrics.
        """
        if not speed_estimation_available or not pixels_per_meter or pixels_per_meter <= 0:
            return None, False

        if len(self.points) < 4:
            return None, False

        # Average velocity over recent 4-6 frames
        recent = self.points[-5:]
        dt = recent[-1].timestamp - recent[0].timestamp
        if dt <= 0.01:
            return None, False

        dx = recent[-1].x - recent[0].x
        dy = recent[-1].y - recent[0].y
        dist_px = math.hypot(dx, dy)
        dist_meters = dist_px / pixels_per_meter
        speed_mps = dist_meters / dt
        speed_kmh = round(speed_mps * 3.6, 1)

        # Sanity bound for cricket ball (slow spin/push 10 km/h up to express fast bowling 160+ km/h)
        if 5.0 <= speed_kmh <= 180.0:
            return speed_kmh, True

        return None, False
