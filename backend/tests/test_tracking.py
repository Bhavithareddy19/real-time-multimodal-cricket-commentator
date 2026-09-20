import time
import pytest
import cv2
import numpy as np
from backend.app.vision.trajectory import KalmanFilter2D, TrajectoryBuffer
from backend.app.vision.tracker import BallTracker, PlayerTracker
from backend.app.vision.pose import PoseEngine
from backend.app.models.schemas import Detection

def test_kalman_filter_prediction_and_update():
    kf = KalmanFilter2D(process_noise_std=1.0, measurement_noise_std=1.0)
    assert not kf.is_initialized

    t0 = time.time()
    kf.initialize(100.0, 200.0, t0)
    assert kf.is_initialized

    # Predict after 0.033s
    px, py, vx, vy = kf.predict(t0 + 0.033)
    assert abs(px - 100.0) < 1.0
    assert abs(py - 200.0) < 1.0

    # Update with measurement moving in x direction
    ux, uy, uvx, uvy = kf.update(105.0, 200.0)
    assert ux > 100.0
    assert uvx > 0.0

def test_trajectory_buffer_and_speed_estimation():
    tb = TrajectoryBuffer(max_length=20)
    t = 1000.0
    
    # Simulate a ball moving 5 pixels per frame at 30 FPS
    for i in range(10):
        tb.add_measurement(x=50.0 + i * 5.0, y=100.0, timestamp=t + i * 0.0333, confidence=0.9)
    
    points = tb.get_smoothed_points()
    assert len(points) == 10
    
    # When uncalibrated or speed_estimation_available=False, must return (None, False)
    speed, avail = tb.estimate_speed_kmh(pixels_per_meter=25.0, speed_estimation_available=False)
    assert speed is None
    assert avail is False

    # When calibrated and enabled
    speed, avail = tb.estimate_speed_kmh(pixels_per_meter=25.0, speed_estimation_available=True)
    assert avail is True
    assert speed is not None
    assert 10.0 <= speed <= 100.0  # realistic physical speed

def test_ball_tracker_occlusion_handling():
    tracker = BallTracker(max_missed_frames=5)
    dummy_frame = np.zeros((360, 640, 3), dtype=np.uint8)
    t0 = time.time()

    # Step 1: Detect ball
    det = [Detection(class_id=32, class_name="sports ball", confidence=0.9, bbox=[100.0, 100.0, 110.0, 110.0])]
    state1 = tracker.update(dummy_frame, det, t0)
    assert state1 is not None
    assert tracker.is_tracking is True

    # Step 2: Next frame, ball missing (occlusion) -> Kalman predicts
    state2 = tracker.update(dummy_frame, [], t0 + 0.033)
    assert state2 is not None
    assert tracker.is_tracking is True
    assert tracker.missed_frames == 1

    # Step 3: Ball returns -> tracker associates and resets missed_frames
    det3 = [Detection(class_id=32, class_name="sports ball", confidence=0.88, bbox=[105.0, 100.0, 115.0, 110.0])]
    state3 = tracker.update(dummy_frame, det3, t0 + 0.066)
    assert state3 is not None
    assert tracker.missed_frames == 0

def test_player_tracker_association():
    pt = PlayerTracker(max_distance=50.0)
    t0 = time.time()

    dets1 = [Detection(class_id=0, class_name="person", confidence=0.9, bbox=[200.0, 100.0, 250.0, 200.0])]
    res1 = pt.update(dets1, t0)
    assert len(res1) == 1
    track_id = res1[0].track_id
    assert track_id is not None

    # Slightly moved person in next frame should retain same track_id
    dets2 = [Detection(class_id=0, class_name="person", confidence=0.9, bbox=[203.0, 102.0, 253.0, 202.0])]
    res2 = pt.update(dets2, t0 + 0.033)
    assert len(res2) == 1
    assert res2[0].track_id == track_id

@pytest.mark.asyncio
async def test_pose_engine_inference():
    pe = PoseEngine()
    dummy = np.zeros((360, 640, 3), dtype=np.uint8)
    # Draw simple human silhouette
    cv2.circle(dummy, (320, 120), 20, (255, 255, 255), -1)  # head
    cv2.line(dummy, (320, 140), (320, 240), (255, 255, 255), 10)  # body

    poses, latency_ms = await pe.estimate(dummy)
    assert isinstance(poses, list)
    assert latency_ms > 0
    annotated = pe.draw_skeletons(dummy, poses)
    assert annotated.shape == dummy.shape
