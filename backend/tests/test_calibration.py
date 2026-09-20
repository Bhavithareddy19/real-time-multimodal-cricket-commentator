import numpy as np
import cv2
from backend.app.vision.calibration import AutoPitchCalibrator

def test_auto_pitch_calibrator_fallback():
    calib = AutoPitchCalibrator("config/calibration.json")
    assert len(calib.boundary_polygon) >= 3
    assert len(calib.stump_roi) == 4

def test_auto_pitch_calibrator_with_frame():
    calib = AutoPitchCalibrator("config/calibration.json")
    
    # Create a synthetic frame with green grass and a central tan pitch
    h, w = 360, 640
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    # Green outfield (H: ~50, S: ~150, V: ~150 -> BGR: [35, 120, 45])
    frame[:] = (35, 120, 45)
    
    # Central tan pitch (H: ~20, S: ~70, V: ~180 -> BGR: [140, 185, 205])
    pitch_pts = np.array([[260, 180], [380, 180], [420, 330], [220, 330]], np.int32)
    cv2.fillPoly(frame, [pitch_pts], (140, 185, 205))
    
    # Crease line (white)
    cv2.line(frame, (230, 300), (410, 300), (255, 255, 255), 2)
    
    is_cal, conf, details = calib.calibrate_from_frame(frame)
    assert conf > 0.4
    assert "boundary_polygon" in details
    assert "stump_roi" in details
    assert len(details["stump_roi"]) == 4
