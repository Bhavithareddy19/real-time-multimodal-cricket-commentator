import numpy as np
import cv2
from backend.app.vision.jersey import JerseyColorClassifier

def test_jersey_color_australia_gold():
    # Synthetic frame with an Australian gold jersey player
    h, w = 300, 300
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    # BGR for Gold/Yellow: [0, 215, 255]
    bbox = [100.0, 50.0, 160.0, 220.0]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    frame[y1:y2, x1:x2] = (0, 215, 255)
    
    team, hex_col, conf = JerseyColorClassifier.extract_jersey_color(frame, bbox)
    assert team == "Australia"
    assert conf >= 0.70
    assert hex_col.startswith("#")

def test_jersey_color_india_blue():
    # Synthetic frame with an Indian blue jersey player
    h, w = 300, 300
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    # BGR for Dodger Blue: [255, 144, 30]
    bbox = [80.0, 60.0, 150.0, 240.0]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    frame[y1:y2, x1:x2] = (255, 144, 30)
    
    team, hex_col, conf = JerseyColorClassifier.extract_jersey_color(frame, bbox)
    assert team == "India"
    assert conf >= 0.70

def test_jersey_color_test_white():
    # Synthetic frame with a Test match white jersey player
    h, w = 300, 300
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    # BGR for White: [245, 245, 245]
    bbox = [80.0, 60.0, 150.0, 240.0]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    frame[y1:y2, x1:x2] = (245, 245, 245)
    
    team, hex_col, conf = JerseyColorClassifier.extract_jersey_color(frame, bbox)
    assert "White" in team
    assert conf >= 0.70
