import time
import pytest
from backend.app.cricket.state_machine import CricketEventStateMachine
from backend.app.cricket.shot_classifier import ShotClassifier
from backend.app.cricket.rules import is_point_inside_polygon, has_crossed_boundary, is_point_in_roi
from backend.app.models.schemas import BallState

def test_geometry_boundary_rules():
    boundary = [[100, 100], [500, 100], [500, 500], [100, 500]]
    
    # Point clearly inside
    assert is_point_inside_polygon((300, 300), boundary) is True
    
    # Point clearly outside
    assert is_point_inside_polygon((50, 50), boundary) is False
    assert is_point_inside_polygon((600, 300), boundary) is False

    # Crossing boundary
    assert has_crossed_boundary(prev_point=(490, 300), curr_point=(510, 300), boundary_polygon=boundary) is True
    # Still inside
    assert has_crossed_boundary(prev_point=(200, 200), curr_point=(250, 250), boundary_polygon=boundary) is False

def test_stump_roi_rules():
    stumps = [300, 280, 340, 320]  # [x1, y1, x2, y2]
    assert is_point_in_roi((320, 300), stumps) is True
    assert is_point_in_roi((200, 300), stumps) is False

def test_shot_classifier_cover_drive_and_defensive():
    sc = ShotClassifier()

    # Case 1: Cover drive (incoming from bowler downwards, outgoing through covers towards left/deep)
    # Incoming: (320, 150) -> (320, 270)
    # Contact at (320, 275)
    # Outgoing: (300, 285) -> (270, 300) -> (240, 315) -> (200, 335) (exit angle ~150-160 deg)
    trajectory = [
        (320.0, 150.0), (320.0, 180.0), (320.0, 210.0), (320.0, 240.0), (320.0, 270.0),
        (315.0, 275.0), (285.0, 290.0), (255.0, 305.0), (220.0, 320.0), (180.0, 340.0)
    ]
    res = sc.classify(ball_trajectory=trajectory)
    assert res.shot_type == "COVER_DRIVE"
    assert res.confidence >= 0.80

    # Case 2: Defensive block (almost zero exit velocity)
    defensive_traj = [
        (320.0, 150.0), (320.0, 190.0), (320.0, 230.0), (320.0, 270.0), (320.0, 290.0),
        (320.0, 292.0), (320.5, 292.5), (321.0, 293.0)
    ]
    res_def = sc.classify(ball_trajectory=defensive_traj)
    assert res_def.shot_type == "DEFENSIVE"
    assert res_def.confidence >= 0.80

    # Case 3: Insufficient trajectory points returns UNKNOWN with low confidence
    res_short = sc.classify(ball_trajectory=[(320.0, 150.0), (320.0, 180.0)])
    assert res_short.shot_type == "UNKNOWN"
    assert res_short.confidence < 0.50

def test_state_machine_delivery_to_four():
    sm = CricketEventStateMachine()
    sm.boundary_polygon = [[50, 50], [550, 50], [550, 450], [50, 450]]
    t = 1000.0

    # Frame 1: IDLE -> Bowler approaches, ball moving down
    b1 = BallState(x=320.0, y=100.0, vx=0.0, vy=5.0, confidence=0.9, timestamp=t)
    ev1 = sm.process_frame(frame_id=1, ball_state=b1, trajectory=[(320.0, 100.0)], poses=[], detections=[], timestamp=t)
    assert sm.current_state in ["DELIVERY", "BALL_IN_PLAY"]
    assert ev1 is not None
    assert ev1.event_type == "DELIVERY"

    # Frame 2: Ball in flight
    b2 = BallState(x=320.0, y=200.0, vx=0.0, vy=5.0, confidence=0.9, timestamp=t + 0.033)
    sm.process_frame(frame_id=2, ball_state=b2, trajectory=[(320.0, 100.0), (320.0, 200.0)], poses=[], detections=[], timestamp=t + 0.033)
    assert sm.current_state == "BALL_IN_PLAY"

    # Frame 3: Bat contact & Shot played
    traj = [
        (320.0, 100.0), (320.0, 150.0), (320.0, 200.0), (320.0, 250.0), (320.0, 280.0),
        (300.0, 295.0), (260.0, 315.0), (220.0, 335.0)
    ]
    b3 = BallState(x=220.0, y=335.0, vx=-12.0, vy=6.0, confidence=0.9, timestamp=t + 0.066)
    ev_shot = sm.process_frame(frame_id=3, ball_state=b3, trajectory=traj, poses=[], detections=[], timestamp=t + 0.066)
    assert sm.current_state in ["BAT_CONTACT", "BALL_TRAVEL"]
    assert ev_shot is not None
    assert ev_shot.event_type == "SHOT_PLAYED"

    # Frame 4: Ball travels towards boundary inside field
    b4 = BallState(x=70.0, y=410.0, vx=-15.0, vy=8.0, confidence=0.9, timestamp=t + 0.100)
    traj.append((70.0, 410.0))
    sm.process_frame(frame_id=4, ball_state=b4, trajectory=traj, poses=[], detections=[], timestamp=t + 0.100)
    assert sm.current_state == "BALL_TRAVEL"

    # Frame 5: Ball crosses boundary (from x=70 inside to x=40 outside)
    b5 = BallState(x=40.0, y=425.0, vx=-15.0, vy=8.0, confidence=0.9, timestamp=t + 0.133)
    traj.append((40.0, 425.0))
    ev_four = sm.process_frame(frame_id=5, ball_state=b5, trajectory=traj, poses=[], detections=[], timestamp=t + 0.133)
    assert sm.current_state == "BOUNDARY_DETECTED"
    assert ev_four is not None
    assert ev_four.event_type == "FOUR"
    assert ev_four.runs == 4

    # Frame 6 (CRITICAL DEBOUNCE TEST): Next frame ball is still outside.
    # Must NOT emit duplicate FOUR!
    b6 = BallState(x=30.0, y=430.0, vx=-10.0, vy=5.0, confidence=0.9, timestamp=t + 0.166)
    traj.append((30.0, 430.0))
    ev_dup = sm.process_frame(frame_id=6, ball_state=b6, trajectory=traj, poses=[], detections=[], timestamp=t + 0.166)
    assert ev_dup is None  # Debounced!

def test_state_machine_wicket_detection():
    sm = CricketEventStateMachine()
    sm.stump_roi = [300, 280, 340, 320]
    t = 1000.0

    # Start delivery
    b1 = BallState(x=320.0, y=100.0, vx=0.0, vy=5.0, confidence=0.9, timestamp=t)
    sm.process_frame(frame_id=1, ball_state=b1, trajectory=[(320.0, 100.0)], poses=[], detections=[], timestamp=t)

    # Ball enters stump ROI
    b_wicket = BallState(x=320.0, y=295.0, vx=0.0, vy=5.0, confidence=0.92, timestamp=t + 0.05)
    ev_wkt = sm.process_frame(
        frame_id=2,
        ball_state=b_wicket,
        trajectory=[(320.0, 100.0), (320.0, 295.0)],
        poses=[],
        detections=[],
        timestamp=t + 0.05
    )
    assert sm.current_state == "WICKET_DETECTED"
    assert ev_wkt is not None
    assert ev_wkt.event_type == "WICKET"
