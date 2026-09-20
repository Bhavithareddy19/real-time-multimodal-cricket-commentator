import cv2
import numpy as np
import os
import math

def create_synthetic_cricket_clip(output_path: str = "sample_videos/cover_drive_four.mp4", duration_sec: int = 10, fps: int = 30):
    """
    Synthesizes a realistic 30 FPS cricket video clip for testing and demonstration:
    - Field with boundary rope
    - Central turf pitch and crease markings
    - Three-stump wicket set
    - Batsman figure in stance
    - Bowler delivery runup and release
    - Ball trajectory moving toward batsman, bat contact, and acceleration toward boundary.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    width, height = 640, 360
    total_frames = duration_sec * fps

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    pitch_pts = np.array([[260, 200], [380, 200], [420, 340], [220, 340]], np.int32)
    boundary_pts = np.array([[40, 330], [180, 190], [460, 190], [600, 330], [480, 355], [160, 355]], np.int32)

    # Key timeline in frames:
    # 0 - 60: Bowler runup
    # 60 - 110: Ball in flight to batsman
    # 110: Bat contact (Shot played: COVER DRIVE)
    # 111 - 200: Ball races towards boundary
    # 200+: Boundary crossed, ball complete

    for f in range(total_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # 1. Outfield (lush green)
        frame[:] = (35, 120, 45)

        # 2. Boundary rope
        cv2.polylines(frame, [boundary_pts], isClosed=True, color=(240, 240, 240), thickness=3)

        # 3. Pitch turf (light earthy brown/tan)
        cv2.fillPoly(frame, [pitch_pts], (140, 185, 205))
        cv2.polylines(frame, [pitch_pts], isClosed=True, color=(110, 150, 170), thickness=2)

        # 4. Crease markings (white lines)
        cv2.line(frame, (235, 310), (405, 310), (255, 255, 255), 2)  # Popping crease
        cv2.line(frame, (270, 220), (370, 220), (255, 255, 255), 2)  # Bowling crease

        # 5. Stumps at batting end (3 wooden vertical lines)
        stump_color = (40, 130, 210)
        cv2.line(frame, (315, 300), (315, 325), stump_color, 3)
        cv2.line(frame, (320, 300), (320, 325), stump_color, 3)
        cv2.line(frame, (325, 300), (325, 325), stump_color, 3)
        cv2.line(frame, (313, 300), (327, 300), stump_color, 2)  # bails

        # 6. Batsman in stance (right-handed, waiting at x=340, y=280..320)
        bat_x, bat_y = 345, 290
        # Head
        cv2.circle(frame, (bat_x, bat_y - 30), 8, (230, 230, 250), -1)
        # Torso
        cv2.line(frame, (bat_x, bat_y - 22), (bat_x - 5, bat_y + 10), (245, 245, 245), 6)
        # Legs / Pads
        cv2.line(frame, (bat_x - 5, bat_y + 10), (bat_x - 12, bat_y + 35), (250, 250, 250), 5)
        cv2.line(frame, (bat_x - 5, bat_y + 10), (bat_x + 5, bat_y + 35), (250, 250, 250), 5)

        # Bat position
        if f < 105:
            # Stance / backlift
            cv2.line(frame, (bat_x - 15, bat_y - 10), (bat_x - 28, bat_y - 25), (30, 75, 140), 4)
        elif 105 <= f <= 125:
            # Driving swing through covers
            progress = (f - 105) / 20.0
            end_x = int(bat_x - 10 - progress * 20)
            end_y = int(bat_y + 15 - progress * 20)
            cv2.line(frame, (bat_x - 10, bat_y - 5), (end_x, end_y), (30, 75, 140), 4)
        else:
            # Follow through
            cv2.line(frame, (bat_x - 10, bat_y - 5), (bat_x - 30, bat_y - 5), (30, 75, 140), 4)

        # 7. Bowler action and Ball trajectory
        ball_color = (0, 0, 255)  # Red cricket ball
        ball_pos = None

        if f < 60:
            # Bowler running up from top
            b_x = int(320 + math.sin(f * 0.3) * 5)
            b_y = int(140 + (f / 60.0) * 60)
            cv2.circle(frame, (b_x, b_y - 20), 7, (230, 230, 250), -1)
            cv2.line(frame, (b_x, b_y - 13), (b_x, b_y + 10), (220, 220, 220), 5)
            cv2.line(frame, (b_x, b_y + 10), (b_x - 6, b_y + 25), (220, 220, 220), 4)
            cv2.line(frame, (b_x, b_y + 10), (b_x + 6, b_y + 25), (220, 220, 220), 4)
        elif 60 <= f < 110:
            # Ball released and bowling towards batsman
            t = (f - 60) / 50.0
            bx = int(320 + t * 15)
            # Parabolic trajectory bouncing on pitch
            pitch_bounce = math.sin(t * math.pi) * 35
            by = int(200 + t * 110 - pitch_bounce)
            ball_pos = (bx, by)
        elif 110 <= f <= 210:
            # Shot played: Ball races through extra cover towards boundary!
            t = (f - 110) / 90.0
            bx = int(335 - t * 240)
            by = int(305 + t * 45)
            ball_pos = (bx, by)
        else:
            # Ball settled past boundary
            ball_pos = (80, 355)

        if ball_pos:
            cv2.circle(frame, ball_pos, 5, ball_color, -1)
            # Add subtle motion trail
            cv2.circle(frame, ball_pos, 7, (100, 100, 255), 1)

        out.write(frame)

    out.release()
    print(f"Sample cricket video generated successfully at '{output_path}' ({duration_sec}s @ {fps}fps).")

if __name__ == "__main__":
    create_synthetic_cricket_clip()
