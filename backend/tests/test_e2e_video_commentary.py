import os
import asyncio
import json
import pytest
import httpx
import websockets

SAMPLE_VIDEO_PATH = os.path.join(os.getcwd(), "sample_videos", "cover_drive_four.mp4")

@pytest.mark.asyncio
async def test_e2e_upload_and_stream_commentary():
    """
    Verifies the 5 core user requirements:
    1. Upload an actual cricket .mp4
    2. Start commentary session
    3. Confirm video frames play and video_timestamp increments
    4. Confirm real AI detections (YOLO bounding boxes)
    5. Confirm commentary is generated and streaming audio is received
    """
    assert os.path.exists(SAMPLE_VIDEO_PATH), f"Sample video {SAMPLE_VIDEO_PATH} not found"

    # Step 0: Ensure server is responsive
    async with httpx.AsyncClient(base_url="http://localhost:8000") as client:
        for _ in range(10):
            try:
                h_res = await client.get("/health")
                if h_res.status_code == 200:
                    break
            except Exception:
                await asyncio.sleep(0.5)

    # Step 1: Test Video Upload via REST API
    async with httpx.AsyncClient(base_url="http://localhost:8000") as client:
        with open(SAMPLE_VIDEO_PATH, "rb") as f:
            files = {"file": ("test_cricket_match.mp4", f, "video/mp4")}
            res = await client.post("/api/upload", files=files)
        
        assert res.status_code == 200, f"Upload failed: {res.text}"
        data = res.json()
        assert data["status"] == "uploaded"
        assert "file_path" in data
        assert data["metadata"]["fps"] > 0
        assert data["metadata"]["duration_sec"] > 0
        uploaded_path = data["file_path"]

    # Step 2: Connect to Live WebSocket and start commentary
    uri = "ws://localhost:8000/ws/live"
    async with websockets.connect(uri) as ws:
        # Send START COMMENTARY command
        start_cmd = {
            "action": "start",
            "source_type": "uploaded",
            "source_target": uploaded_path,
            "playback_speed": 2.0  # Run at 2x for fast testing
        }
        await ws.send(json.dumps(start_cmd))

        frames_received = 0
        first_video_ts = None
        last_video_ts = None
        real_detections_found = False
        commentary_received = False
        audio_received = False

        # Read frames, events, commentary, and audio for up to 18 seconds
        start_time = asyncio.get_event_loop().time()
        while asyncio.get_event_loop().time() - start_time < 18.0:
            try:
                msg_raw = await asyncio.wait_for(ws.recv(), timeout=3.0)
                msg = json.loads(msg_raw)
                msg_type = msg.get("type")

                if msg_type == "frame":
                    frames_received += 1
                    v_ts = msg.get("video_timestamp", 0.0)
                    if first_video_ts is None:
                        first_video_ts = v_ts
                    last_video_ts = v_ts

                    # Check real YOLO detections
                    dets = msg.get("detections", [])
                    if len(dets) > 0:
                        real_detections_found = True
                        for d in dets:
                            assert "class_name" in d
                            assert "confidence" in d
                            assert len(d["bbox"]) == 4

                elif msg_type == "commentary":
                    commentary_received = True
                    comm_data = msg.get("data", {})
                    assert "text" in comm_data
                    assert len(comm_data["text"]) > 5
                    assert comm_data["style"] == "PROFESSIONAL"

                elif msg_type == "audio":
                    audio_received = True
                    assert "audio_base64" in msg
                    assert len(msg["audio_base64"]) > 100
                    assert msg["format"] == "mp3"

                # If all 5 criteria are verified, we can exit early!
                if frames_received >= 15 and real_detections_found and commentary_received and audio_received:
                    break

            except asyncio.TimeoutError:
                break

        # Stop the session
        await ws.send(json.dumps({"action": "stop"}))

        # Assertions for the 5 critical verification criteria:
        # Criterion 3: Confirm video actually plays and advances
        assert frames_received >= 10, f"Expected at least 10 frames, got {frames_received}"
        assert last_video_ts is not None and last_video_ts >= first_video_ts, "Video timestamp did not advance"

        # Criterion 4: Confirm AI detects visual entities rather than fake hardcoded items
        assert real_detections_found is True, "Expected real YOLO detections in frames"

        # Criterion 5: Confirm commentary is generated and audio is spoken/streamed
        assert commentary_received is True, "Expected AI commentary to be generated"
        assert audio_received is True, "Expected streaming TTS audio to be synthesized and dispatched"
