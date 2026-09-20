import os
import asyncio
import pytest
import numpy as np

from backend.app.streaming.video_source import (
    VideoSourceManager,
    UploadedVideoSource,
    DirectURLVideoSource,
    LiveStreamSource,
    WebcamSource
)
from backend.app.streaming.frame_pipeline import FramePipeline

SAMPLE_VIDEO_PATH = os.path.join(os.getcwd(), "sample_videos", "cover_drive_four.mp4")

def test_source_type_detection():
    assert VideoSourceManager.detect_source_type("0") == "webcam"
    assert VideoSourceManager.detect_source_type("1") == "webcam"
    assert VideoSourceManager.detect_source_type("rtsp://192.168.1.10:554/stream") == "livestream"
    assert VideoSourceManager.detect_source_type("rtmp://live.example.com/cricket") == "livestream"
    assert VideoSourceManager.detect_source_type("https://example.com/live/feed.m3u8") == "livestream"
    assert VideoSourceManager.detect_source_type("https://example.com/match.mp4") == "direct_url"
    if os.path.exists(SAMPLE_VIDEO_PATH):
        assert VideoSourceManager.detect_source_type(SAMPLE_VIDEO_PATH) == "uploaded"

def test_source_validation_valid_file():
    if os.path.exists(SAMPLE_VIDEO_PATH):
        valid, message, metadata = VideoSourceManager.validate_source(SAMPLE_VIDEO_PATH, "uploaded")
        assert valid is True
        assert metadata["fps"] > 0
        assert metadata["duration_sec"] > 0
        assert metadata["width"] > 0
        assert metadata["height"] > 0

def test_source_validation_invalid_url():
    valid, message, metadata = VideoSourceManager.validate_source("https://invalid-stream-domain.xyz/cricket.m3u8")
    assert valid is False
    assert "cannot be accessed" in message

@pytest.mark.asyncio
async def test_uploaded_video_source_frames():
    if not os.path.exists(SAMPLE_VIDEO_PATH):
        pytest.skip("Sample video asset not found")

    source = UploadedVideoSource(file_path=SAMPLE_VIDEO_PATH, playback_speed=2.0)
    await source.start()
    assert source.is_running is True
    assert source.status == "connected"
    assert source.fps > 0

    count = 0
    timestamps = []
    async for frame_id, video_ts, frame, meta in source.frames():
        count += 1
        timestamps.append(video_ts)
        assert frame is not None
        assert isinstance(frame, np.ndarray)
        assert meta["source_type"] == "uploaded"
        if count >= 10:
            break

    assert count == 10
    # Video timestamps should be non-decreasing
    assert timestamps[-1] >= timestamps[0]

    # Test speed update
    source.set_speed(1.5)
    assert source.playback_speed == 1.5

    # Test pause and resume
    source.pause()
    assert source.is_paused is True
    assert source.status == "paused"
    source.resume()
    assert source.is_paused is False
    assert source.status == "connected"

    await source.stop()
    assert source.is_running is False
    assert source.status == "idle"

@pytest.mark.asyncio
async def test_frame_pipeline_with_uploaded_source():
    if not os.path.exists(SAMPLE_VIDEO_PATH):
        pytest.skip("Sample video asset not found")

    pipeline = FramePipeline()
    sub_q = pipeline.add_subscriber()

    await pipeline.start(
        source_type="uploaded",
        source_target=SAMPLE_VIDEO_PATH,
        playback_speed=2.0
    )
    assert pipeline.is_running is True
    assert pipeline.video_source is not None

    received_frames = 0
    while received_frames < 5:
        try:
            msg = await asyncio.wait_for(sub_q.get(), timeout=2.0)
            if msg.get("type") == "frame":
                received_frames += 1
                assert "video_timestamp" in msg
                assert "source_status" in msg
                assert msg["source_status"]["source_type"] == "uploaded"
        except asyncio.TimeoutError:
            break

    assert received_frames >= 3

    # Test speed change on pipeline
    pipeline.set_speed(1.5)
    assert pipeline.playback_speed == 1.5

    # Test pause on pipeline
    pipeline.pause()
    assert pipeline.is_paused is True
    pipeline.resume()
    assert pipeline.is_paused is False

    await pipeline.stop()
    assert pipeline.is_running is False
    pipeline.remove_subscriber(sub_q)
