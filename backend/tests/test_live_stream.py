import asyncio
import json
import pytest
from backend.app.main import app, frame_pipeline
from backend.app.streaming.frame_pipeline import FramePipeline

@pytest.mark.asyncio
async def test_frame_pipeline_live_processing():
    """Verifies that FramePipeline can ingest a real video, run YOLO, and deliver frames with metrics."""
    test_pipeline = FramePipeline()
    sub_q = test_pipeline.add_subscriber()
    
    # Start on sample video
    await test_pipeline.start(source_type="video", source_target="sample_videos/cover_drive_four.mp4")
    assert test_pipeline.is_running is True
    
    # Wait for at least 3 processed frames
    frames_received = []
    for _ in range(3):
        item = await asyncio.wait_for(sub_q.get(), timeout=5.0)
        frames_received.append(item)
    
    assert len(frames_received) == 3
    first_frame = frames_received[0]
    assert first_frame["frame_id"] >= 1
    assert "metrics" in first_frame
    assert "detections" in first_frame
    assert "jpeg" in first_frame
    assert len(first_frame["jpeg"]) > 1000  # Valid JPEG buffer
    
    metrics = test_pipeline.get_metrics()
    assert metrics.vision_latency_ms > 0
    assert metrics.queue_size >= 0
    
    # Clean shutdown
    await test_pipeline.stop()
    assert test_pipeline.is_running is False
