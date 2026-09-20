import asyncio
import pytest
import cv2
import numpy as np
from httpx import AsyncClient, ASGITransport

from backend.app.main import app
from backend.app.vision.detector import VisionEngine

@pytest.mark.asyncio
async def test_health():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "device" in data

@pytest.mark.asyncio
async def test_sources():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/api/sources")
        assert response.status_code == 200
        data = response.json()
        assert "sources" in data
        assert len(data["sources"]) > 0

@pytest.mark.asyncio
async def test_vision_engine_inference():
    ve = VisionEngine()
    dummy = np.zeros((360, 640, 3), dtype=np.uint8)
    # Draw a simulated person rectangle
    cv2.rectangle(dummy, (200, 100), (300, 300), (255, 255, 255), -1)
    
    detections, latency_ms = await ve.detect(dummy)
    assert isinstance(detections, list)
    assert latency_ms > 0
    
    # Test overlay drawing
    annotated = ve.draw_overlays(dummy, detections, fps=30.0, latency_ms=latency_ms)
    assert annotated.shape == dummy.shape
