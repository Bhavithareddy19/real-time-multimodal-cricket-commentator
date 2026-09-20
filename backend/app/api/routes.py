import os
import cv2
import json
import time
import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, UploadFile, File

from backend.app.config import get_settings
from backend.app.models.schemas import (
    SessionStartRequest,
    SessionStatusResponse,
    MatchScorecard,
    PerformanceMetrics,
    SourceValidationRequest,
    SourceValidationResponse,
    SourceStatus
)
from backend.app.streaming.frame_pipeline import FramePipeline
from backend.app.streaming.video_source import VideoSourceManager

logger = logging.getLogger("api.routes")
api_router = APIRouter()

# Global pipeline reference set in main.py
pipeline_instance: FramePipeline = None

def set_pipeline(pipeline: FramePipeline):
    global pipeline_instance
    pipeline_instance = pipeline

@api_router.get("/health")
async def health_check():
    settings = get_settings()
    return {
        "status": "healthy",
        "service": "Real-Time Multimodal Cricket Commentator",
        "device": settings.get_resolved_device(),
        "version": "1.0.0"
    }

@api_router.get("/api/config")
async def get_configuration():
    settings = get_settings()
    return {
        "app_env": settings.APP_ENV,
        "device": settings.get_resolved_device(),
        "yolo_model": settings.YOLO_MODEL,
        "yolo_pose_model": settings.YOLO_POSE_MODEL,
        "target_fps": settings.TARGET_FPS,
        "llm_provider": settings.LLM_PROVIDER,
        "tts_provider": settings.TTS_PROVIDER,
        "commentary_style": settings.COMMENTARY_STYLE
    }

@api_router.get("/api/sources")
async def list_available_sources():
    """Enumerates available webcam devices, sample video files, and uploaded videos."""
    settings = get_settings()
    sources = []

    # Check local webcam devices (up to 2)
    for idx in range(2):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            sources.append({
                "type": "webcam",
                "id": str(idx),
                "name": f"Camera Device {idx}" + (" (Default)" if idx == settings.WEBCAM_INDEX else ""),
                "available": True
            })
            cap.release()
        else:
            # Try default backend
            cap2 = cv2.VideoCapture(idx)
            if cap2.isOpened():
                sources.append({
                    "type": "webcam",
                    "id": str(idx),
                    "name": f"Camera Device {idx}",
                    "available": True
                })
                cap2.release()

    # Scan sample_videos directory
    video_dir = os.path.join(os.getcwd(), "sample_videos")
    if os.path.exists(video_dir):
        for f in sorted(os.listdir(video_dir)):
            if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
                sources.append({
                    "type": "uploaded",
                    "id": os.path.join(video_dir, f),
                    "name": f"Sample: {f}",
                    "available": True
                })

    # Scan uploads subdirectory
    upload_dir = os.path.join(video_dir, "uploads")
    if os.path.exists(upload_dir):
        for f in sorted(os.listdir(upload_dir)):
            if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
                sources.append({
                    "type": "uploaded",
                    "id": os.path.join(upload_dir, f),
                    "name": f"Uploaded: {f}",
                    "available": True
                })

    # RTSP option
    sources.append({
        "type": "livestream",
        "id": "rtsp://",
        "name": "RTSP Live Stream (Custom URL)",
        "available": True
    })

    return {"sources": sources}

@api_router.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    """
    Uploads a cricket match video (MP4, MOV, MKV, AVI) for AI commentary.
    Saves to sample_videos/uploads/ and validates stream accessibility.
    """
    allowed_exts = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
    _, ext = os.path.splitext(file.filename)
    if ext.lower() not in allowed_exts:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{ext}'. Supported formats: {', '.join(allowed_exts)}"
        )

    upload_dir = os.path.join(os.getcwd(), "sample_videos", "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    clean_filename = f"{int(time.time())}_{os.path.basename(file.filename)}"
    dest_path = os.path.join(upload_dir, clean_filename)

    try:
        with open(dest_path, "wb") as f:
            while chunk := await file.read(1024 * 1024):
                f.write(chunk)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {e}")

    valid, message, metadata = VideoSourceManager.validate_source(dest_path, "uploaded")
    if not valid:
        if os.path.exists(dest_path):
            os.remove(dest_path)
        raise HTTPException(status_code=400, detail=message)

    return {
        "status": "uploaded",
        "file_path": dest_path,
        "filename": file.filename,
        "metadata": metadata
    }

@api_router.post("/api/sources/validate", response_model=SourceValidationResponse)
async def validate_source(req: SourceValidationRequest):
    """
    Validates accessibility of an online video or stream URL.
    Refuses inaccessible/DRM streams honestly.
    """
    valid, message, metadata = VideoSourceManager.validate_source(req.url, req.source_type)
    return SourceValidationResponse(
        valid=valid,
        source_type=metadata.get("source_type", req.source_type or "unknown"),
        message=message,
        duration_sec=metadata.get("duration_sec"),
        fps=metadata.get("fps"),
        width=metadata.get("width"),
        height=metadata.get("height")
    )

@api_router.post("/api/session/start")
async def start_session(req: SessionStartRequest):
    if pipeline_instance.is_running:
        await pipeline_instance.stop()
    
    try:
        await pipeline_instance.start(
            source_type=req.source_type,
            source_target=req.source_target,
            playback_speed=req.playback_speed
        )
        return {
            "status": "started",
            "source_type": req.source_type,
            "source_target": req.source_target,
            "playback_speed": req.playback_speed
        }
    except Exception as e:
        logger.error(f"Failed to start session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/api/session/stop")
async def stop_session():
    await pipeline_instance.stop()
    return {"status": "stopped"}

@api_router.post("/api/session/pause")
async def pause_session():
    pipeline_instance.pause()
    return {"status": "paused"}

@api_router.post("/api/session/resume")
async def resume_session():
    pipeline_instance.resume()
    return {"status": "resumed"}

@api_router.post("/api/session/speed")
async def set_playback_speed(data: Dict[str, float]):
    speed = float(data.get("speed", 1.0))
    pipeline_instance.set_speed(speed)
    return {"status": "updated", "playback_speed": speed}

@api_router.get("/api/session/status")
async def get_session_status():
    status_info = pipeline_instance.get_source_status()
    source_status_obj = SourceStatus(
        source_type=status_info.get("source_type", "uploaded"),
        status=status_info.get("status", "idle"),
        target=status_info.get("target", ""),
        duration_sec=status_info.get("duration_sec"),
        video_time_sec=status_info.get("video_time_sec", 0.0),
        playback_speed=status_info.get("playback_speed", 1.0)
    )
    return SessionStatusResponse(
        is_running=pipeline_instance.is_running,
        source_type=pipeline_instance.source_type,
        frames_processed=pipeline_instance.processed_counter,
        scorecard=MatchScorecard(),
        metrics=pipeline_instance.get_metrics(),
        source_status=source_status_obj
    )

@api_router.get("/api/metrics")
async def get_metrics():
    return pipeline_instance.get_metrics()

@api_router.get("/api/calibration")
async def get_calibration():
    calib_path = os.path.join("config", "calibration.json")
    if os.path.exists(calib_path):
        with open(calib_path, "r") as f:
            return json.load(f)
    return {}

@api_router.post("/api/calibration")
async def save_calibration(data: Dict[str, Any]):
    calib_path = os.path.join("config", "calibration.json")
    os.makedirs(os.path.dirname(calib_path), exist_ok=True)
    with open(calib_path, "w") as f:
        json.dump(data, f, indent=2)
    return {"status": "saved", "config": data}

@api_router.post("/api/settings/llm")
async def update_llm_settings(payload: Dict[str, Any]):
    """Configures LLM provider (Gemini / OpenAI) dynamically."""
    from backend.app.ai.llm import create_llm_provider
    prov_name = payload.get("provider", "gemini")
    api_key = payload.get("api_key", "")
    model = payload.get("model", "")

    new_prov = create_llm_provider(prov_name, api_key=api_key, model=model)
    if pipeline_instance:
        pipeline_instance.commentary_engine.set_provider(new_prov)

    return {
        "status": "success",
        "provider": new_prov.__class__.__name__,
        "is_configured": new_prov.is_configured(),
        "model": getattr(new_prov, "model", "")
    }

