import os
import sys
import logging
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("cricket.main")

from backend.app.config import get_settings
from backend.app.vision.detector import VisionEngine
from backend.app.streaming.frame_pipeline import FramePipeline
from backend.app.api.routes import api_router, set_pipeline as set_routes_pipeline
from backend.app.api.websocket import ws_router, set_pipeline as set_ws_pipeline

# Global pipeline instance
vision_engine = VisionEngine()
frame_pipeline = FramePipeline(vision_engine=vision_engine)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Real-Time Multimodal Cricket Commentator Backend...")
    settings = get_settings()
    logger.info(f"Target device resolved: {settings.get_resolved_device()}")
    
    # Warmup vision engine in background thread so server starts instantly
    asyncio.create_task(asyncio.to_thread(vision_engine.load_model))
    
    # Link pipeline to routes and websockets
    set_routes_pipeline(frame_pipeline)
    set_ws_pipeline(frame_pipeline)
    
    yield
    
    logger.info("Shutting down backend services...")
    await frame_pipeline.stop()
    logger.info("All pipeline tasks and resources cleanly terminated.")

app = FastAPI(
    title="Real-Time Multimodal Cricket Commentator API",
    description="Multimodal CV, State Engine, LLM Commentary & Streaming TTS",
    version="1.0.0",
    lifespan=lifespan
)

# CORS setup for Vite frontend (http://localhost:5173, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(api_router)
app.include_router(ws_router)

# Mount compiled frontend if available for unified single-port deployment
from fastapi.staticfiles import StaticFiles
dist_dir = os.path.join(os.getcwd(), "frontend", "dist")
if os.path.exists(dist_dir):
    logger.info(f"Mounting production frontend from {dist_dir}")
    app.mount("/", StaticFiles(directory=dist_dir, html=True), name="static")

def run():
    settings = get_settings()
    uvicorn.run(
        "backend.app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=False,
        log_level="info"
    )

if __name__ == "__main__":
    run()
