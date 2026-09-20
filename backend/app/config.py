from functools import lru_cache
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict
import torch
import os

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # App
    APP_ENV: str = "development"
    PORT: int = 8000
    HOST: str = "0.0.0.0"

    # LLM
    LLM_PROVIDER: str = "gemini"
    LLM_MODEL: str = "gemini-2.0-flash"
    GEMINI_API_KEY: str = ""
    OPENAI_API_KEY: str = ""
    LLM_API_BASE: str = "https://api.openai.com/v1"

    # TTS
    TTS_PROVIDER: str = "edge-tts"
    TTS_VOICE: str = "en-GB-RyanNeural"
    TTS_API_KEY: str = ""

    # Computer Vision & Hardware
    DEVICE: str = "auto"
    USE_FP16: bool = False
    USE_TENSORRT: bool = False
    YOLO_MODEL: str = "yolov8n.pt"
    YOLO_POSE_MODEL: str = "yolov8n-pose.pt"
    TARGET_FPS: int = 30
    FRAME_WIDTH: int = 640
    FRAME_HEIGHT: int = 360

    # Video Source
    DEFAULT_VIDEO_SOURCE: Literal["webcam", "video", "rtsp"] = "webcam"
    WEBCAM_INDEX: int = 0
    VIDEO_FILE_PATH: str = ""
    RTSP_URL: str = ""

    # Redis & DB (Optional)
    REDIS_URL: str = ""
    DATABASE_URL: str = ""

    # Feature Flags
    ENABLE_AUDIO: bool = True
    ENABLE_LLM: bool = True
    COMMENTARY_STYLE: str = "PROFESSIONAL"

    def get_resolved_device(self) -> str:
        """Resolve device auto-detection safely."""
        if self.DEVICE == "auto":
            if torch.cuda.is_available():
                return "cuda"
            return "cpu"
        return self.DEVICE

@lru_cache()
def get_settings() -> Settings:
    return Settings()
