from typing import Literal, Optional, List, Dict, Any
from pydantic import BaseModel, Field
import time
import uuid

class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

class Detection(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2]
    track_id: Optional[int] = None
    team: Optional[str] = None
    jersey_color: Optional[str] = None

class PoseKeypoint(BaseModel):
    x: float
    y: float
    confidence: float

class PlayerPose(BaseModel):
    track_id: int
    role: Literal["batsman", "bowler", "fielder", "umpire", "unknown"] = "unknown"
    confidence: float
    keypoints: Dict[str, PoseKeypoint]
    team: Optional[str] = None
    jersey_color: Optional[str] = None

class BallState(BaseModel):
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    confidence: float
    timestamp: float
    speed_kmh: Optional[float] = None
    speed_estimation_available: bool = False

class CricketEvent(BaseModel):
    match_id: str
    delivery_id: str
    frame_id: int
    timestamp: float = Field(default_factory=time.time)
    video_timestamp: Optional[float] = None
    event_type: Literal[
        "IDLE",
        "BOWLER_RUNUP",
        "DELIVERY",
        "SHOT_PLAYED",
        "DOT_BALL",
        "SINGLE",
        "DOUBLE",
        "TRIPLE",
        "FOUR",
        "SIX",
        "WICKET",
        "MISSED_BALL"
    ]
    shot_type: Optional[str] = None
    ball_speed_kmh: Optional[float] = None
    runs: Optional[int] = None
    confidence: float = 1.0
    metadata: Dict[str, Any] = Field(default_factory=dict)

class CommentaryItem(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    delivery_id: str
    event_type: str
    text: str
    style: str = "PROFESSIONAL"
    is_fallback: bool = False
    tokens: Optional[List[str]] = None
    llm_latency_ms: float = 0.0
    timestamp: float = Field(default_factory=time.time)
    video_timestamp: Optional[float] = None

class SourceStatus(BaseModel):
    source_type: str = "uploaded"  # uploaded, direct_url, livestream, webcam
    status: str = "idle"          # connected, reconnecting, unavailable, idle, paused
    target: str = ""
    duration_sec: Optional[float] = None
    video_time_sec: float = 0.0
    playback_speed: float = 1.0

class PerformanceMetrics(BaseModel):
    input_fps: float = 0.0
    processing_fps: float = 0.0
    frame_latency_ms: float = 0.0
    vision_latency_ms: float = 0.0
    event_latency_ms: float = 0.0
    llm_latency_ms: float = 0.0
    tts_latency_ms: float = 0.0
    e2e_latency_ms: float = 0.0
    dropped_frames: int = 0
    queue_size: int = 0
    cache_hits: int = 0
    cache_misses: int = 0

class MatchScorecard(BaseModel):
    match_id: str = "match_001"
    overs: float = 0.0
    runs: int = 0
    wickets: int = 0
    current_delivery: int = 0
    recent_events: List[str] = Field(default_factory=list)

class SessionStartRequest(BaseModel):
    source_type: Literal["uploaded", "video", "direct_url", "livestream", "rtsp", "webcam", "auto"] = "auto"
    source_target: Optional[str] = None
    playback_speed: float = 1.0
    commentary_style: str = "PROFESSIONAL"
    enable_audio: bool = True
    enable_llm: bool = True

class SessionStatusResponse(BaseModel):
    is_running: bool
    source_type: str
    frames_processed: int
    scorecard: MatchScorecard
    metrics: PerformanceMetrics
    source_status: Optional[SourceStatus] = None

class SourceValidationRequest(BaseModel):
    url: str
    source_type: Optional[str] = "auto"

class SourceValidationResponse(BaseModel):
    valid: bool
    source_type: str
    message: str
    duration_sec: Optional[float] = None
    fps: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
