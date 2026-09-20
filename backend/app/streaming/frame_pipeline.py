import asyncio
import time
import logging
import cv2
import numpy as np
from typing import Optional, Literal, Tuple, List, Callable, Dict, Any

from backend.app.config import get_settings
from backend.app.vision.detector import VisionEngine
from backend.app.models.schemas import Detection, PerformanceMetrics, CricketEvent, CommentaryItem
from backend.app.cricket.state_machine import CricketEventStateMachine
from backend.app.cricket.events import EventDispatcher
from backend.app.ai.commentary import CommentaryEngine
from backend.app.streaming.audio_pipeline import AudioPipeline
from backend.app.streaming.video_source import VideoSource, VideoSourceManager

logger = logging.getLogger("streaming.frame_pipeline")

class FramePipeline:
    """
    Asynchronous decoupled frame ingestion and processing pipeline.
    Uses VideoSource abstraction for multi-input flexibility (Upload, URL, Live Stream, Webcam).
    Enforces real-time video clock synchronization and queue backpressure management.
    """

    def __init__(self, vision_engine: Optional[VisionEngine] = None):
        self.settings = get_settings()
        self.vision_engine = vision_engine or VisionEngine()
        
        # Queues: bounded to 2 frames to enforce real-time latency
        self.raw_frame_queue: asyncio.Queue[Tuple[int, float, float, np.ndarray, Dict[str, Any]]] = asyncio.Queue(maxsize=2)
        
        # Lifecycle state
        self.is_running: bool = False
        self.is_paused: bool = False
        self.source_type: str = "uploaded"
        self.source_target: Optional[str] = None
        self.playback_speed: float = 1.0
        self.video_source: Optional[VideoSource] = None
        self.latest_video_time: float = 0.0
        
        # Telemetry & Performance
        self.frame_counter: int = 0
        self.processed_counter: int = 0
        self.dropped_counter: int = 0
        self.current_fps: float = 0.0
        self.processing_fps: float = 0.0
        self.latest_vision_latency_ms: float = 0.0
        self.latest_event_latency_ms: float = 0.0
        self.latest_detections: List[Detection] = []
        self.latest_ball_state: Optional[Any] = None
        self.latest_trajectory: List[Tuple[float, float]] = []
        self.latest_poses: List[Any] = []
        self.latest_event: Optional[CricketEvent] = None
        self.latest_jpeg: Optional[bytes] = None
        self.latest_frame_time: float = 0.0
        
        # State Machine, Event Dispatcher, Commentary & Audio
        self.state_machine = CricketEventStateMachine()
        self.event_dispatcher = EventDispatcher()
        self.commentary_engine = CommentaryEngine()
        self.audio_pipeline = AudioPipeline()
        self.latest_llm_latency_ms: float = 0.0
        self.latest_commentary: Optional[CommentaryItem] = None
        self.commentary_style: str = self.settings.COMMENTARY_STYLE
        
        # Subscribers for processed frames: list of client queues
        self._subscribers: List[asyncio.Queue] = []
        
        # Workers
        self._producer_task: Optional[asyncio.Task] = None
        self._vision_task: Optional[asyncio.Task] = None

    def add_subscriber(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=3)
        self._subscribers.append(q)
        return q

    def remove_subscriber(self, q: asyncio.Queue):
        if q in self._subscribers:
            self._subscribers.remove(q)

    def get_source_status(self) -> Dict[str, Any]:
        """Returns live source metadata, duration, playback clock, and connection status."""
        if self.video_source:
            return self.video_source.get_status_info()
        return {
            "source_type": self.source_type,
            "target": self.source_target or "",
            "status": "paused" if self.is_paused else ("connected" if self.is_running else "idle"),
            "duration_sec": None,
            "video_time_sec": round(self.latest_video_time, 2),
            "playback_speed": self.playback_speed,
            "fps": round(self.current_fps, 1),
            "width": self.settings.FRAME_WIDTH,
            "height": self.settings.FRAME_HEIGHT
        }

    def get_metrics(self) -> PerformanceMetrics:
        tts_lat = self.audio_pipeline.latest_first_audio_ms or self.audio_pipeline.latest_tts_latency_ms
        e2e = round(self.latest_vision_latency_ms + self.latest_event_latency_ms + self.latest_llm_latency_ms + tts_lat, 1)
        return PerformanceMetrics(
            input_fps=round(self.current_fps, 1),
            processing_fps=round(self.processing_fps, 1),
            frame_latency_ms=round(self.latest_vision_latency_ms + self.latest_event_latency_ms, 1),
            vision_latency_ms=round(self.latest_vision_latency_ms, 1),
            event_latency_ms=round(self.latest_event_latency_ms, 1),
            llm_latency_ms=round(self.latest_llm_latency_ms, 1),
            tts_latency_ms=round(tts_lat, 1),
            e2e_latency_ms=e2e,
            dropped_frames=self.dropped_counter,
            queue_size=self.raw_frame_queue.qsize()
        )

    async def _generate_and_dispatch_commentary(self, event: CricketEvent):
        """Asynchronously triggers commentary generation and speech synthesis."""
        try:
            item = await self.commentary_engine.generate(event=event, style=self.commentary_style)
            item.video_timestamp = event.video_timestamp
            self.latest_commentary = item
            self.latest_llm_latency_ms = item.llm_latency_ms
            
            # Broadcast commentary text
            payload = {
                "type": "commentary",
                "data": item.model_dump()
            }
            for sub_q in list(self._subscribers):
                try:
                    sub_q.put_nowait(payload)
                except Exception:
                    pass

            # Synthesize TTS Audio with priority management
            if self.settings.ENABLE_AUDIO:
                def audio_dispatch(audio_payload: Dict[str, Any]):
                    if "video_timestamp" not in audio_payload:
                        audio_payload["video_timestamp"] = event.video_timestamp
                    for sub_q in list(self._subscribers):
                        try:
                            sub_q.put_nowait(audio_payload)
                        except Exception:
                            pass

                await self.audio_pipeline.process_commentary(item, audio_dispatch)

        except Exception as e:
            logger.error(f"Error generating commentary/audio for event {event.event_type}: {e}")

    async def start(
        self,
        source_type: str = "auto",
        source_target: Optional[str] = None,
        playback_speed: float = 1.0
    ):
        """Starts video ingestion via VideoSource abstraction and vision worker tasks."""
        if self.is_running:
            logger.warning("Pipeline is already running. Stopping previous session first.")
            await self.stop()

        self.source_type = source_type
        self.source_target = source_target
        self.playback_speed = playback_speed
        self.is_running = True
        self.is_paused = False
        self.frame_counter = 0
        self.processed_counter = 0
        self.dropped_counter = 0
        self.latest_video_time = 0.0

        # Instantiate concrete VideoSource via manager
        self.video_source = VideoSourceManager.create_source(
            source_type=source_type,
            source_target=source_target,
            playback_speed=playback_speed
        )
        await self.video_source.start()

        # Warm up vision engine in background thread
        await asyncio.to_thread(self.vision_engine.load_model)

        # Launch producer and worker
        self._producer_task = asyncio.create_task(self._frame_producer(), name="FrameProducer")
        self._vision_task = asyncio.create_task(self._vision_worker(), name="VisionWorker")
        logger.info(f"FramePipeline started with {self.video_source.source_type} (target: {source_target})")

    async def stop(self):
        """Gracefully cancels all active pipeline workers and releases hardware resources."""
        self.is_running = False
        self.is_paused = False

        if self._producer_task:
            self._producer_task.cancel()
            try:
                await self._producer_task
            except (asyncio.CancelledError, Exception):
                pass
            self._producer_task = None

        if self._vision_task:
            self._vision_task.cancel()
            try:
                await self._vision_task
            except (asyncio.CancelledError, Exception):
                pass
            self._vision_task = None

        # Release video source
        if self.video_source is not None:
            await self.video_source.stop()
            self.video_source = None

        # Empty queues
        while not self.raw_frame_queue.empty():
            try:
                self.raw_frame_queue.get_nowait()
            except Exception:
                break

        logger.info("FramePipeline cleanly stopped and resources released.")

    def pause(self):
        """Pauses video ingestion and flushes pending audio."""
        self.is_paused = True
        if self.video_source:
            self.video_source.pause()

        # Immediately tell clients to pause/clear pending audio playback
        clear_payload = {
            "type": "audio_control",
            "action": "clear_queue",
            "reason": "Playback paused"
        }
        for q in list(self._subscribers):
            try:
                q.put_nowait(clear_payload)
            except Exception:
                pass
        logger.info("FramePipeline paused.")

    def resume(self):
        """Resumes video ingestion and processing."""
        self.is_paused = False
        if self.video_source:
            self.video_source.resume()
        logger.info("FramePipeline resumed.")

    def set_speed(self, speed: float):
        """Updates playback speed dynamically for recorded/uploaded sources."""
        self.playback_speed = speed
        if self.video_source:
            self.video_source.set_speed(speed)
        # Broadcast speed update to clients
        speed_payload = {
            "type": "status",
            "status": "speed_updated",
            "playback_speed": speed
        }
        for q in list(self._subscribers):
            try:
                q.put_nowait(speed_payload)
            except Exception:
                pass

    async def _frame_producer(self):
        """Asynchronous frame producer pulling frames from VideoSource."""
        fps_start_time = time.perf_counter()
        fps_frame_count = 0

        try:
            if not self.video_source:
                return

            async for frame_id, video_ts, frame, meta in self.video_source.frames():
                if not self.is_running:
                    break

                if self.is_paused:
                    await asyncio.sleep(0.05)
                    continue

                self.frame_counter = frame_id
                self.latest_video_time = video_ts
                fps_frame_count += 1

                # Calculate measured input FPS periodically
                elapsed = time.perf_counter() - fps_start_time
                if elapsed >= 1.0:
                    self.current_fps = fps_frame_count / elapsed
                    fps_frame_count = 0
                    fps_start_time = time.perf_counter()

                # Queue frame with backpressure / frame dropping
                item = (frame_id, video_ts, time.perf_counter(), frame, meta)
                try:
                    self.raw_frame_queue.put_nowait(item)
                except asyncio.QueueFull:
                    # Drop oldest frame to ensure fresh real-time processing
                    try:
                        self.raw_frame_queue.get_nowait()
                        self.dropped_counter += 1
                        self.raw_frame_queue.put_nowait(item)
                    except Exception:
                        pass

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error in frame producer: {e}", exc_info=True)

    async def _vision_worker(self):
        """Asynchronous vision worker consuming frames and executing YOLO detection."""
        proc_start_time = time.perf_counter()
        proc_count = 0

        try:
            while self.is_running:
                try:
                    frame_id, video_ts, timestamp, frame, meta = await asyncio.wait_for(
                        self.raw_frame_queue.get(),
                        timeout=0.5
                    )
                except asyncio.TimeoutError:
                    continue

                # Run YOLO detection
                detections, inference_time_ms = await self.vision_engine.detect(frame)

                # Run Ball Tracking, Player Tracking, and Pose Estimation
                # Run pose every 2 frames on CPU to balance FPS
                run_pose = (self.processed_counter % 2 == 0)
                ball_state, trajectory, poses, track_latency_ms = await self.vision_engine.process_frame_tracking(
                    frame=frame,
                    detections=detections,
                    timestamp=timestamp,
                    run_pose=run_pose
                )

                self.latest_vision_latency_ms = inference_time_ms + track_latency_ms
                self.latest_detections = detections
                self.latest_ball_state = ball_state
                self.latest_trajectory = trajectory
                self.latest_poses = poses
                self.processed_counter += 1
                proc_count += 1

                # Autonomous pitch/boundary calibration on early frame
                if self.processed_counter == 5:
                    self.state_machine.update_dynamic_calibration(frame)

                # Execute Cricket Event State Machine with synchronized video timestamp
                t_ev_start = time.perf_counter()
                event = self.state_machine.process_frame(
                    frame_id=frame_id,
                    ball_state=ball_state,
                    trajectory=trajectory,
                    poses=poses,
                    detections=detections,
                    timestamp=timestamp,
                    video_timestamp=video_ts
                )
                self.latest_event_latency_ms = (time.perf_counter() - t_ev_start) * 1000.0

                if event:
                    self.latest_event = event
                    self.event_dispatcher.record_event(event)
                    # Trigger commentary generation asynchronously
                    asyncio.create_task(self._generate_and_dispatch_commentary(event))

                # Calculate processing FPS
                proc_elapsed = time.perf_counter() - proc_start_time
                if proc_elapsed >= 1.0:
                    self.processing_fps = proc_count / proc_elapsed
                    proc_count = 0
                    proc_start_time = time.perf_counter()

                # Render overlays with active state and boundary
                annotated = self.vision_engine.draw_overlays(
                    frame=frame,
                    detections=detections,
                    ball_state=ball_state,
                    trajectory=trajectory,
                    poses=poses,
                    fps=self.processing_fps or self.current_fps,
                    latency_ms=self.latest_vision_latency_ms,
                    current_event=self.state_machine.current_state,
                    boundary_polygon=self.state_machine.boundary_polygon
                )

                # Encode to JPEG for real-time WebSocket transport
                ret, jpeg_buffer = cv2.imencode('.jpg', annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                if ret:
                    jpeg_bytes = jpeg_buffer.tobytes()
                    self.latest_jpeg = jpeg_bytes
                    self.latest_frame_time = timestamp

                    # Dispatch to all active client subscriber queues
                    payload = {
                        "type": "frame",
                        "frame_id": frame_id,
                        "timestamp": timestamp,
                        "video_timestamp": round(video_ts, 2),
                        "source_status": self.get_source_status(),
                        "detections": [d.model_dump() for d in detections],
                        "ball_state": ball_state.model_dump() if ball_state else None,
                        "trajectory": trajectory,
                        "poses": [p.model_dump() for p in poses],
                        "metrics": self.get_metrics().model_dump(),
                        "debug": {
                            "current_state": self.state_machine.current_state,
                            "previous_state": getattr(self.state_machine, "previous_state", "IDLE"),
                            "delivery_id": self.state_machine.delivery_id,
                            "ball_x": round(ball_state.x, 1) if ball_state else None,
                            "ball_y": round(ball_state.y, 1) if ball_state else None,
                            "ball_vx": round(ball_state.vx, 2) if ball_state else 0.0,
                            "ball_vy": round(ball_state.vy, 2) if ball_state else 0.0,
                            "ball_speed_kmh": ball_state.speed_kmh if ball_state and ball_state.speed_estimation_available else None,
                            "speed_calibrated": ball_state.speed_estimation_available if ball_state else False,
                            "is_calibrated": getattr(self.state_machine.calibrator, "is_calibrated", False),
                            "calibration_conf": getattr(self.state_machine.calibrator, "calibration_confidence", 0.0),
                            "detected_teams": list({d.team for d in detections if getattr(d, "team", None) and d.team not in ["Unknown", "Club / Custom"]}),
                            "llm_mode": "REAL AI (LLM)" if (self.commentary_engine.provider and self.commentary_engine.provider.is_configured()) else "FALLBACK (Templates)",
                            "provider_name": self.commentary_engine.provider.__class__.__name__ if self.commentary_engine.provider else "Fallback",
                            "shot_type": self.state_machine.current_shot.shot_type if self.state_machine.current_shot else None,
                            "shot_confidence": round(self.state_machine.current_shot.confidence, 2) if self.state_machine.current_shot else None,
                            "shot_reasoning": self.state_machine.current_shot.reasoning if self.state_machine.current_shot else None,
                            "batsman_detected": any(p.role == "batsman" for p in poses),
                            "poses_count": len(poses),
                            "detections_count": len(detections),
                            "queue_depth": self.raw_frame_queue.qsize(),
                            "dropped_frames": self.dropped_counter
                        },
                        "jpeg": jpeg_bytes
                    }
                    for sub_q in list(self._subscribers):
                        try:
                            # Drop oldest if subscriber is slow
                            if sub_q.full():
                                try:
                                    sub_q.get_nowait()
                                except Exception:
                                    pass
                            sub_q.put_nowait(payload)
                            
                            # If event occurred, dispatch event payload
                            if event:
                                ev_payload = {
                                    "type": "event",
                                    "data": event.model_dump()
                                }
                                sub_q.put_nowait(ev_payload)
                        except Exception:
                            pass

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error in vision worker: {e}", exc_info=True)
