import os
import time
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import AsyncGenerator, Tuple, Optional, Dict, Any, Literal
import cv2
import numpy as np

from backend.app.config import get_settings

logger = logging.getLogger("streaming.video_source")

class VideoSource(ABC):
    """
    Abstract base class for all video ingestion sources.
    Decouples video origin from downstream computer vision and commentary systems.
    """

    def __init__(self, source_target: str, playback_speed: float = 1.0):
        self.source_target: str = source_target
        self.playback_speed: float = max(0.25, min(playback_speed, 4.0))
        self.is_running: bool = False
        self.is_paused: bool = False
        self.status: Literal["idle", "connected", "reconnecting", "unavailable", "paused"] = "idle"
        self.duration_sec: Optional[float] = None
        self.current_video_time: float = 0.0
        self.fps: float = 30.0
        self.width: int = 640
        self.height: int = 480
        self.frame_counter: int = 0
        self.settings = get_settings()

    @property
    @abstractmethod
    def source_type(self) -> str:
        """Returns source type identifier."""
        pass

    @abstractmethod
    async def start(self) -> None:
        """Initializes hardware capture or network connections."""
        pass

    @abstractmethod
    async def frames(self) -> AsyncGenerator[Tuple[int, float, np.ndarray, Dict[str, Any]], None]:
        """
        Yields (frame_id, video_timestamp, frame_image, metadata).
        Paced according to playback speed and source clock.
        """
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Gracefully closes capture devices and releases resources."""
        pass

    def pause(self) -> None:
        self.is_paused = True
        self.status = "paused"

    def resume(self) -> None:
        self.is_paused = False
        self.status = "connected"

    def set_speed(self, speed: float) -> None:
        self.playback_speed = max(0.25, min(speed, 4.0))
        logger.info(f"Playback speed changed to {self.playback_speed}x for source {self.source_target}")

    def get_status_info(self) -> Dict[str, Any]:
        return {
            "source_type": self.source_type,
            "target": self.source_target,
            "status": self.status,
            "duration_sec": self.duration_sec,
            "video_time_sec": round(self.current_video_time, 2),
            "playback_speed": self.playback_speed,
            "fps": round(self.fps, 1),
            "width": self.width,
            "height": self.height
        }


class UploadedVideoSource(VideoSource):
    """
    Ingests recorded/uploaded local video files (MP4, MOV, MKV, AVI).
    Guaranteed primary working mode. Paced in real-time according to playback speed.
    """

    def __init__(self, file_path: str, playback_speed: float = 1.0, loop: bool = True):
        super().__init__(file_path, playback_speed)
        self.loop = loop
        self._cap: Optional[cv2.VideoCapture] = None

    @property
    def source_type(self) -> str:
        return "uploaded"

    async def start(self) -> None:
        if not os.path.exists(self.source_target):
            self.status = "unavailable"
            raise FileNotFoundError(f"Video file not found: {self.source_target}")

        self._cap = await asyncio.to_thread(cv2.VideoCapture, self.source_target)
        if not self._cap or not self._cap.isOpened():
            self.status = "unavailable"
            raise RuntimeError(f"Failed to open video file: {self.source_target}")

        # Extract video properties
        raw_fps = self._cap.get(cv2.CAP_PROP_FPS)
        self.fps = raw_fps if 1.0 <= raw_fps <= 120.0 else 30.0
        total_frames = self._cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.duration_sec = total_frames / self.fps if total_frames > 0 else None
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH) or self.settings.FRAME_WIDTH)
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or self.settings.FRAME_HEIGHT)

        self.is_running = True
        self.is_paused = False
        self.status = "connected"
        self.frame_counter = 0
        logger.info(f"UploadedVideoSource started: {self.source_target} ({self.fps:.1f} FPS, {self.duration_sec:.1f}s)")

    async def frames(self) -> AsyncGenerator[Tuple[int, float, np.ndarray, Dict[str, Any]], None]:
        if not self._cap or not self.is_running:
            return

        frame_interval = 1.0 / self.fps

        while self.is_running:
            if self.is_paused:
                await asyncio.sleep(0.08)
                continue

            t_start = time.perf_counter()

            # Handle fast playback (e.g. 2x): skip intermediate frames to avoid lag
            if self.playback_speed > 1.2:
                skip_steps = int(round(self.playback_speed)) - 1
                for _ in range(skip_steps):
                    ret_skip = await asyncio.to_thread(self._cap.grab)
                    if not ret_skip:
                        break
                    self.frame_counter += 1

            ret, frame = await asyncio.to_thread(self._cap.read)
            if not ret or frame is None:
                if self.loop:
                    await asyncio.to_thread(self._cap.set, cv2.CAP_PROP_POS_FRAMES, 0)
                    self.current_video_time = 0.0
                    await asyncio.sleep(0.02)
                    continue
                else:
                    logger.info("Uploaded video reached EOF.")
                    self.status = "idle"
                    break

            self.frame_counter += 1
            pos_msec = self._cap.get(cv2.CAP_PROP_POS_MSEC)
            self.current_video_time = pos_msec / 1000.0 if pos_msec >= 0 else (self.frame_counter / self.fps)

            # Resize if dimensions exceed configured maximums
            h, w = frame.shape[:2]
            if w > self.settings.FRAME_WIDTH or h > self.settings.FRAME_HEIGHT:
                frame = cv2.resize(frame, (self.settings.FRAME_WIDTH, self.settings.FRAME_HEIGHT))

            metadata = {
                "source_type": self.source_type,
                "video_timestamp": self.current_video_time,
                "duration_sec": self.duration_sec,
                "playback_speed": self.playback_speed
            }

            yield (self.frame_counter, self.current_video_time, frame, metadata)

            # Accurate real-time pacing based on playback_speed
            target_interval = frame_interval / self.playback_speed
            elapsed = time.perf_counter() - t_start
            sleep_needed = target_interval - elapsed
            if sleep_needed > 0.002:
                await asyncio.sleep(sleep_needed)
            else:
                await asyncio.sleep(0.001)

    async def stop(self) -> None:
        self.is_running = False
        self.status = "idle"
        if self._cap is not None:
            await asyncio.to_thread(self._cap.release)
            self._cap = None
        logger.info(f"UploadedVideoSource stopped: {self.source_target}")


class DirectURLVideoSource(VideoSource):
    """
    Streams directly from an accessible HTTP/HTTPS video URL.
    Validates accessibility beforehand; respects DRM and access control.
    """

    def __init__(self, url: str, playback_speed: float = 1.0):
        super().__init__(url, playback_speed)
        self._cap: Optional[cv2.VideoCapture] = None

    @property
    def source_type(self) -> str:
        return "direct_url"

    async def start(self) -> None:
        self._cap = await asyncio.to_thread(cv2.VideoCapture, self.source_target)
        if not self._cap or not self._cap.isOpened():
            self.status = "unavailable"
            raise RuntimeError(f"Cannot access online video URL: {self.source_target}")

        raw_fps = self._cap.get(cv2.CAP_PROP_FPS)
        self.fps = raw_fps if 1.0 <= raw_fps <= 120.0 else 30.0
        total_frames = self._cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.duration_sec = total_frames / self.fps if total_frames > 0 else None
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH) or self.settings.FRAME_WIDTH)
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or self.settings.FRAME_HEIGHT)

        self.is_running = True
        self.is_paused = False
        self.status = "connected"
        self.frame_counter = 0
        logger.info(f"DirectURLVideoSource connected: {self.source_target}")

    async def frames(self) -> AsyncGenerator[Tuple[int, float, np.ndarray, Dict[str, Any]], None]:
        if not self._cap or not self.is_running:
            return

        frame_interval = 1.0 / self.fps

        while self.is_running:
            if self.is_paused:
                await asyncio.sleep(0.08)
                continue

            t_start = time.perf_counter()

            ret, frame = await asyncio.to_thread(self._cap.read)
            if not ret or frame is None:
                logger.info("Online video stream finished or paused.")
                self.status = "idle"
                break

            self.frame_counter += 1
            pos_msec = self._cap.get(cv2.CAP_PROP_POS_MSEC)
            self.current_video_time = pos_msec / 1000.0 if pos_msec >= 0 else (self.frame_counter / self.fps)

            h, w = frame.shape[:2]
            if w > self.settings.FRAME_WIDTH or h > self.settings.FRAME_HEIGHT:
                frame = cv2.resize(frame, (self.settings.FRAME_WIDTH, self.settings.FRAME_HEIGHT))

            metadata = {
                "source_type": self.source_type,
                "video_timestamp": self.current_video_time,
                "duration_sec": self.duration_sec,
                "playback_speed": self.playback_speed
            }

            yield (self.frame_counter, self.current_video_time, frame, metadata)

            target_interval = frame_interval / self.playback_speed
            elapsed = time.perf_counter() - t_start
            sleep_needed = target_interval - elapsed
            if sleep_needed > 0.002:
                await asyncio.sleep(sleep_needed)
            else:
                await asyncio.sleep(0.001)

    async def stop(self) -> None:
        self.is_running = False
        self.status = "idle"
        if self._cap is not None:
            await asyncio.to_thread(self._cap.release)
            self._cap = None


class LiveStreamSource(VideoSource):
    """
    Ingests live accessible streams (RTSP, RTMP, HLS, live HTTP).
    Continuously yields incoming frames and automatically reconnects on network drop.
    """

    def __init__(self, stream_url: str, max_reconnect_attempts: int = 5):
        super().__init__(stream_url, 1.0)
        self.max_reconnect_attempts = max_reconnect_attempts
        self._cap: Optional[cv2.VideoCapture] = None

    @property
    def source_type(self) -> str:
        return "livestream"

    async def _connect_stream(self) -> bool:
        if self._cap is not None:
            await asyncio.to_thread(self._cap.release)
            self._cap = None

        self._cap = await asyncio.to_thread(cv2.VideoCapture, self.source_target)
        return bool(self._cap and self._cap.isOpened())

    async def start(self) -> None:
        self.status = "reconnecting"
        connected = await self._connect_stream()
        if not connected:
            self.status = "unavailable"
            raise ConnectionError(f"Unable to connect to live stream: {self.source_target}")

        raw_fps = self._cap.get(cv2.CAP_PROP_FPS)
        self.fps = raw_fps if 1.0 <= raw_fps <= 120.0 else 30.0
        self.duration_sec = None  # Continuous live stream
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH) or self.settings.FRAME_WIDTH)
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or self.settings.FRAME_HEIGHT)

        self.is_running = True
        self.is_paused = False
        self.status = "connected"
        self.frame_counter = 0
        logger.info(f"LiveStreamSource connected: {self.source_target}")

    async def frames(self) -> AsyncGenerator[Tuple[int, float, np.ndarray, Dict[str, Any]], None]:
        reconnect_count = 0

        while self.is_running:
            if self.is_paused:
                await asyncio.sleep(0.08)
                continue

            if not self._cap or not self._cap.isOpened():
                if reconnect_count >= self.max_reconnect_attempts:
                    self.status = "unavailable"
                    logger.error("LiveStream max reconnect attempts reached. Stream unavailable.")
                    break

                self.status = "reconnecting"
                reconnect_count += 1
                wait_time = min(2.0 * reconnect_count, 10.0)
                logger.warning(f"LiveStream disconnected. Reconnecting in {wait_time:.1f}s (Attempt {reconnect_count}/{self.max_reconnect_attempts})...")
                await asyncio.sleep(wait_time)

                if await self._connect_stream():
                    logger.info("LiveStream successfully reconnected!")
                    self.status = "connected"
                    reconnect_count = 0
                continue

            ret, frame = await asyncio.to_thread(self._cap.read)
            if not ret or frame is None:
                # Mark disconnected to trigger reconnect loop on next iteration
                if self._cap:
                    await asyncio.to_thread(self._cap.release)
                    self._cap = None
                continue

            self.frame_counter += 1
            now = time.time()
            self.current_video_time = self.frame_counter / self.fps

            h, w = frame.shape[:2]
            if w > self.settings.FRAME_WIDTH or h > self.settings.FRAME_HEIGHT:
                frame = cv2.resize(frame, (self.settings.FRAME_WIDTH, self.settings.FRAME_HEIGHT))

            metadata = {
                "source_type": self.source_type,
                "video_timestamp": self.current_video_time,
                "duration_sec": None,
                "status": self.status
            }

            yield (self.frame_counter, self.current_video_time, frame, metadata)
            await asyncio.sleep(0.001)

    async def stop(self) -> None:
        self.is_running = False
        self.status = "idle"
        if self._cap is not None:
            await asyncio.to_thread(self._cap.release)
            self._cap = None


class WebcamSource(VideoSource):
    """
    Ingests live camera frames from local hardware webcam.
    Optional additional input mode.
    """

    def __init__(self, device_index: int = 0):
        super().__init__(str(device_index), 1.0)
        self.device_index = device_index
        self._cap: Optional[cv2.VideoCapture] = None

    @property
    def source_type(self) -> str:
        return "webcam"

    async def start(self) -> None:
        # On Windows, cv2.CAP_DSHOW provides fastest device startup
        def open_cam():
            cap = cv2.VideoCapture(self.device_index, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(self.device_index)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.settings.FRAME_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.settings.FRAME_HEIGHT)
                cap.set(cv2.CAP_PROP_FPS, self.settings.TARGET_FPS)
            return cap

        self._cap = await asyncio.to_thread(open_cam)
        if not self._cap or not self._cap.isOpened():
            self.status = "unavailable"
            raise RuntimeError(f"Cannot open webcam device index {self.device_index}")

        self.fps = self._cap.get(cv2.CAP_PROP_FPS) or float(self.settings.TARGET_FPS)
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH) or self.settings.FRAME_WIDTH)
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or self.settings.FRAME_HEIGHT)
        self.duration_sec = None

        self.is_running = True
        self.is_paused = False
        self.status = "connected"
        self.frame_counter = 0
        logger.info(f"WebcamSource started on device {self.device_index}")

    async def frames(self) -> AsyncGenerator[Tuple[int, float, np.ndarray, Dict[str, Any]], None]:
        if not self._cap or not self.is_running:
            return

        while self.is_running:
            if self.is_paused:
                await asyncio.sleep(0.08)
                continue

            ret, frame = await asyncio.to_thread(self._cap.read)
            if not ret or frame is None:
                await asyncio.sleep(0.02)
                continue

            self.frame_counter += 1
            now = time.time()
            self.current_video_time = self.frame_counter / self.fps

            h, w = frame.shape[:2]
            if w > self.settings.FRAME_WIDTH or h > self.settings.FRAME_HEIGHT:
                frame = cv2.resize(frame, (self.settings.FRAME_WIDTH, self.settings.FRAME_HEIGHT))

            metadata = {
                "source_type": self.source_type,
                "video_timestamp": self.current_video_time,
                "duration_sec": None
            }

            yield (self.frame_counter, self.current_video_time, frame, metadata)
            await asyncio.sleep(0.001)

    async def stop(self) -> None:
        self.is_running = False
        self.status = "idle"
        if self._cap is not None:
            await asyncio.to_thread(self._cap.release)
            self._cap = None


class VideoSourceManager:
    """
    Factory and validator for all video source types.
    Ensures URL sources are verified without attempting to bypass DRM or authentication.
    """

    @classmethod
    def detect_source_type(cls, target: str) -> str:
        """Heuristically infers the source type from string target."""
        if not target:
            return "uploaded"

        target_clean = str(target).strip()
        if target_clean.isdigit():
            return "webcam"

        lowered = target_clean.lower()
        if lowered.startswith("rtsp://") or lowered.startswith("rtmp://"):
            return "livestream"

        if lowered.startswith("http://") or lowered.startswith("https://"):
            if ".m3u8" in lowered or "live" in lowered:
                return "livestream"
            return "direct_url"

        if os.path.exists(target_clean):
            return "uploaded"

        return "direct_url"

    @classmethod
    def create_source(
        cls,
        source_type: str,
        source_target: Optional[str] = None,
        playback_speed: float = 1.0,
        **kwargs
    ) -> VideoSource:
        """
        Creates and returns a concrete VideoSource instance.
        """
        resolved_type = source_type.lower()
        target = source_target or ""

        if resolved_type in ["auto", "detect"]:
            resolved_type = cls.detect_source_type(target)

        if resolved_type in ["uploaded", "video"]:
            if not target:
                # Default to configured sample video
                settings = get_settings()
                target = settings.VIDEO_FILE_PATH or "sample_videos/cover_drive_four.mp4"
            return UploadedVideoSource(file_path=target, playback_speed=playback_speed)

        elif resolved_type in ["direct_url", "url", "online"]:
            return DirectURLVideoSource(url=target, playback_speed=playback_speed)

        elif resolved_type in ["livestream", "rtsp", "live"]:
            return LiveStreamSource(stream_url=target)

        elif resolved_type == "webcam":
            idx = int(target) if target.isdigit() else 0
            return WebcamSource(device_index=idx)

        else:
            logger.warning(f"Unrecognized source type '{source_type}', falling back to UploadedVideoSource.")
            return UploadedVideoSource(file_path=target, playback_speed=playback_speed)

    @classmethod
    def validate_source(cls, url_or_path: str, source_type: Optional[str] = None) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Checks if the video target is technically accessible by OpenCV/FFmpeg without blocking.
        Does not attempt to bypass DRM or authentication.
        """
        if not url_or_path or not str(url_or_path).strip():
            return False, "No URL or file path provided.", {}

        target = str(url_or_path).strip()
        detected_type = source_type if source_type and source_type != "auto" else cls.detect_source_type(target)

        if detected_type == "uploaded":
            if not os.path.exists(target):
                return False, f"File does not exist on server: {target}", {}
            cap = cv2.VideoCapture(target)
        elif detected_type == "webcam":
            idx = int(target) if target.isdigit() else 0
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(idx)
        else:
            # Online URL or Stream
            cap = cv2.VideoCapture(target)

        try:
            if not cap.isOpened():
                return (
                    False,
                    "This video source cannot be accessed by the application. Try uploading the video file or using a directly accessible stream.",
                    {}
                )

            ret, frame = cap.read()
            if not ret or frame is None:
                return (
                    False,
                    "This video source cannot be accessed by the application. Try uploading the video file or using a directly accessible stream.",
                    {}
                )

            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            duration = (total_frames / fps) if total_frames > 0 else None
            h, w = frame.shape[:2]

            return (
                True,
                "Source successfully verified and accessible.",
                {
                    "source_type": detected_type,
                    "fps": round(fps, 1),
                    "duration_sec": round(duration, 1) if duration else None,
                    "width": w,
                    "height": h
                }
            )
        finally:
            cap.release()
