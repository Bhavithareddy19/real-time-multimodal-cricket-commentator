import asyncio
import json
import logging
import base64
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Set

from backend.app.streaming.frame_pipeline import FramePipeline

logger = logging.getLogger("api.websocket")
ws_router = APIRouter()

# Global pipeline reference set in main.py
pipeline_instance: FramePipeline = None

def set_pipeline(pipeline: FramePipeline):
    global pipeline_instance
    pipeline_instance = pipeline

class ConnectionManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"Client connected. Total active: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(f"Client disconnected. Remaining: {len(self.active_connections)}")

    async def broadcast_json(self, data: dict):
        dead = []
        for connection in list(self.active_connections):
            try:
                await connection.send_json(data)
            except Exception:
                dead.append(connection)
        for d in dead:
            self.disconnect(d)

manager = ConnectionManager()

@ws_router.websocket("/ws/live")
async def websocket_live_endpoint(websocket: WebSocket):
    """
    Unified high-performance WebSocket stream delivering live frames,
    detections, telemetry metrics, and receiving interactive control commands.
    """
    await manager.connect(websocket)
    sub_queue = pipeline_instance.add_subscriber()

    async def receiver_task():
        try:
            while True:
                data = await websocket.receive_text()
                msg = json.loads(data)
                cmd = msg.get("action")
                if cmd == "start":
                    source_type = msg.get("source_type", "auto")
                    source_target = msg.get("source_target")
                    speed = float(msg.get("playback_speed", msg.get("speed", 1.0)))
                    await pipeline_instance.start(source_type=source_type, source_target=source_target, playback_speed=speed)
                    await websocket.send_json({"type": "status", "status": "started", "source": source_type, "playback_speed": speed})
                elif cmd == "stop":
                    await pipeline_instance.stop()
                    await websocket.send_json({"type": "status", "status": "stopped"})
                elif cmd == "pause":
                    pipeline_instance.pause()
                    await websocket.send_json({"type": "status", "status": "paused"})
                elif cmd == "resume":
                    pipeline_instance.resume()
                    await websocket.send_json({"type": "status", "status": "resumed"})
                elif cmd == "set_speed":
                    speed = float(msg.get("playback_speed", msg.get("speed", 1.0)))
                    pipeline_instance.set_speed(speed)
                    await websocket.send_json({"type": "status", "status": "speed_updated", "playback_speed": speed})
                elif cmd == "set_style":
                    new_style = msg.get("style", "PROFESSIONAL")
                    pipeline_instance.commentary_style = new_style
                    await websocket.send_json({"type": "status", "status": "style_updated", "style": new_style})
                elif cmd == "set_llm_config":
                    from backend.app.ai.llm import create_llm_provider
                    prov_name = msg.get("provider", "auto")
                    key = msg.get("api_key")
                    model = msg.get("model")
                    new_prov = create_llm_provider(prov_name, api_key=key, model=model)
                    pipeline_instance.commentary_engine.set_provider(new_prov)
                    await websocket.send_json({
                        "type": "status",
                        "status": "llm_updated",
                        "provider": new_prov.__class__.__name__,
                        "is_configured": new_prov.is_configured()
                    })
        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.warning(f"Error handling incoming WS message: {e}")

    receiver = asyncio.create_task(receiver_task())

    try:
        while True:
            # Await new processed frame from pipeline subscriber queue
            item = await sub_queue.get()
            jpeg_bytes = item.pop("jpeg", None)
            
            # Pack frame as base64 or send as structured message
            if jpeg_bytes:
                item["image_base64"] = base64.b64encode(jpeg_bytes).decode('ascii')
            
            await websocket.send_json(item)

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected by client.")
    except asyncio.CancelledError:
        pass
    except Exception as e:
        logger.warning(f"WebSocket streaming loop exception: {e}")
    finally:
        receiver.cancel()
        pipeline_instance.remove_subscriber(sub_queue)
        manager.disconnect(websocket)

@ws_router.websocket("/ws/video")
async def websocket_video_only(websocket: WebSocket):
    """Dedicated video-only stream."""
    await websocket_live_endpoint(websocket)

@ws_router.websocket("/ws/metrics")
async def websocket_metrics_endpoint(websocket: WebSocket):
    """Dedicated performance metrics stream (10 Hz)."""
    await websocket.accept()
    try:
        while True:
            metrics = pipeline_instance.get_metrics()
            await websocket.send_json({"type": "metrics", "data": metrics.model_dump()})
            await asyncio.sleep(0.1)
    except (WebSocketDisconnect, asyncio.CancelledError):
        pass
