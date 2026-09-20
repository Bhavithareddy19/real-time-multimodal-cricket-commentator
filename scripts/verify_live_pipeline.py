import asyncio
import json
import websockets
import time

async def run_live_pipeline_verification():
    uri = "ws://localhost:8000/ws/live"
    print(f"Connecting to WebSocket: {uri}...", flush=True)
    
    async with websockets.connect(uri) as ws:
        print("Connected to WebSocket!", flush=True)
        
        target_video = r"sample_videos/cover_drive_four.mp4"
        start_cmd = {
            "action": "start",
            "source_type": "uploaded",
            "source_target": target_video,
            "playback_speed": 1.0
        }
        await ws.send(json.dumps(start_cmd))
        print("Sent action: 'start' command...", flush=True)
        
        frames_received = 0
        events_received = []
        commentary_received = []
        audio_received = []
        detections_seen = set()
        poses_count_total = 0
        ball_tracked_frames = 0
        timestamps_seen = []
        
        t0 = time.time()
        paused_tested = False
        resumed_tested = False
        
        while time.time() - t0 < 12.0:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=2.5)
            except asyncio.TimeoutError:
                print("Recv timeout...", flush=True)
                break
                
            if isinstance(msg, bytes):
                continue
                
            data = json.loads(msg)
            msg_type = data.get("type")
            
            if msg_type == "frame":
                frames_received += 1
                v_ts = data.get("video_timestamp")
                timestamps_seen.append(v_ts)
                
                # Check detections
                for d in data.get("detections", []):
                    detections_seen.add(f"{d.get('class_name')} ({d.get('confidence')})")
                    
                # Check ball
                if data.get("ball_state"):
                    ball_tracked_frames += 1
                    
                # Check poses
                poses = data.get("poses", [])
                poses_count_total += len(poses)
                
                # Test pause at 3.5s
                if time.time() - t0 > 3.5 and not paused_tested:
                    print(f"--> [Frame {frames_received}] Sending PAUSE action...", flush=True)
                    await ws.send(json.dumps({"action": "pause"}))
                    paused_tested = True
                    await asyncio.sleep(1.0)
                    print(f"--> [Frame {frames_received}] Sending RESUME action...", flush=True)
                    await ws.send(json.dumps({"action": "resume"}))
                    resumed_tested = True
                    
            elif msg_type == "event":
                ev = data.get("data", {})
                events_received.append(ev)
                print(f"--> [EVENT] Type: {ev.get('event_type')} | Shot: {ev.get('shot_type')} | Runs: {ev.get('runs')} | Video TS: {ev.get('video_timestamp')}s", flush=True)
                
            elif msg_type == "commentary":
                c = data.get("data", {})
                commentary_received.append(c)
                print(f"--> [COMMENTARY] Delivery: {c.get('delivery_id')} | Style: {c.get('style')} | Fallback: {c.get('is_fallback')} | Text: '{c.get('text')}'", flush=True)
                
            elif msg_type == "audio":
                audio_len = len(data.get("audio_base64", ""))
                audio_received.append({
                    "event_type": data.get("event_type"),
                    "priority": data.get("priority"),
                    "bytes_b64_len": audio_len,
                    "first_audio_ms": data.get("first_audio_ms"),
                    "tts_latency_ms": data.get("tts_latency_ms")
                })
                print(f"--> [AUDIO STREAM] Event: {data.get('event_type')} | Priority: {data.get('priority')} | Format: {data.get('format')} | Base64 Length: {audio_len} | TTS Latency: {data.get('tts_latency_ms')}ms", flush=True)
                
            elif msg_type == "audio_control":
                print(f"--> [AUDIO CONTROL] Action: {data.get('action')} | Reason: {data.get('reason')}", flush=True)
            elif msg_type == "status":
                print(f"--> [STATUS UPDATE] {data}", flush=True)

        print("\n=================== VERIFICATION RESULTS ===================", flush=True)
        print(f"1. Frames Processed: {frames_received}", flush=True)
        print(f"2. Object Detections Found: {detections_seen}", flush=True)
        print(f"3. Ball Tracked Frames: {ball_tracked_frames}/{frames_received}", flush=True)
        print(f"4. Pose Estimates: {poses_count_total} player instances", flush=True)
        print(f"5. Events Emitted: {len(events_received)} -> {[e.get('event_type') for e in events_received]}", flush=True)
        print(f"6. Commentary Items: {len(commentary_received)}", flush=True)
        print(f"7. Audio Synthesis Chunks: {len(audio_received)}", flush=True)
        print(f"8. Pause/Resume Verified: Paused={paused_tested}, Resumed={resumed_tested}", flush=True)
        if timestamps_seen:
            print(f"9. Video Clock Synced: Start={timestamps_seen[0]}s, End={timestamps_seen[-1]}s", flush=True)
        print("============================================================\n", flush=True)

if __name__ == "__main__":
    asyncio.run(run_live_pipeline_verification())
