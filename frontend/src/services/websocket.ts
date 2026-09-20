import { audioPlayer } from './audioPlayer';

export interface Detection {
  class_id: number;
  class_name: string;
  confidence: number;
  bbox: [number, number, number, number];
  track_id?: number;
  team?: string;
  jersey_color?: string;
}

export interface PerformanceMetrics {
  input_fps: number;
  processing_fps: number;
  frame_latency_ms: number;
  vision_latency_ms: number;
  event_latency_ms: number;
  llm_latency_ms: number;
  tts_latency_ms: number;
  e2e_latency_ms: number;
  dropped_frames: number;
  queue_size: number;
  cache_hits: number;
  cache_misses: number;
}

export interface SourceStatus {
  source_type: string;
  status: 'connected' | 'reconnecting' | 'unavailable' | 'idle' | 'paused';
  target: string;
  duration_sec?: number | null;
  video_time_sec: number;
  playback_speed: number;
  fps?: number;
  width?: number;
  height?: number;
}

export interface CricketEvent {
  match_id: string;
  delivery_id: string;
  frame_id: number;
  timestamp: number;
  video_timestamp?: number;
  event_type: string;
  shot_type?: string;
  ball_speed_kmh?: number;
  runs?: number;
  confidence: number;
  metadata?: Record<string, any>;
}

export interface CommentaryItem {
  id: string;
  delivery_id: string;
  event_type: string;
  text: string;
  style: string;
  is_fallback: boolean;
  llm_latency_ms: number;
  timestamp: number;
  video_timestamp?: number;
}

export interface BallState {
  x: number;
  y: number;
  vx: number;
  vy: number;
  confidence: number;
  timestamp: number;
  speed_kmh?: number;
  speed_estimation_available: boolean;
}

export interface PlayerPose {
  track_id: number;
  role: string;
  confidence: number;
  team?: string;
  jersey_color?: string;
  keypoints: Record<string, { x: number; y: number; confidence: number }>;
}

export interface DebugTelemetry {
  current_state: string;
  previous_state: string;
  delivery_id: string;
  ball_x?: number | null;
  ball_y?: number | null;
  ball_vx: number;
  ball_vy: number;
  ball_speed_kmh?: number | null;
  speed_calibrated: boolean;
  is_calibrated?: boolean;
  calibration_conf?: number;
  detected_teams?: string[];
  llm_mode?: string;
  provider_name?: string;
  shot_type?: string | null;
  shot_confidence?: number | null;
  shot_reasoning?: string | null;
  batsman_detected: boolean;
  poses_count: number;
  detections_count: number;
  queue_depth: number;
  dropped_frames: number;
}

export type ConnectionState = 'connected' | 'connecting' | 'disconnected';

type FrameCallback = (data: {
  frameId: number;
  imageBase64: string;
  videoTimestamp?: number;
  sourceStatus?: SourceStatus;
  detections: Detection[];
  ballState?: BallState | null;
  trajectory?: [number, number][];
  poses?: PlayerPose[];
  metrics: PerformanceMetrics;
  debug?: DebugTelemetry;
}) => void;

type EventCallback = (event: CricketEvent) => void;
type CommentaryCallback = (item: CommentaryItem) => void;
type StatusCallback = (status: { state: ConnectionState; message?: string }) => void;

export class WebSocketClient {
  private ws: WebSocket | null = null;
  private url: string;
  private reconnectTimer: any = null;
  private frameListeners: FrameCallback[] = [];
  private eventListeners: EventCallback[] = [];
  private commentaryListeners: CommentaryCallback[] = [];
  private statusListeners: StatusCallback[] = [];
  private connectionState: ConnectionState = 'disconnected';

  constructor(url?: string) {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.hostname || 'localhost';
    this.url = url || `${protocol}//${host}:8000/ws/live`;
  }

  public connect() {
    if (this.ws && (this.ws.readyState === WebSocket.OPEN || this.ws.readyState === WebSocket.CONNECTING)) {
      return;
    }

    this.setConnectionState('connecting');
    try {
      this.ws = new WebSocket(this.url);

      this.ws.onopen = () => {
        this.setConnectionState('connected');
        if (this.reconnectTimer) {
          clearTimeout(this.reconnectTimer);
          this.reconnectTimer = null;
        }
      };

      this.ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data);
          if (msg.type === 'frame') {
            this.frameListeners.forEach(listener => listener({
              frameId: msg.frame_id,
              imageBase64: msg.image_base64,
              videoTimestamp: msg.video_timestamp,
              sourceStatus: msg.source_status,
              detections: msg.detections || [],
              ballState: msg.ball_state || null,
              trajectory: msg.trajectory || [],
              poses: msg.poses || [],
              metrics: msg.metrics || {},
              debug: msg.debug
            }));
          } else if (msg.type === 'event') {
            this.eventListeners.forEach(listener => listener(msg.data));
          } else if (msg.type === 'commentary') {
            this.commentaryListeners.forEach(listener => listener(msg.data));
          } else if (msg.type === 'audio') {
            if (msg.audio_base64) {
              audioPlayer.playBase64Audio(msg.audio_base64);
            }
          } else if (msg.type === 'audio_control') {
            if (msg.action === 'clear_queue') {
              audioPlayer.clearQueue();
            }
          }
        } catch (err) {
          console.error('Failed to parse WebSocket message:', err);
        }
      };

      this.ws.onclose = () => {
        this.setConnectionState('disconnected');
        this.scheduleReconnect();
      };

      this.ws.onerror = () => {
        this.setConnectionState('disconnected');
      };
    } catch (e) {
      this.setConnectionState('disconnected');
      this.scheduleReconnect();
    }
  }

  private scheduleReconnect() {
    if (!this.reconnectTimer) {
      this.reconnectTimer = setTimeout(() => {
        this.reconnectTimer = null;
        this.connect();
      }, 2000);
    }
  }

  private setConnectionState(state: ConnectionState) {
    this.connectionState = state;
    this.statusListeners.forEach(l => l({ state }));
  }

  public sendCommand(action: 'start' | 'stop' | 'pause' | 'resume', extra: Record<string, any> = {}) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ action, ...extra }));
    } else {
      console.warn('Cannot send command: WebSocket not connected.');
    }
  }

  public onFrame(cb: FrameCallback) {
    this.frameListeners.push(cb);
    return () => {
      this.frameListeners = this.frameListeners.filter(l => l !== cb);
    };
  }

  public onEvent(cb: EventCallback) {
    this.eventListeners.push(cb);
    return () => {
      this.eventListeners = this.eventListeners.filter(l => l !== cb);
    };
  }

  public onCommentary(cb: CommentaryCallback) {
    this.commentaryListeners.push(cb);
    return () => {
      this.commentaryListeners = this.commentaryListeners.filter(l => l !== cb);
    };
  }

  public onStatus(cb: StatusCallback) {
    this.statusListeners.push(cb);
    cb({ state: this.connectionState });
    return () => {
      this.statusListeners = this.statusListeners.filter(l => l !== cb);
    };
  }

  public disconnect() {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    this.setConnectionState('disconnected');
  }
}

export const wsService = new WebSocketClient();
