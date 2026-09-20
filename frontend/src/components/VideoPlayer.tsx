import React, { useRef, useEffect, useState } from 'react';
import {
  Play,
  Square,
  Pause,
  Video,
  Eye,
  EyeOff,
  Upload,
  Link as LinkIcon,
  CheckCircle2,
  AlertCircle,
  Clock,
  Zap,
  Activity,
  RefreshCw,
  Film,
  Layers,
  Sparkles,
  Terminal
} from 'lucide-react';
import { wsService, Detection, PerformanceMetrics, SourceStatus, CricketEvent, DebugTelemetry } from '../services/websocket';
import { EventDebugPanel } from './EventDebugPanel';
import { AISettingsModal } from './AISettingsModal';

interface Props {
  onMetricsUpdate?: (m: PerformanceMetrics) => void;
  onSessionStateChange?: (running: boolean) => void;
}

export const VideoPlayer: React.FC<Props> = ({ onMetricsUpdate, onSessionStateChange }) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  // Ingestion & Control State
  const [isRunning, setIsRunning] = useState<boolean>(false);
  const [isPaused, setIsPaused] = useState<boolean>(false);
  const [inputMode, setInputMode] = useState<'upload' | 'url' | 'sample'>('sample');
  const [sourceType, setSourceType] = useState<string>('auto');
  const [urlInput, setUrlInput] = useState<string>('');
  const [selectedTarget, setSelectedTarget] = useState<string>('sample_videos/cover_drive_four.mp4');
  const [uploadedFile, setUploadedFile] = useState<{ name: string; path: string; duration?: number } | null>(null);
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const [validationError, setValidationError] = useState<string | null>(null);
  const [isValidating, setIsValidating] = useState<boolean>(false);
  const [sourcesList, setSourcesList] = useState<any[]>([]);

  // AI Configuration Modal State
  const [isAISettingsOpen, setIsAISettingsOpen] = useState<boolean>(false);
  const [activeLLMProvider, setActiveLLMProvider] = useState<string>(() => {
    const saved = localStorage.getItem('cricket_llm_provider');
    if (saved === 'google-gemini') return 'Google Gemini';
    if (saved === 'openai') return 'OpenAI';
    return 'Local Fallback';
  });

  // Real-time Playback and Sync State
  const [playbackSpeed, setPlaybackSpeed] = useState<number>(1.0);
  const [videoTime, setVideoTime] = useState<number>(0.0);
  const [sourceStatus, setSourceStatus] = useState<SourceStatus | null>(null);
  const [currentEvent, setCurrentEvent] = useState<CricketEvent | null>(null);
  const [showOverlays, setShowOverlays] = useState<boolean>(true);
  const [showDebugHud, setShowDebugHud] = useState<boolean>(true);
  const [debugTelemetry, setDebugTelemetry] = useState<DebugTelemetry | null>(null);
  const [activeDetections, setActiveDetections] = useState<Detection[]>([]);
  const [ballState, setBallState] = useState<any>(null);
  const [posesCount, setPosesCount] = useState<number>(0);
  const [fps, setFps] = useState<number>(0);
  const [visionLatency, setVisionLatency] = useState<number>(0);

  // Format seconds to mm:ss
  const formatTime = (secs: number) => {
    if (isNaN(secs) || secs < 0) return '00:00';
    const m = Math.floor(secs / 60);
    const s = Math.floor(secs % 60);
    return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
  };

  // Fetch available sources from backend REST API
  useEffect(() => {
    const fetchSources = async () => {
      try {
        const res = await fetch('http://localhost:8000/api/sources');
        if (res.ok) {
          const data = await res.json();
          setSourcesList(data.sources || []);
        }
      } catch (err) {
        console.warn('Sources fetch error:', err);
      }
    };
    fetchSources();
  }, []);

  // Listen to live WebSocket frames and sync clock
  useEffect(() => {
    const unsubFrame = wsService.onFrame(({ imageBase64, videoTimestamp, sourceStatus: ss, detections, ballState: bs, poses, metrics, debug }) => {
      if (videoTimestamp !== undefined) setVideoTime(videoTimestamp);
      if (debug) {
        setDebugTelemetry(debug);
        if (debug.provider_name) {
          if (debug.provider_name.includes('Gemini')) setActiveLLMProvider('Google Gemini');
          else if (debug.provider_name.includes('OpenAI')) setActiveLLMProvider('OpenAI');
          else if (debug.provider_name.includes('Fallback')) setActiveLLMProvider('Local Fallback');
        }
      }
      if (bs) setBallState(bs);
      if (poses) setPosesCount(poses.length);
      if (metrics) {
        setFps(metrics.processing_fps || metrics.input_fps || 0);
        setVisionLatency(metrics.vision_latency_ms || 0);
        if (onMetricsUpdate) onMetricsUpdate(metrics);
      }

      if (!canvasRef.current || !imageBase64) return;
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      const img = new Image();
      img.onload = () => {
        if (canvas.width !== img.width || canvas.height !== img.height) {
          canvas.width = img.width;
          canvas.height = img.height;
        }
        ctx.drawImage(img, 0, 0);

        // Client-side bounding box overlay
        if (showOverlays && detections) {
          detections.forEach((det) => {
            const [x1, y1, x2, y2] = det.bbox;
            const isBall = det.class_name.includes('ball');
            const isPerson = det.class_name === 'person';

            ctx.strokeStyle = isBall ? '#ef4444' : isPerson ? '#3b82f6' : '#10b981';
            ctx.lineWidth = isBall ? 3 : 2;
            ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

            ctx.fillStyle = isBall ? '#ef4444' : isPerson ? '#3b82f6' : '#10b981';
            const label = `${det.class_name.toUpperCase()} ${(det.confidence * 100).toFixed(0)}%`;
            ctx.font = 'bold 11px Inter, sans-serif';
            const textWidth = ctx.measureText(label).width;
            ctx.fillRect(x1, Math.max(0, y1 - 18), textWidth + 8, 18);

            ctx.fillStyle = '#ffffff';
            ctx.fillText(label, x1 + 4, Math.max(13, y1 - 4));
          });
        }

        // Draw Developer Debug velocity vector arrow on canvas
        if (showDebugHud && bs && (bs.vx !== 0 || bs.vy !== 0)) {
          const vx = bs.vx * 4;
          const vy = bs.vy * 4;
          const endX = bs.x + vx;
          const endY = bs.y + vy;

          ctx.beginPath();
          ctx.moveTo(bs.x, bs.y);
          ctx.lineTo(endX, endY);
          ctx.strokeStyle = '#f43f5e';
          ctx.lineWidth = 3;
          ctx.stroke();

          // Arrow head
          const angle = Math.atan2(vy, vx);
          ctx.beginPath();
          ctx.moveTo(endX, endY);
          ctx.lineTo(endX - 8 * Math.cos(angle - Math.PI / 6), endY - 8 * Math.sin(angle - Math.PI / 6));
          ctx.lineTo(endX - 8 * Math.cos(angle + Math.PI / 6), endY - 8 * Math.sin(angle + Math.PI / 6));
          ctx.fillStyle = '#f43f5e';
          ctx.fill();

          // Speed tag next to ball
          ctx.fillStyle = 'rgba(15, 23, 42, 0.9)';
          ctx.fillRect(bs.x + 8, bs.y - 16, 85, 16);
          ctx.fillStyle = '#f43f5e';
          ctx.font = 'bold 10px JetBrains Mono, monospace';
          ctx.fillText(`V:(${bs.vx},${bs.vy})`, bs.x + 12, bs.y - 4);
        }
      };
      img.src = `data:image/jpeg;base64,${imageBase64}`;
    });

    const unsubEvent = wsService.onEvent((ev) => {
      setCurrentEvent(ev);
    });

    return () => {
      unsubFrame();
      unsubEvent();
    };
  }, [showOverlays, showDebugHud, onMetricsUpdate]);

  // Handle File Upload
  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setIsUploading(true);
    setValidationError(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch('http://localhost:8000/api/upload', {
        method: 'POST',
        body: formData
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || 'Upload failed');
      }

      const data = await res.json();
      setUploadedFile({
        name: data.filename,
        path: data.file_path,
        duration: data.metadata?.duration_sec
      });
      setSelectedTarget(data.file_path);
      setSourceType('uploaded');
      setInputMode('upload');
    } catch (err: any) {
      setValidationError(err.message || 'Failed to upload video');
    } finally {
      setIsUploading(false);
    }
  };

  // Start Commentary Session
  const handleStartCommentary = async () => {
    setValidationError(null);
    let target = selectedTarget;
    let type = sourceType;

    if (inputMode === 'url') {
      if (!urlInput.trim()) {
        setValidationError('Please enter a valid cricket video or stream URL.');
        return;
      }
      setIsValidating(true);
      try {
        const valRes = await fetch('http://localhost:8000/api/sources/validate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ url: urlInput.trim(), source_type: sourceType })
        });
        const valData = await valRes.json();
        if (!valData.valid) {
          setValidationError(valData.message || 'This video source cannot be accessed by the application. Try uploading the video file or using a directly accessible stream.');
          setIsValidating(false);
          return;
        }
        target = urlInput.trim();
        type = valData.source_type || 'direct_url';
      } catch (err) {
        setValidationError('Network error while validating video URL. Please check connection.');
        setIsValidating(false);
        return;
      } finally {
        setIsValidating(false);
      }
    } else if (inputMode === 'upload') {
      if (!uploadedFile) {
        setValidationError('Please select or upload a cricket video file first.');
        return;
      }
      target = uploadedFile.path;
      type = 'uploaded';
    }

    wsService.sendCommand('start', {
      source_type: type,
      source_target: target,
      playback_speed: playbackSpeed
    });

    setIsRunning(true);
    setIsPaused(false);
    if (onSessionStateChange) onSessionStateChange(true);
  };

  // Stop Session
  const handleStop = () => {
    wsService.sendCommand('stop');
    setIsRunning(false);
    setIsPaused(false);
    setCurrentEvent(null);
    if (onSessionStateChange) onSessionStateChange(false);
    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext('2d');
      if (ctx) ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    }
  };

  // Pause / Resume
  const handlePauseResume = () => {
    if (isPaused) {
      wsService.sendCommand('resume');
      setIsPaused(false);
    } else {
      wsService.sendCommand('pause');
      setIsPaused(true);
    }
  };

  // Change Playback Speed
  const handleSpeedChange = (speed: number) => {
    setPlaybackSpeed(speed);
    wsService.sendCommand('set_speed' as any, { speed });
  };

  // Status Indicator formatting
  const getStatusBadge = () => {
    const st = sourceStatus?.status || (isRunning ? (isPaused ? 'paused' : 'connected') : 'idle');
    if (st === 'connected') {
      return (
        <span style={{ display: 'flex', alignItems: 'center', gap: '6px', color: '#34d399', fontWeight: 700, fontSize: '0.8rem' }}>
          <span style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: '#10b981', boxShadow: '0 0 8px #10b981' }}></span>
          🟢 LIVE
        </span>
      );
    } else if (st === 'reconnecting') {
      return (
        <span style={{ display: 'flex', alignItems: 'center', gap: '6px', color: '#fbbf24', fontWeight: 700, fontSize: '0.8rem' }}>
          <span style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: '#f59e0b', animation: 'pulse 1s infinite' }}></span>
          🟡 RECONNECTING...
        </span>
      );
    } else if (st === 'unavailable') {
      return (
        <span style={{ display: 'flex', alignItems: 'center', gap: '6px', color: '#f87171', fontWeight: 700, fontSize: '0.8rem' }}>
          <span style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: '#ef4444' }}></span>
          🔴 SOURCE UNAVAILABLE
        </span>
      );
    } else if (st === 'paused') {
      return (
        <span style={{ display: 'flex', alignItems: 'center', gap: '6px', color: '#38bdf8', fontWeight: 700, fontSize: '0.8rem' }}>
          <Pause size={12} color="#38bdf8" />
          PAUSED
        </span>
      );
    }
    return <span style={{ color: '#64748b', fontSize: '0.8rem' }}>STANDBY</span>;
  };

  return (
    <div style={{ background: '#111827', borderRadius: '12px', border: '1px solid #1f2937', overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
      {/* Top Header Bar */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '12px 18px', background: '#1a2234', borderBottom: '1px solid #1f2937' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <Video size={18} color="#60a5fa" />
          <span style={{ fontWeight: 800, fontSize: '0.95rem', letterSpacing: '0.04em', color: '#f3f4f6' }}>CRICKET VIDEO BROADCAST</span>
          {getStatusBadge()}
        </div>

        {/* Telemetry quick stats & Toggles */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          {/* AI Settings Button */}
          <button
            onClick={() => setIsAISettingsOpen(true)}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              background: 'linear-gradient(135deg, rgba(37, 99, 235, 0.25), rgba(124, 58, 237, 0.25))',
              border: '1px solid #6366f1',
              color: '#a5b4fc',
              padding: '4px 12px',
              borderRadius: '6px',
              cursor: 'pointer',
              fontSize: '0.75rem',
              fontWeight: 700
            }}
          >
            <Sparkles size={14} color="#818cf8" />
            AI SETTINGS ({activeLLMProvider})
          </button>

          {/* Debug HUD Toggle Button */}
          <button
            onClick={() => setShowDebugHud(!showDebugHud)}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              background: showDebugHud ? 'rgba(168, 85, 247, 0.2)' : 'transparent',
              border: `1px solid ${showDebugHud ? '#a855f7' : '#374151'}`,
              color: showDebugHud ? '#c084fc' : '#9ca3af',
              padding: '4px 10px',
              borderRadius: '6px',
              cursor: 'pointer',
              fontSize: '0.75rem',
              fontWeight: 600
            }}
          >
            <Terminal size={14} />
            {showDebugHud ? 'DEV HUD ON' : 'DEV HUD OFF'}
          </button>

          {/* Overlays Toggle */}
          <button
            onClick={() => setShowOverlays(!showOverlays)}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              background: showOverlays ? 'rgba(59, 130, 246, 0.2)' : 'transparent',
              border: `1px solid ${showOverlays ? '#3b82f6' : '#374151'}`,
              color: showOverlays ? '#60a5fa' : '#9ca3af',
              padding: '4px 10px',
              borderRadius: '6px',
              cursor: 'pointer',
              fontSize: '0.75rem',
              fontWeight: 600
            }}
          >
            {showOverlays ? <Eye size={14} /> : <EyeOff size={14} />}
            {showOverlays ? 'OVERLAYS ON' : 'OVERLAYS OFF'}
          </button>

          <div style={{ fontSize: '0.75rem', color: '#10b981', fontFamily: 'JetBrains Mono, monospace', background: '#0f172a', padding: '4px 10px', borderRadius: '4px', border: '1px solid #1e293b' }}>
            FPS: <span style={{ fontWeight: 700 }}>{fps.toFixed(1)}</span> | CV: <span style={{ fontWeight: 700 }}>{visionLatency.toFixed(1)}ms</span>
          </div>
        </div>
      </div>

      {/* Main Viewport: Canvas or Standby Ingestion Card */}
      <div style={{ position: 'relative', width: '100%', minHeight: '420px', background: '#030712', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <canvas
          ref={canvasRef}
          style={{ maxWidth: '100%', maxHeight: '460px', objectFit: 'contain', display: isRunning ? 'block' : 'none' }}
        />

        {/* Standby Setup Card (when not running) */}
        {!isRunning && (
          <div style={{ maxWidth: '560px', width: '90%', padding: '24px', background: 'rgba(15, 23, 42, 0.95)', border: '1px solid #334155', borderRadius: '12px', margin: '24px auto', textAlign: 'center', boxShadow: '0 8px 30px rgba(0,0,0,0.6)' }}>
            <h2 style={{ fontSize: '1.25rem', fontWeight: 800, margin: '0 0 6px', color: '#f8fafc', letterSpacing: '0.02em' }}>
              REAL-TIME AI CRICKET COMMENTATOR
            </h2>
            <p style={{ fontSize: '0.85rem', color: '#94a3b8', margin: '0 0 16px' }}>
              Upload an actual match video or paste a live stream URL to begin real-time multimodal commentary.
            </p>

            {/* 1-Click Quick Demonstration Preset Cards */}
            <div style={{ marginBottom: '18px', textAlign: 'left' }}>
              <div style={{ fontSize: '0.72rem', color: '#94a3b8', fontWeight: 700, marginBottom: '8px', letterSpacing: '0.04em' }}>
                QUICK-START PRESET FOOTAGE:
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
                {/* Preset 1: Synthetic Demo */}
                <div
                  onClick={() => {
                    setSelectedTarget('sample_videos/cover_drive_four.mp4');
                    setSourceType('uploaded');
                    setInputMode('sample');
                    setUploadedFile(null);
                  }}
                  style={{
                    background: selectedTarget.includes('cover_drive_four') ? 'rgba(59, 130, 246, 0.15)' : '#0b1120',
                    border: `1.5px solid ${selectedTarget.includes('cover_drive_four') ? '#3b82f6' : '#334155'}`,
                    borderRadius: '8px',
                    padding: '10px 12px',
                    cursor: 'pointer',
                    transition: 'all 0.2s ease'
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '4px' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                      <span style={{ fontSize: '1rem' }}>🏏</span>
                      <strong style={{ fontSize: '0.8rem', color: '#60a5fa' }}>Synthetic Demo</strong>
                    </div>
                    <span style={{ fontSize: '0.62rem', background: '#1e3a8a', color: '#93c5fd', padding: '1px 5px', borderRadius: '4px', fontWeight: 700 }}>4 RUNS</span>
                  </div>
                  <div style={{ fontSize: '0.7rem', color: '#94a3b8', lineHeight: '1.3' }}>
                    Calibrated pitch, vector HUD, ball trajectory, boundary crossing
                  </div>
                </div>

                {/* Preset 2: Real Match Broadcast */}
                <div
                  onClick={() => {
                    setSelectedTarget('sample_videos/real_match_gilchrist.mp4');
                    setSourceType('uploaded');
                    setInputMode('sample');
                    setUploadedFile(null);
                  }}
                  style={{
                    background: selectedTarget.includes('real_match_gilchrist') ? 'rgba(234, 179, 8, 0.15)' : '#0b1120',
                    border: `1.5px solid ${selectedTarget.includes('real_match_gilchrist') ? '#eab308' : '#334155'}`,
                    borderRadius: '8px',
                    padding: '10px 12px',
                    cursor: 'pointer',
                    transition: 'all 0.2s ease'
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '4px' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                      <span style={{ fontSize: '1rem' }}>🏟️</span>
                      <strong style={{ fontSize: '0.8rem', color: '#facc15' }}>Real Broadcast</strong>
                    </div>
                    <span style={{ fontSize: '0.62rem', background: '#854d0e', color: '#fef08a', padding: '1px 5px', borderRadius: '4px', fontWeight: 700 }}>GILCHRIST MCG</span>
                  </div>
                  <div style={{ fontSize: '0.7rem', color: '#94a3b8', lineHeight: '1.3' }}>
                    Real match footage: Australia vs India, jersey classification &amp; full poses
                  </div>
                </div>
              </div>
            </div>

            {/* Ingestion Mode Toggle Tabs */}
            <div style={{ display: 'flex', gap: '8px', marginBottom: '16px', justifyContent: 'center' }}>
              <button
                onClick={() => { setInputMode('upload'); setSourceType('uploaded'); }}
                style={{
                  padding: '6px 14px',
                  borderRadius: '6px',
                  fontSize: '0.8rem',
                  fontWeight: 600,
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px',
                  background: inputMode === 'upload' ? '#2563eb' : '#1e293b',
                  color: inputMode === 'upload' ? '#fff' : '#94a3b8',
                  border: `1px solid ${inputMode === 'upload' ? '#3b82f6' : '#334155'}`
                }}
              >
                <Upload size={14} /> Upload Video (.mp4, .mov, .mkv)
              </button>

              <button
                onClick={() => { setInputMode('url'); setSourceType('auto'); }}
                style={{
                  padding: '6px 14px',
                  borderRadius: '6px',
                  fontSize: '0.8rem',
                  fontWeight: 600,
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px',
                  background: inputMode === 'url' ? '#2563eb' : '#1e293b',
                  color: inputMode === 'url' ? '#fff' : '#94a3b8',
                  border: `1px solid ${inputMode === 'url' ? '#3b82f6' : '#334155'}`
                }}
              >
                <LinkIcon size={14} /> Video / Stream URL
              </button>

              <button
                onClick={() => { setInputMode('sample'); setSourceType('uploaded'); }}
                style={{
                  padding: '6px 14px',
                  borderRadius: '6px',
                  fontSize: '0.8rem',
                  fontWeight: 600,
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px',
                  background: inputMode === 'sample' ? '#2563eb' : '#1e293b',
                  color: inputMode === 'sample' ? '#fff' : '#94a3b8',
                  border: `1px solid ${inputMode === 'sample' ? '#3b82f6' : '#334155'}`
                }}
              >
                <Film size={14} /> Samples &amp; Webcam
              </button>
            </div>

            {/* Mode 1: Upload Video */}
            {inputMode === 'upload' && (
              <div style={{ background: '#0b1120', padding: '16px', borderRadius: '8px', border: '1px dashed #38bdf8', marginBottom: '16px' }}>
                <input
                  type="file"
                  ref={fileInputRef}
                  onChange={handleFileUpload}
                  accept=".mp4,.mov,.mkv,.avi,.webm"
                  style={{ display: 'none' }}
                />
                <button
                  onClick={() => fileInputRef.current?.click()}
                  disabled={isUploading}
                  style={{
                    background: 'linear-gradient(135deg, #1e293b, #0f172a)',
                    border: '1px solid #475569',
                    color: '#f8fafc',
                    padding: '10px 20px',
                    borderRadius: '8px',
                    cursor: isUploading ? 'not-allowed' : 'pointer',
                    fontWeight: 700,
                    fontSize: '0.85rem',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px',
                    margin: '0 auto 8px'
                  }}
                >
                  <Upload size={16} color="#38bdf8" />
                  {isUploading ? 'Uploading & Verifying...' : 'Choose Cricket Video File'}
                </button>
                {uploadedFile ? (
                  <div style={{ color: '#34d399', fontSize: '0.8rem', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px' }}>
                    <CheckCircle2 size={15} />
                    <span>Loaded: <strong>{uploadedFile.name}</strong> {uploadedFile.duration ? `(${formatTime(uploadedFile.duration)})` : ''}</span>
                  </div>
                ) : (
                  <div style={{ color: '#64748b', fontSize: '0.75rem' }}>Supports MP4, MOV, MKV, AVI recorded cricket matches.</div>
                )}
              </div>
            )}

            {/* Mode 2: Online Video URL */}
            {inputMode === 'url' && (
              <div style={{ background: '#0b1120', padding: '16px', borderRadius: '8px', border: '1px solid #334155', marginBottom: '16px', textAlign: 'left' }}>
                <label style={{ display: 'block', fontSize: '0.75rem', color: '#94a3b8', fontWeight: 600, marginBottom: '6px' }}>
                  PASTE CRICKET VIDEO / LIVE STREAM URL:
                </label>
                <div style={{ display: 'flex', gap: '8px' }}>
                  <input
                    type="text"
                    value={urlInput}
                    onChange={(e) => setUrlInput(e.target.value)}
                    placeholder="https://.../match.mp4 or rtsp://.../stream"
                    style={{
                      flex: 1,
                      background: '#1e293b',
                      border: '1px solid #475569',
                      borderRadius: '6px',
                      color: '#f8fafc',
                      padding: '8px 12px',
                      fontSize: '0.85rem'
                    }}
                  />
                  <select
                    value={sourceType}
                    onChange={(e) => setSourceType(e.target.value)}
                    style={{
                      background: '#1e293b',
                      border: '1px solid #475569',
                      borderRadius: '6px',
                      color: '#f8fafc',
                      padding: '8px',
                      fontSize: '0.8rem',
                      fontWeight: 600
                    }}
                  >
                    <option value="auto">Auto Detect</option>
                    <option value="direct_url">Direct Video URL</option>
                    <option value="livestream">Live Stream (RTSP/HLS)</option>
                  </select>
                </div>
                <div style={{ fontSize: '0.7rem', color: '#64748b', marginTop: '6px' }}>
                  Only directly accessible streams are supported. DRM, paywalls, and private streams are respected.
                </div>
              </div>
            )}

            {/* Mode 3: Local Sample Videos or Webcam */}
            {inputMode === 'sample' && (
              <div style={{ background: '#0b1120', padding: '16px', borderRadius: '8px', border: '1px solid #334155', marginBottom: '16px', textAlign: 'left' }}>
                <label style={{ display: 'block', fontSize: '0.75rem', color: '#94a3b8', fontWeight: 600, marginBottom: '6px' }}>
                  SELECT SYSTEM SOURCE:
                </label>
                <select
                  value={`${sourceType}:${selectedTarget}`}
                  onChange={(e) => {
                    const [st, ...tgt] = e.target.value.split(':');
                    setSourceType(st);
                    setSelectedTarget(tgt.join(':'));
                  }}
                  style={{
                    width: '100%',
                    background: '#1e293b',
                    border: '1px solid #475569',
                    borderRadius: '6px',
                    color: '#f8fafc',
                    padding: '8px 12px',
                    fontSize: '0.85rem',
                    fontWeight: 500
                  }}
                >
                  {sourcesList.length > 0 ? (
                    sourcesList.map((s, idx) => (
                      <option key={idx} value={`${s.type}:${s.id}`}>
                        {s.type.toUpperCase()}: {s.name}
                      </option>
                    ))
                  ) : (
                    <>
                      <option value="uploaded:sample_videos/cover_drive_four.mp4">SAMPLE: Cover Drive 4 (Synthetic Test)</option>
                      <option value="webcam:0">WEBCAM: Device 0</option>
                    </>
                  )}
                </select>
              </div>
            )}

            {/* Error diagnostic banner if validation fails */}
            {validationError && (
              <div style={{ background: 'rgba(239, 68, 68, 0.15)', border: '1px solid #ef4444', borderRadius: '6px', padding: '10px 14px', marginBottom: '16px', textAlign: 'left', display: 'flex', gap: '8px', alignItems: 'flex-start' }}>
                <AlertCircle size={18} color="#ef4444" style={{ flexShrink: 0, marginTop: '2px' }} />
                <span style={{ fontSize: '0.8rem', color: '#fca5a5', lineHeight: '1.4' }}>{validationError}</span>
              </div>
            )}

            {/* Primary Action Button */}
            <button
              onClick={handleStartCommentary}
              disabled={isValidating || isUploading}
              style={{
                width: '100%',
                background: 'linear-gradient(135deg, #2563eb, #1d4ed8)',
                color: '#ffffff',
                border: 'none',
                padding: '12px 24px',
                borderRadius: '8px',
                fontWeight: 800,
                fontSize: '1rem',
                letterSpacing: '0.04em',
                cursor: isValidating || isUploading ? 'not-allowed' : 'pointer',
                boxShadow: '0 4px 15px rgba(37, 99, 235, 0.4)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '8px'
              }}
            >
              {isValidating ? (
                <>
                  <RefreshCw size={18} style={{ animation: 'spin 1s linear infinite' }} />
                  VALIDATING STREAM...
                </>
              ) : (
                <>
                  <Play size={18} fill="#ffffff" />
                  START COMMENTARY
                </>
              )}
            </button>

            {/* Multimodal AI Capabilities Info Bar */}
            <div style={{
              marginTop: '16px',
              padding: '10px 14px',
              background: 'rgba(15, 23, 42, 0.6)',
              borderRadius: '8px',
              fontSize: '0.72rem',
              color: '#94a3b8',
              display: 'flex',
              justifyContent: 'space-around',
              alignItems: 'center',
              border: '1px solid #1e293b',
              flexWrap: 'wrap',
              gap: '8px'
            }}>
              <span>🤖 <strong>Vision:</strong> YOLOv8 + Pose</span>
              <span>🎯 <strong>Tracking:</strong> Kalman 2D</span>
              <span>📐 <strong>Field:</strong> Dynamic Calibration</span>
              <span>🧠 <strong>AI:</strong> {activeLLMProvider}</span>
              <span>🔊 <strong>Voice:</strong> Edge-TTS</span>
            </div>
          </div>
        )}

        {/* Live Detected Badges during playback */}
        {isRunning && (
          <div style={{ position: 'absolute', bottom: '12px', left: '12px', display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
            <div style={{ background: 'rgba(15, 23, 42, 0.85)', backdropFilter: 'blur(4px)', padding: '4px 10px', borderRadius: '6px', border: '1px solid rgba(255, 255, 255, 0.1)', fontSize: '0.75rem', color: '#e2e8f0', display: 'flex', gap: '6px', alignItems: 'center' }}>
              <Layers size={14} color="#38bdf8" />
              <span>Objects: <strong>{activeDetections.length}</strong></span>
            </div>

            {ballState && (
              <div style={{ background: 'rgba(239, 68, 68, 0.2)', backdropFilter: 'blur(4px)', padding: '4px 10px', borderRadius: '6px', border: '1px solid #ef4444', fontSize: '0.75rem', color: '#fca5a5', display: 'flex', gap: '6px', alignItems: 'center' }}>
                <span style={{ width: '7px', height: '7px', borderRadius: '50%', backgroundColor: '#ef4444', boxShadow: '0 0 6px #ef4444' }}></span>
                <span>BALL: <strong>({ballState.x}, {ballState.y})</strong></span>
                {ballState.speed_kmh && ballState.speed_estimation_available && (
                  <span style={{ color: '#fff', fontWeight: 700 }}>• {ballState.speed_kmh} km/h</span>
                )}
              </div>
            )}

            {posesCount > 0 && (
              <div style={{ background: 'rgba(168, 85, 247, 0.2)', backdropFilter: 'blur(4px)', padding: '4px 10px', borderRadius: '6px', border: '1px solid #a855f7', fontSize: '0.75rem', color: '#e9d5ff', display: 'flex', gap: '6px', alignItems: 'center' }}>
                <span>POSES: <strong>{posesCount} Skeletons</strong></span>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Developer Event Debug HUD Overlay */}
      {isRunning && (
        <EventDebugPanel debug={debugTelemetry} isOpen={showDebugHud} />
      )}

      {/* Synchronized Processing HUD & Playback Control Bar */}
      {isRunning && (
        <div style={{ background: '#131b2e', borderTop: '1px solid #1f2937', padding: '12px 18px' }}>
          {/* Synchronized AI Clock and Event Spotlight */}
          <div style={{ display: 'flex', flexWrap: 'wrap', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px', gap: '10px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.85rem', color: '#94a3b8' }}>
                <Clock size={15} color="#60a5fa" />
                <span>Video Time:</span>
                <strong style={{ color: '#f8fafc', fontFamily: 'JetBrains Mono, monospace', fontSize: '0.95rem' }}>
                  {formatTime(videoTime)}
                  {sourceStatus?.duration_sec ? ` / ${formatTime(sourceStatus.duration_sec)}` : ''}
                </strong>
              </div>

              <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.85rem' }}>
                <Activity size={15} color="#34d399" />
                <span style={{ color: '#94a3b8' }}>AI Processing:</span>
                <strong style={{ color: isPaused ? '#38bdf8' : '#34d399' }}>
                  {isPaused ? 'PAUSED' : 'LIVE'}
                </strong>
              </div>

              {currentEvent && (
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px', background: '#312e81', border: '1px solid #4f46e5', padding: '2px 8px', borderRadius: '4px', fontSize: '0.75rem', color: '#c7d2fe' }}>
                  <Sparkles size={12} color="#a5b4fc" />
                  <span>Event: <strong>{currentEvent.event_type}</strong></span>
                </div>
              )}
            </div>

            {/* Playback speed selector: 0.5x, 1x, 1.5x, 2x */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <span style={{ fontSize: '0.75rem', color: '#94a3b8', fontWeight: 600 }}>SPEED:</span>
              {[0.5, 1.0, 1.5, 2.0].map((s) => (
                <button
                  key={s}
                  onClick={() => handleSpeedChange(s)}
                  style={{
                    background: playbackSpeed === s ? '#2563eb' : '#1e293b',
                    color: playbackSpeed === s ? '#ffffff' : '#94a3b8',
                    border: `1px solid ${playbackSpeed === s ? '#3b82f6' : '#334155'}`,
                    borderRadius: '4px',
                    padding: '3px 8px',
                    fontSize: '0.75rem',
                    fontWeight: 700,
                    cursor: 'pointer'
                  }}
                >
                  {s}x
                </button>
              ))}
            </div>
          </div>

          {/* Player controls */}
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', paddingTop: '8px', borderTop: '1px solid #1e293b' }}>
            <div style={{ fontSize: '0.8rem', color: '#64748b' }}>
              Source: <strong style={{ color: '#94a3b8' }}>{sourceStatus?.source_type.toUpperCase() || 'VIDEO'}</strong>
            </div>

            <div style={{ display: 'flex', gap: '10px' }}>
              <button
                onClick={handlePauseResume}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px',
                  background: '#374151',
                  color: '#ffffff',
                  border: 'none',
                  padding: '6px 14px',
                  borderRadius: '6px',
                  fontWeight: 600,
                  fontSize: '0.8rem',
                  cursor: 'pointer'
                }}
              >
                {isPaused ? <Play size={14} /> : <Pause size={14} />}
                {isPaused ? 'RESUME' : 'PAUSE'}
              </button>

              <button
                onClick={handleStop}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px',
                  background: '#dc2626',
                  color: '#ffffff',
                  border: 'none',
                  padding: '6px 14px',
                  borderRadius: '6px',
                  fontWeight: 700,
                  fontSize: '0.8rem',
                  cursor: 'pointer'
                }}
              >
                <Square size={13} fill="#ffffff" />
                STOP
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Dynamic AI Settings Modal */}
      <AISettingsModal
        isOpen={isAISettingsOpen}
        onClose={() => setIsAISettingsOpen(false)}
        onConfigSaved={(p) => setActiveLLMProvider(p === 'google-gemini' ? 'Google Gemini' : p === 'openai' ? 'OpenAI' : 'Local Fallback')}
      />
    </div>
  );
};
