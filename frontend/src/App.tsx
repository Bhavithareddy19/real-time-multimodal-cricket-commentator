import React, { useState, useEffect } from 'react';
import { VideoPlayer } from './components/VideoPlayer';
import { CommentaryPanel } from './components/CommentaryPanel';
import { EventTimeline } from './components/EventTimeline';
import { MetricsPanel } from './components/MetricsPanel';
import { ConnectionStatus } from './components/ConnectionStatus';
import { wsService, PerformanceMetrics, ConnectionState, SourceStatus } from './services/websocket';
import { Radio } from 'lucide-react';

export const App: React.FC = () => {
  const [metrics, setMetrics] = useState<PerformanceMetrics>({
    input_fps: 0,
    processing_fps: 0,
    frame_latency_ms: 0,
    vision_latency_ms: 0,
    event_latency_ms: 0,
    llm_latency_ms: 0,
    tts_latency_ms: 0,
    e2e_latency_ms: 0,
    dropped_frames: 0,
    queue_size: 0,
    cache_hits: 0,
    cache_misses: 0
  });

  const [connState, setConnState] = useState<ConnectionState>('disconnected');
  const [isRunning, setIsRunning] = useState<boolean>(false);
  const [stylePreset, setStylePreset] = useState<string>('PROFESSIONAL');
  const [sourceStatus, setSourceStatus] = useState<SourceStatus | null>(null);
  const [totalEvents, setTotalEvents] = useState<number>(0);
  const [totalCommentaries, setTotalCommentaries] = useState<number>(0);

  useEffect(() => {
    wsService.connect();
    const unsubStatus = wsService.onStatus((s) => setConnState(s.state));
    const unsubFrame = wsService.onFrame(({ sourceStatus: ss }) => {
      if (ss) setSourceStatus(ss);
    });
    const unsubEvent = wsService.onEvent(() => {
      setTotalEvents((prev) => prev + 1);
    });
    const unsubCommentary = wsService.onCommentary(() => {
      setTotalCommentaries((prev) => prev + 1);
    });

    return () => {
      unsubStatus();
      unsubFrame();
      unsubEvent();
      unsubCommentary();
      wsService.disconnect();
    };
  }, []);

  return (
    <div style={{ minHeight: '100vh', backgroundColor: '#090d16', color: '#f3f4f6', display: 'flex', flexDirection: 'column' }}>
      {/* Top Broadcast Header */}
      <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '14px 28px', backgroundColor: '#0f172a', borderBottom: '1px solid #1e293b', boxShadow: '0 4px 20px rgba(0,0,0,0.4)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div style={{ width: '36px', height: '36px', borderRadius: '8px', background: 'linear-gradient(135deg, #2563eb, #7c3aed)', display: 'flex', alignItems: 'center', justifyContent: 'center', boxShadow: '0 0 15px rgba(37,99,235,0.5)' }}>
            <Radio size={20} color="#ffffff" />
          </div>
          <div>
            <div style={{ fontSize: '1.15rem', fontWeight: 800, letterSpacing: '0.04em', background: 'linear-gradient(90deg, #60a5fa, #c084fc)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
              CRICKET AI BROADCAST
            </div>
            <div style={{ fontSize: '0.75rem', color: '#94a3b8', fontWeight: 500 }}>
              Real-Time Multimodal Computer Vision, State Machine &amp; Streaming Commentary
            </div>
          </div>
        </div>

        {/* Live status indicators */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          <ConnectionStatus
            state={connState}
            sourceType={sourceStatus?.source_type ? sourceStatus.source_type.toUpperCase() : 'VIDEO'}
            isRunning={isRunning}
          />
        </div>
      </header>

      {/* Main Grid Workspace */}
      <main style={{ flex: 1, padding: '20px 28px', display: 'flex', flexDirection: 'column', gap: '20px', maxWidth: '1600px', margin: '0 auto', width: '100%', boxSizing: 'border-box' }}>
        {/* Upper row: Video Player + Event Timeline */}
        <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1.4fr) minmax(0, 1fr)', gap: '20px' }}>
          {/* Video Player */}
          <VideoPlayer
            onMetricsUpdate={(m) => setMetrics(m)}
            onSessionStateChange={(running) => setIsRunning(running)}
          />

          {/* Match Scorecard & Event Timeline */}
          <EventTimeline />
        </div>

        {/* Middle row: Live Commentary */}
        <CommentaryPanel
          stylePreset={stylePreset}
          onStyleChange={(s) => setStylePreset(s)}
        />

        {/* Lower row: System Telemetry Metrics */}
        <MetricsPanel
          metrics={metrics}
          sourceStatus={sourceStatus}
          totalEvents={totalEvents}
          totalCommentaries={totalCommentaries}
        />
      </main>

      {/* Broadcast Footer */}
      <footer style={{ padding: '12px 28px', backgroundColor: '#0b0f19', borderTop: '1px solid #1e293b', display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '0.75rem', color: '#64748b' }}>
        <div style={{ display: 'flex', gap: '14px' }}>
          <span>Architecture: VideoSource Ingestion Pipeline</span>
          <span>•</span>
          <span>CV: YOLOv8n CPU</span>
          <span>•</span>
          <span>TTS: Streaming Edge-TTS</span>
          <span>•</span>
          <span>Clock Sync: Enabled</span>
        </div>
        <div>
          Production Portfolio AI System © 2026
        </div>
      </footer>
    </div>
  );
};
