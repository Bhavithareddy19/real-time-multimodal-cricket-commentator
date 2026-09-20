import React from 'react';
import { Activity, Cpu, Zap, Clock, AlertTriangle, Video, Mic, Volume2 } from 'lucide-react';
import { PerformanceMetrics, SourceStatus } from '../services/websocket';

interface Props {
  metrics: PerformanceMetrics;
  sourceStatus?: SourceStatus | null;
  totalEvents?: number;
  totalCommentaries?: number;
}

export const MetricsPanel: React.FC<Props> = ({
  metrics,
  sourceStatus,
  totalEvents = 0,
  totalCommentaries = 0
}) => {
  const formatMs = (val: number | undefined) => {
    if (val === undefined || val === 0) return '-- ms';
    return `${val.toFixed(1)} ms`;
  };

  const formatTime = (secs: number | undefined | null) => {
    if (!secs || isNaN(secs) || secs < 0) return 'Live / Standby';
    const m = Math.floor(secs / 60);
    const s = Math.floor(secs % 60);
    return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
  };

  const getStatusText = () => {
    const s = sourceStatus?.status;
    if (s === 'connected') return '🟢 Connected';
    if (s === 'reconnecting') return '🟡 Reconnecting...';
    if (s === 'unavailable') return '🔴 Unavailable';
    if (s === 'paused') return '⏸️ Paused';
    return '⚪ Idle';
  };

  return (
    <div style={{ background: '#111827', borderRadius: '12px', border: '1px solid #1f2937', padding: '16px', display: 'flex', flexDirection: 'column', gap: '14px' }}>
      {/* Top Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid #1f2937', paddingBottom: '10px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Activity size={18} color="#10b981" />
          <span style={{ fontWeight: 700, fontSize: '0.9rem', letterSpacing: '0.05em', color: '#f3f4f6' }}>SYSTEM TELEMETRY & LATENCY</span>
        </div>
        <div style={{ display: 'flex', gap: '10px', fontSize: '0.75rem' }}>
          <span style={{ color: '#9ca3af' }}>Target E2E: <strong style={{ color: '#60a5fa' }}>&lt;500 ms</strong></span>
          <span style={{ color: '#9ca3af' }}>|</span>
          <span style={{ color: '#9ca3af' }}>Measured E2E: <strong style={{ color: metrics.e2e_latency_ms > 0 ? '#34d399' : '#f59e0b' }}>{formatMs(metrics.e2e_latency_ms)}</strong></span>
        </div>
      </div>

      {/* Side-by-side: Source Status & AI Pipeline Dashboard Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '12px' }}>
        {/* SOURCE STATUS CARD */}
        <div style={{ background: '#1a2234', padding: '12px 16px', borderRadius: '8px', border: '1px solid #1f2937' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.75rem', color: '#60a5fa', fontWeight: 800, letterSpacing: '0.05em', marginBottom: '8px', borderBottom: '1px solid #232f48', paddingBottom: '4px' }}>
            <Video size={14} /> SOURCE STATUS
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px', fontSize: '0.8rem' }}>
            <div>
              <div style={{ color: '#9ca3af', fontSize: '0.7rem' }}>Type</div>
              <strong style={{ color: '#f8fafc', textTransform: 'capitalize' }}>
                {sourceStatus?.source_type ? sourceStatus.source_type.replace('_', ' ') : 'Video/Webcam'}
              </strong>
            </div>
            <div>
              <div style={{ color: '#9ca3af', fontSize: '0.7rem' }}>Status</div>
              <strong style={{ color: '#f8fafc' }}>{getStatusText()}</strong>
            </div>
            <div>
              <div style={{ color: '#9ca3af', fontSize: '0.7rem' }}>Duration</div>
              <strong style={{ color: '#f8fafc' }}>{formatTime(sourceStatus?.duration_sec)}</strong>
            </div>
            <div>
              <div style={{ color: '#9ca3af', fontSize: '0.7rem' }}>Playback Speed</div>
              <strong style={{ color: '#38bdf8' }}>{sourceStatus?.playback_speed ? `${sourceStatus.playback_speed}x` : '1.0x'}</strong>
            </div>
          </div>
        </div>

        {/* AI PIPELINE CARD */}
        <div style={{ background: '#1a2234', padding: '12px 16px', borderRadius: '8px', border: '1px solid #1f2937' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.75rem', color: '#34d399', fontWeight: 800, letterSpacing: '0.05em', marginBottom: '8px', borderBottom: '1px solid #232f48', paddingBottom: '4px' }}>
            <Activity size={14} /> AI PIPELINE SUMMARY
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px', fontSize: '0.8rem' }}>
            <div>
              <div style={{ color: '#9ca3af', fontSize: '0.7rem' }}>Processing FPS</div>
              <strong style={{ color: '#34d399', fontFamily: 'JetBrains Mono, monospace' }}>
                {(metrics.processing_fps || metrics.input_fps || 0).toFixed(1)}
              </strong>
            </div>
            <div>
              <div style={{ color: '#9ca3af', fontSize: '0.7rem' }}>Events Detected</div>
              <strong style={{ color: '#f8fafc', fontFamily: 'JetBrains Mono, monospace' }}>{totalEvents}</strong>
            </div>
            <div>
              <div style={{ color: '#9ca3af', fontSize: '0.7rem' }}>Commentary Lines</div>
              <strong style={{ color: '#f8fafc', fontFamily: 'JetBrains Mono, monospace' }}>{totalCommentaries}</strong>
            </div>
            <div>
              <div style={{ color: '#9ca3af', fontSize: '0.7rem' }}>Streaming Audio</div>
              <strong style={{ color: '#10b981', display: 'flex', alignItems: 'center', gap: '4px' }}>
                <Volume2 size={13} /> Active
              </strong>
            </div>
          </div>
        </div>
      </div>

      {/* Latency Breakdown Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: '10px' }}>
        {/* Vision Inference */}
        <div style={{ background: '#182030', padding: '10px 12px', borderRadius: '8px', border: '1px solid #1f2937' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.75rem', color: '#9ca3af', marginBottom: '4px' }}>
            <Cpu size={14} color="#3b82f6" />
            <span>VISION (YOLO)</span>
          </div>
          <div style={{ fontSize: '1.2rem', fontWeight: 800, color: '#60a5fa', fontFamily: 'JetBrains Mono, monospace' }}>
            {formatMs(metrics.vision_latency_ms)}
          </div>
          <div style={{ fontSize: '0.7rem', color: '#6b7280', marginTop: '2px' }}>YOLOv8n CPU</div>
        </div>

        {/* Event State Machine */}
        <div style={{ background: '#182030', padding: '10px 12px', borderRadius: '8px', border: '1px solid #1f2937' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.75rem', color: '#9ca3af', marginBottom: '4px' }}>
            <Clock size={14} color="#a855f7" />
            <span>EVENT LOGIC</span>
          </div>
          <div style={{ fontSize: '1.2rem', fontWeight: 800, color: '#c084fc', fontFamily: 'JetBrains Mono, monospace' }}>
            {formatMs(metrics.event_latency_ms)}
          </div>
          <div style={{ fontSize: '0.7rem', color: '#6b7280', marginTop: '2px' }}>State Machine</div>
        </div>

        {/* LLM First Token */}
        <div style={{ background: '#182030', padding: '10px 12px', borderRadius: '8px', border: '1px solid #1f2937' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.75rem', color: '#9ca3af', marginBottom: '4px' }}>
            <Zap size={14} color="#ec4899" />
            <span>LLM 1ST TOKEN</span>
          </div>
          <div style={{ fontSize: '1.2rem', fontWeight: 800, color: '#f472b6', fontFamily: 'JetBrains Mono, monospace' }}>
            {formatMs(metrics.llm_latency_ms)}
          </div>
          <div style={{ fontSize: '0.7rem', color: '#6b7280', marginTop: '2px' }}>Event-Triggered</div>
        </div>

        {/* TTS First Audio */}
        <div style={{ background: '#182030', padding: '10px 12px', borderRadius: '8px', border: '1px solid #1f2937' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.75rem', color: '#9ca3af', marginBottom: '4px' }}>
            <Activity size={14} color="#10b981" />
            <span>TTS 1ST AUDIO</span>
          </div>
          <div style={{ fontSize: '1.2rem', fontWeight: 800, color: '#34d399', fontFamily: 'JetBrains Mono, monospace' }}>
            {formatMs(metrics.tts_latency_ms)}
          </div>
          <div style={{ fontSize: '0.7rem', color: '#6b7280', marginTop: '2px' }}>Streaming MP3</div>
        </div>

        {/* Dropped Frames & Queue */}
        <div style={{ background: '#182030', padding: '10px 12px', borderRadius: '8px', border: '1px solid #1f2937' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.75rem', color: '#9ca3af', marginBottom: '4px' }}>
            <AlertTriangle size={14} color="#eab308" />
            <span>DROPPED / QUEUE</span>
          </div>
          <div style={{ fontSize: '1.2rem', fontWeight: 800, color: metrics.dropped_frames > 0 ? '#facc15' : '#9ca3af', fontFamily: 'JetBrains Mono, monospace' }}>
            {metrics.dropped_frames || 0} <span style={{ fontSize: '0.75rem', fontWeight: 500, color: '#6b7280' }}>/ Q:{metrics.queue_size || 0}</span>
          </div>
          <div style={{ fontSize: '0.7rem', color: '#6b7280', marginTop: '2px' }}>Backpressure Drop</div>
        </div>
      </div>
    </div>
  );
};
