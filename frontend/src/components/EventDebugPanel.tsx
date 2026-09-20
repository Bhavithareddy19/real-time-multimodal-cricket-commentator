import React from 'react';
import { Terminal, Activity, Compass, Cpu, Layers, GitCommit, ShieldAlert, CheckCircle2 } from 'lucide-react';
import { DebugTelemetry } from '../services/websocket';

interface Props {
  debug?: DebugTelemetry | null;
  isOpen: boolean;
  onClose?: () => void;
}

export const EventDebugPanel: React.FC<Props> = ({ debug, isOpen }) => {
  if (!isOpen || !debug) return null;

  const getStateColor = (state: string) => {
    switch (state) {
      case 'BOUNDARY_DETECTED':
      case 'FOUR':
      case 'SIX':
        return '#10b981';
      case 'WICKET_DETECTED':
      case 'WICKET':
        return '#ef4444';
      case 'BAT_CONTACT':
      case 'SHOT_PLAYED':
        return '#f59e0b';
      case 'BALL_IN_PLAY':
      case 'BALL_TRAVEL':
        return '#3b82f6';
      case 'DELIVERY':
        return '#8b5cf6';
      default:
        return '#64748b';
    }
  };

  return (
    <div style={{
      background: 'rgba(10, 15, 29, 0.95)',
      border: '1px solid #1e3a8a',
      borderRadius: '8px',
      padding: '14px',
      margin: '8px 12px 12px',
      boxShadow: '0 8px 24px rgba(0, 0, 0, 0.6)',
      fontSize: '0.8rem',
      color: '#e2e8f0',
      fontFamily: 'JetBrains Mono, monospace'
    }}>
      {/* HUD Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px', borderBottom: '1px solid #1e293b', paddingBottom: '6px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Terminal size={15} color="#38bdf8" />
          <span style={{ fontWeight: 800, color: '#38bdf8', letterSpacing: '0.05em' }}>DEVELOPER EVENT DEBUG HUD</span>
          <span style={{ fontSize: '0.65rem', background: '#1e3a8a', color: '#93c5fd', padding: '1px 6px', borderRadius: '4px' }}>LIVE TELEMETRY</span>
        </div>
        <div style={{ fontSize: '0.7rem', color: '#94a3b8' }}>
          Delivery: <strong style={{ color: '#fff' }}>{debug.delivery_id || 'CALL'}</strong>
        </div>
      </div>

      {/* Grid of Debug Diagnostics */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '12px' }}>
        {/* 1. State Machine Transitions */}
        <div style={{ background: '#0b1329', padding: '10px', borderRadius: '6px', border: '1px solid #1e293b' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', color: '#94a3b8', fontSize: '0.7rem', marginBottom: '6px' }}>
            <GitCommit size={13} color="#c084fc" />
            <span>CRICKET EVENT STATE MACHINE</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
            <span style={{ color: '#64748b', fontSize: '0.75rem' }}>{debug.previous_state}</span>
            <span style={{ color: '#94a3b8' }}>&rarr;</span>
            <span style={{
              background: getStateColor(debug.current_state),
              color: '#ffffff',
              padding: '2px 8px',
              borderRadius: '4px',
              fontWeight: 800,
              fontSize: '0.75rem'
            }}>
              {debug.current_state}
            </span>
          </div>
          <div style={{ fontSize: '0.68rem', color: '#64748b' }}>
            Debounce lock: Active per delivery
          </div>
        </div>

        {/* 2. Ball Kinematics & Kalman Filter */}
        <div style={{ background: '#0b1329', padding: '10px', borderRadius: '6px', border: '1px solid #1e293b' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', color: '#94a3b8', fontSize: '0.7rem', marginBottom: '6px' }}>
            <Compass size={13} color="#f43f5e" />
            <span>BALL KINEMATICS (2D KALMAN)</span>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
            <span>Pos: <strong style={{ color: '#fff' }}>({debug.ball_x ?? '--'}, {debug.ball_y ?? '--'})</strong></span>
            <span>Vel: <strong style={{ color: '#38bdf8' }}>({debug.ball_vx}, {debug.ball_vy})</strong></span>
          </div>
          <div style={{ fontSize: '0.72rem', color: debug.speed_calibrated ? '#34d399' : '#fbbf24' }}>
            {debug.speed_calibrated && debug.ball_speed_kmh
              ? `Speed: ${debug.ball_speed_kmh} km/h (Calibrated)`
              : 'Speed: Pitch uncalibrated (no fake speed)'}
          </div>
        </div>

        {/* 3. Shot Classification & Pose */}
        <div style={{ background: '#0b1329', padding: '10px', borderRadius: '6px', border: '1px solid #1e293b' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', color: '#94a3b8', fontSize: '0.7rem', marginBottom: '6px' }}>
            <Activity size={13} color="#fbbf24" />
            <span>SHOT CLASSIFICATION &amp; POSE</span>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
            <span style={{ color: '#f59e0b', fontWeight: 700 }}>
              {debug.shot_type || 'Awaiting Contact'}
            </span>
            {debug.shot_confidence && (
              <span style={{ color: '#94a3b8' }}>
                {(debug.shot_confidence * 100).toFixed(0)}% conf
              </span>
            )}
          </div>
          <div style={{ fontSize: '0.68rem', color: '#94a3b8', lineHeight: '1.3' }}>
            {debug.shot_reasoning || (debug.batsman_detected ? 'Batsman in batting stance' : 'Scanning player keypoints')}
          </div>
        </div>

        {/* 4. Queue & Frame Drops (Backpressure) */}
        <div style={{ background: '#0b1329', padding: '10px', borderRadius: '6px', border: '1px solid #1e293b' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', color: '#94a3b8', fontSize: '0.7rem', marginBottom: '6px' }}>
            <ShieldAlert size={13} color="#eab308" />
            <span>BACKPRESSURE &amp; PIPELINE</span>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
            <span>Queue: <strong style={{ color: debug.queue_depth > 1 ? '#f59e0b' : '#34d399' }}>{debug.queue_depth} / 2</strong></span>
            <span>Drops: <strong style={{ color: debug.dropped_frames > 0 ? '#f59e0b' : '#94a3b8' }}>{debug.dropped_frames}</strong></span>
          </div>
          <div style={{ fontSize: '0.68rem', color: '#64748b' }}>
            Skeletons: {debug.poses_count} | Objects: {debug.detections_count}
          </div>
        </div>

        {/* 5. Field & Camera Auto-Calibration */}
        <div style={{ background: '#0b1329', padding: '10px', borderRadius: '6px', border: '1px solid #1e293b' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', color: '#94a3b8', fontSize: '0.7rem', marginBottom: '6px' }}>
            <Layers size={13} color="#38bdf8" />
            <span>FIELD AUTO-CALIBRATION</span>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
            <span style={{
              background: debug.is_calibrated ? 'rgba(16, 185, 129, 0.2)' : 'rgba(100, 116, 139, 0.2)',
              color: debug.is_calibrated ? '#34d399' : '#94a3b8',
              padding: '2px 6px',
              borderRadius: '4px',
              fontWeight: 700,
              fontSize: '0.72rem'
            }}>
              {debug.is_calibrated ? 'AUTO-CALIBRATED' : 'DEFAULT HOMOGRAPHY'}
            </span>
            {debug.calibration_conf !== undefined && (
              <span style={{ color: '#94a3b8' }}>
                {(debug.calibration_conf * 100).toFixed(0)}% conf
              </span>
            )}
          </div>
          <div style={{ fontSize: '0.68rem', color: '#64748b' }}>
            Grass Segmentation &amp; Inset Boundary Rope
          </div>
        </div>

        {/* 6. Jersey / Team Identification */}
        <div style={{ background: '#0b1329', padding: '10px', borderRadius: '6px', border: '1px solid #1e293b' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', color: '#94a3b8', fontSize: '0.7rem', marginBottom: '6px' }}>
            <Cpu size={13} color="#a855f7" />
            <span>PLAYER / TEAM IDENTIFICATION</span>
          </div>
          <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap', marginBottom: '4px' }}>
            {debug.detected_teams && debug.detected_teams.length > 0 ? (
              debug.detected_teams.map((t, idx) => (
                <span key={idx} style={{
                  background: t.includes('Australia') ? '#eab308' : t.includes('India') ? '#3b82f6' : '#64748b',
                  color: t.includes('Australia') ? '#000' : '#fff',
                  padding: '1px 6px',
                  borderRadius: '4px',
                  fontWeight: 800,
                  fontSize: '0.7rem'
                }}>
                  {t}
                </span>
              ))
            ) : (
              <span style={{ color: '#64748b', fontSize: '0.72rem' }}>Scanning torso colors...</span>
            )}
          </div>
          <div style={{ fontSize: '0.68rem', color: '#64748b' }}>
            2-Cluster HSV K-Means on Torso Crops
          </div>
        </div>

        {/* 7. AI Commentary Engine Mode */}
        <div style={{ background: '#0b1329', padding: '10px', borderRadius: '6px', border: '1px solid #1e293b' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', color: '#94a3b8', fontSize: '0.7rem', marginBottom: '6px' }}>
            <Terminal size={13} color="#ec4899" />
            <span>AI COMMENTARY ENGINE</span>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
            <span style={{
              background: debug.llm_mode?.includes('REAL') ? 'rgba(59, 130, 246, 0.25)' : 'rgba(168, 85, 247, 0.25)',
              color: debug.llm_mode?.includes('REAL') ? '#60a5fa' : '#c084fc',
              padding: '2px 6px',
              borderRadius: '4px',
              fontWeight: 800,
              fontSize: '0.72rem'
            }}>
              {debug.llm_mode || 'FALLBACK (Templates)'}
            </span>
            <span style={{ color: '#94a3b8', fontSize: '0.7rem' }}>
              {debug.provider_name || 'Template Engine'}
            </span>
          </div>
          <div style={{ fontSize: '0.68rem', color: '#64748b' }}>
            Zero-latency failover to deterministic broadcast lines
          </div>
        </div>
      </div>
    </div>
  );
};
