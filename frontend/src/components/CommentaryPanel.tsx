import React, { useState, useEffect } from 'react';
import { Volume2, VolumeX, MessageSquare, Sparkles, Mic } from 'lucide-react';
import { wsService, CommentaryItem } from '../services/websocket';
import { audioPlayer } from '../services/audioPlayer';

interface Props {
  stylePreset?: string;
  onStyleChange?: (style: string) => void;
}

export const CommentaryPanel: React.FC<Props> = ({ stylePreset = 'PROFESSIONAL', onStyleChange }) => {
  const [commentaries, setCommentaries] = useState<CommentaryItem[]>([
    {
      id: 'init-01',
      delivery_id: '12.1',
      event_type: 'DOT_BALL',
      text: 'Good length delivery outside off stump, batsman defends solidly towards short cover. No run taken.',
      style: 'PROFESSIONAL',
      is_fallback: false,
      llm_latency_ms: 215,
      timestamp: Date.now() / 1000 - 15
    }
  ]);
  const [isMuted, setIsMuted] = useState<boolean>(false);
  const [currentSpeech, setCurrentSpeech] = useState<string>('');

  useEffect(() => {
    const unsub = wsService.onCommentary((item) => {
      setCommentaries((prev) => [item, ...prev.slice(0, 19)]);
      setCurrentSpeech(item.text);
    });
    return () => unsub();
  }, []);

  const toggleMute = () => {
    const nextState = !isMuted;
    setIsMuted(nextState);
    audioPlayer.setMuted(nextState);
  };

  const handleStyleChange = (style: string) => {
    if (onStyleChange) onStyleChange(style);
    wsService.sendCommand('set_style' as any, { style });
  };

  const latest = commentaries[0];

  return (
    <div style={{ background: '#111827', borderRadius: '12px', border: '1px solid #1f2937', padding: '16px', display: 'flex', flexDirection: 'column' }}>
      {/* Header with audio and style controls */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px', borderBottom: '1px solid #1f2937', paddingBottom: '10px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <MessageSquare size={18} color="#f59e0b" />
          <span style={{ fontWeight: 700, fontSize: '0.9rem', letterSpacing: '0.05em', color: '#f3f4f6' }}>LIVE AI COMMENTARY</span>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          {/* Commentary Style Selector */}
          <select
            value={stylePreset}
            onChange={(e) => handleStyleChange(e.target.value)}
            style={{
              background: '#1e293b',
              border: '1px solid #334155',
              color: '#f8fafc',
              padding: '4px 8px',
              borderRadius: '6px',
              fontSize: '0.75rem',
              fontWeight: 600,
              cursor: 'pointer'
            }}
          >
            <option value="PROFESSIONAL">PROFESSIONAL</option>
            <option value="ENERGETIC">ENERGETIC</option>
            <option value="ANALYTICAL">ANALYTICAL</option>
            <option value="MINIMAL">MINIMAL</option>
          </select>

          {/* Mute/Unmute Audio Button */}
          <button
            onClick={toggleMute}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '4px',
              background: isMuted ? 'rgba(239, 68, 68, 0.2)' : 'rgba(16, 185, 129, 0.2)',
              border: `1px solid ${isMuted ? '#ef4444' : '#10b981'}`,
              color: isMuted ? '#f87171' : '#34d399',
              padding: '4px 8px',
              borderRadius: '6px',
              cursor: 'pointer',
              fontSize: '0.75rem',
              fontWeight: 600
            }}
          >
            {isMuted ? <VolumeX size={14} /> : <Volume2 size={14} />}
            {isMuted ? 'MUTED' : 'AUDIO ON'}
          </button>
        </div>
      </div>

      {/* Hero Commentary Spotlight Banner */}
      <div style={{ background: 'linear-gradient(135deg, rgba(30, 41, 59, 0.8), rgba(15, 23, 42, 0.95))', border: '1px solid #334155', borderRadius: '8px', padding: '14px 16px', marginBottom: '12px', position: 'relative' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <Sparkles size={14} color="#f59e0b" />
            <span style={{ fontSize: '0.7rem', fontWeight: 700, color: '#f59e0b', letterSpacing: '0.05em' }}>LATEST BROADCAST CALL</span>
            {latest && (
              <span style={{ fontSize: '0.65rem', padding: '1px 6px', borderRadius: '4px', background: latest.is_fallback ? '#4b5563' : '#2563eb', color: '#fff', fontWeight: 600 }}>
                {latest.is_fallback ? 'FALLBACK COMMENTARY' : 'AI COMMENTARY (LLM)'}
              </span>
            )}
          </div>
          {latest && latest.llm_latency_ms > 0 && (
            <span style={{ fontSize: '0.7rem', color: '#9ca3af', fontFamily: 'JetBrains Mono, monospace' }}>
              Latency: {latest.llm_latency_ms.toFixed(0)}ms
            </span>
          )}
        </div>

        <div style={{ fontSize: '1.05rem', fontWeight: 600, color: '#f8fafc', lineHeight: '1.45', minHeight: '44px', display: 'flex', alignItems: 'center' }}>
          "{latest ? latest.text : 'Awaiting cricket event from live video feed...'}"
        </div>
      </div>

      {/* Recent Commentary Feed */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', maxHeight: '130px', overflowY: 'auto', paddingRight: '4px' }}>
        {commentaries.slice(1).map((item) => (
          <div key={item.id} style={{ display: 'flex', gap: '10px', alignItems: 'flex-start', background: '#1a2234', padding: '8px 12px', borderRadius: '6px', fontSize: '0.8rem', border: '1px solid #1f2937' }}>
            <span style={{ background: '#374151', color: '#e5e7eb', padding: '2px 6px', borderRadius: '4px', fontWeight: 700, fontSize: '0.7rem', whiteSpace: 'nowrap' }}>
              {item.delivery_id || 'CALL'}
            </span>
            <span style={{ color: '#d1d5db', flex: 1 }}>{item.text}</span>
            <span style={{ fontSize: '0.7rem', color: '#6b7280', whiteSpace: 'nowrap' }}>{item.event_type}</span>
          </div>
        ))}
      </div>
    </div>
  );
};
