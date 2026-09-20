import React, { useState, useEffect } from 'react';
import { Trophy, Shield, Circle, Flame, Target } from 'lucide-react';
import { wsService, CricketEvent } from '../services/websocket';

export const EventTimeline: React.FC = () => {
  const [score, setScore] = useState<{ runs: number; wickets: number; over: number; ball: number }>({
    runs: 87,
    wickets: 2,
    over: 12,
    ball: 3
  });

  const [events, setEvents] = useState<CricketEvent[]>([
    {
      match_id: 'match_001',
      delivery_id: '12.1',
      frame_id: 110,
      timestamp: Date.now() / 1000 - 30,
      event_type: 'DOT_BALL',
      shot_type: 'DEFENSIVE',
      runs: 0,
      confidence: 0.94
    },
    {
      match_id: 'match_001',
      delivery_id: '12.2',
      frame_id: 220,
      timestamp: Date.now() / 1000 - 15,
      event_type: 'FOUR',
      shot_type: 'COVER_DRIVE',
      ball_speed_kmh: 128,
      runs: 4,
      confidence: 0.91
    }
  ]);

  useEffect(() => {
    const unsub = wsService.onEvent((ev) => {
      setEvents((prev) => [ev, ...prev.slice(0, 15)]);

      // Update scorecard based on event
      setScore((curr) => {
        let runsToAdd = ev.runs || 0;
        let wktToAdd = ev.event_type === 'WICKET' ? 1 : 0;
        if (ev.event_type === 'FOUR') runsToAdd = 4;
        if (ev.event_type === 'SIX') runsToAdd = 6;
        if (ev.event_type === 'SINGLE') runsToAdd = 1;
        if (ev.event_type === 'DOUBLE') runsToAdd = 2;

        let nextBall = curr.ball + 1;
        let nextOver = curr.over;
        if (nextBall > 6) {
          nextBall = 1;
          nextOver += 1;
        }

        return {
          runs: curr.runs + runsToAdd,
          wickets: curr.wickets + wktToAdd,
          over: nextOver,
          ball: nextBall
        };
      });
    });
    return () => unsub();
  }, []);

  const getEventBadge = (type: string) => {
    switch (type) {
      case 'FOUR':
        return <span style={{ background: '#2563eb', color: '#fff', padding: '2px 8px', borderRadius: '4px', fontWeight: 800, fontSize: '0.75rem' }}>4 (FOUR)</span>;
      case 'SIX':
        return <span style={{ background: '#7c3aed', color: '#fff', padding: '2px 8px', borderRadius: '4px', fontWeight: 800, fontSize: '0.75rem' }}>6 (SIX)</span>;
      case 'WICKET':
        return <span style={{ background: '#dc2626', color: '#fff', padding: '2px 8px', borderRadius: '4px', fontWeight: 800, fontSize: '0.75rem' }}>W (OUT)</span>;
      case 'DOT_BALL':
      case 'DOT':
        return <span style={{ background: '#4b5563', color: '#d1d5db', padding: '2px 8px', borderRadius: '4px', fontWeight: 700, fontSize: '0.75rem' }}>0 (DOT)</span>;
      default:
        return <span style={{ background: '#059669', color: '#fff', padding: '2px 8px', borderRadius: '4px', fontWeight: 700, fontSize: '0.75rem' }}>{type}</span>;
    }
  };

  return (
    <div style={{ background: '#111827', borderRadius: '12px', border: '1px solid #1f2937', padding: '16px', display: 'flex', flexDirection: 'column' }}>
      {/* Scoreboard Bar */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px', borderBottom: '1px solid #1f2937', paddingBottom: '10px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Trophy size={18} color="#eab308" />
          <span style={{ fontWeight: 700, fontSize: '0.9rem', letterSpacing: '0.05em', color: '#f3f4f6' }}>MATCH STATUS & TIMELINE</span>
        </div>

        {/* Live Score Counter */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '14px', background: '#1e293b', padding: '6px 14px', borderRadius: '8px', border: '1px solid #334155' }}>
          <div>
            <span style={{ fontSize: '0.7rem', color: '#9ca3af', marginRight: '6px' }}>SCORE:</span>
            <span style={{ fontSize: '1.1rem', fontWeight: 800, color: '#f8fafc', fontFamily: 'JetBrains Mono, monospace' }}>
              {score.runs}/{score.wickets}
            </span>
          </div>
          <div style={{ width: '1px', height: '14px', background: '#475569' }}></div>
          <div>
            <span style={{ fontSize: '0.7rem', color: '#9ca3af', marginRight: '6px' }}>OVERS:</span>
            <span style={{ fontSize: '1.1rem', fontWeight: 800, color: '#60a5fa', fontFamily: 'JetBrains Mono, monospace' }}>
              {score.over}.{score.ball}
            </span>
          </div>
          <div style={{ width: '1px', height: '14px', background: '#475569' }}></div>
          <div>
            <span style={{ fontSize: '0.7rem', color: '#9ca3af', marginRight: '6px' }}>CRR:</span>
            <span style={{ fontSize: '1.0rem', fontWeight: 700, color: '#34d399', fontFamily: 'JetBrains Mono, monospace' }}>
              {((score.runs / Math.max(1, score.over + score.ball / 6))).toFixed(2)}
            </span>
          </div>
        </div>
      </div>

      {/* Over Ball Indicators Row */}
      <div style={{ display: 'flex', gap: '8px', alignItems: 'center', marginBottom: '12px', background: '#1a2234', padding: '10px 14px', borderRadius: '8px' }}>
        <span style={{ fontSize: '0.75rem', fontWeight: 700, color: '#9ca3af', marginRight: '4px' }}>OVER {score.over}:</span>
        {events.slice(0, 6).reverse().map((ev, i) => (
          <div key={i} style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
            {getEventBadge(ev.event_type)}
          </div>
        ))}
      </div>

      {/* Events Log Table */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', maxHeight: '160px', overflowY: 'auto' }}>
        {events.map((ev, idx) => (
          <div key={idx} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: '#172033', padding: '6px 12px', borderRadius: '6px', fontSize: '0.8rem', border: '1px solid #1f2937' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <span style={{ fontFamily: 'JetBrains Mono, monospace', color: '#60a5fa', fontWeight: 700 }}>
                {ev.delivery_id || `D-${ev.frame_id}`}
              </span>
              {getEventBadge(ev.event_type)}
              {ev.shot_type && (
                <span style={{ color: '#e2e8f0', fontWeight: 600, fontSize: '0.75rem' }}>
                  Shot: {ev.shot_type}
                </span>
              )}
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: '12px', color: '#9ca3af', fontSize: '0.75rem' }}>
              {ev.ball_speed_kmh && (
                <span>Speed: <strong style={{ color: '#38bdf8' }}>{ev.ball_speed_kmh} km/h</strong></span>
              )}
              <span>Conf: <strong style={{ color: '#34d399' }}>{(ev.confidence * 100).toFixed(0)}%</strong></span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
