import React from 'react';
import { Wifi, WifiOff, RefreshCw } from 'lucide-react';
import { ConnectionState } from '../services/websocket';

interface Props {
  state: ConnectionState;
  sourceType: string;
  isRunning: boolean;
}

export const ConnectionStatus: React.FC<Props> = ({ state, sourceType, isRunning }) => {
  const getBadge = () => {
    switch (state) {
      case 'connected':
        return (
          <span style={{ display: 'inline-flex', alignItems: 'center', gap: '6px', color: '#10b981', fontWeight: 600, fontSize: '0.85rem' }}>
            <span style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: '#10b981', display: 'inline-block', boxShadow: '0 0 8px #10b981' }}></span>
            LIVE WS CONNECTED
          </span>
        );
      case 'connecting':
        return (
          <span style={{ display: 'inline-flex', alignItems: 'center', gap: '6px', color: '#f59e0b', fontWeight: 600, fontSize: '0.85rem' }}>
            <RefreshCw size={14} className="spin" style={{ animation: 'spin 1s linear infinite' }} />
            CONNECTING...
          </span>
        );
      case 'disconnected':
      default:
        return (
          <span style={{ display: 'inline-flex', alignItems: 'center', gap: '6px', color: '#ef4444', fontWeight: 600, fontSize: '0.85rem' }}>
            <WifiOff size={14} />
            DISCONNECTED
          </span>
        );
    }
  };

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '16px', background: 'rgba(17, 24, 39, 0.8)', padding: '6px 14px', borderRadius: '8px', border: '1px solid #1f2937' }}>
      {getBadge()}
      <div style={{ width: '1px', height: '16px', background: '#374151' }}></div>
      <div style={{ fontSize: '0.8rem', color: '#9ca3af', display: 'flex', gap: '6px' }}>
        <span>SOURCE:</span>
        <span style={{ color: '#60a5fa', fontWeight: 600, textTransform: 'uppercase' }}>{sourceType}</span>
      </div>
      <div style={{ width: '1px', height: '16px', background: '#374151' }}></div>
      <div style={{ fontSize: '0.8rem', color: isRunning ? '#34d399' : '#9ca3af', fontWeight: 600 }}>
        {isRunning ? '● ACTIVE STREAM' : '○ IDLE'}
      </div>
    </div>
  );
};
