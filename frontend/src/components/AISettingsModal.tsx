import React, { useState, useEffect } from 'react';
import { Sparkles, Key, Check, AlertCircle, X, Shield, Cpu, RefreshCw, Eye, EyeOff } from 'lucide-react';
import { wsService } from '../services/websocket';

interface Props {
  isOpen: boolean;
  onClose: () => void;
  onConfigSaved?: (provider: string, model: string) => void;
}

export const AISettingsModal: React.FC<Props> = ({ isOpen, onClose, onConfigSaved }) => {
  const [provider, setProvider] = useState<'google-gemini' | 'openai' | 'fallback'>('google-gemini');
  const [apiKey, setApiKey] = useState<string>('');
  const [modelName, setModelName] = useState<string>('gemini-2.0-flash');
  const [showKey, setShowKey] = useState<boolean>(false);
  const [isSaving, setIsSaving] = useState<boolean>(false);
  const [statusMessage, setStatusMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

  // Load saved settings from localStorage on mount
  useEffect(() => {
    const savedProvider = localStorage.getItem('cricket_llm_provider') as any;
    const savedKey = localStorage.getItem('cricket_llm_api_key') || '';
    const savedModel = localStorage.getItem('cricket_llm_model') || '';

    if (savedProvider) setProvider(savedProvider);
    if (savedKey) setApiKey(savedKey);
    if (savedModel) setModelName(savedModel);
  }, []);

  // Update default model on provider change
  const handleProviderChange = (newProvider: 'google-gemini' | 'openai' | 'fallback') => {
    setProvider(newProvider);
    setStatusMessage(null);
    if (newProvider === 'google-gemini') {
      setModelName('gemini-2.0-flash');
    } else if (newProvider === 'openai') {
      setModelName('gpt-4o-mini');
    } else {
      setModelName('local-templates');
    }
  };

  const handleSave = async () => {
    setIsSaving(true);
    setStatusMessage(null);

    try {
      // Save locally
      localStorage.setItem('cricket_llm_provider', provider);
      localStorage.setItem('cricket_llm_api_key', apiKey.trim());
      localStorage.setItem('cricket_llm_model', modelName.trim());

      // Send to backend via REST API
      const res = await fetch('http://localhost:8000/api/settings/llm', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          provider: provider,
          api_key: apiKey.trim(),
          model_name: modelName.trim()
        })
      });

      const data = await res.json();
      if (res.ok && data.status === 'ok') {
        setStatusMessage({
          type: 'success',
          text: `Configured ${provider === 'google-gemini' ? 'Google Gemini' : provider === 'openai' ? 'OpenAI' : 'Local Fallback'} successfully!`
        });

        // Also notify live WebSocket session
        wsService.sendCommand('set_llm_config' as any, {
          provider: provider,
          api_key: apiKey.trim(),
          model: modelName.trim()
        });

        if (onConfigSaved) {
          onConfigSaved(provider, modelName);
        }

        setTimeout(() => {
          onClose();
        }, 1200);
      } else {
        setStatusMessage({
          type: 'error',
          text: data.detail || 'Failed to update backend LLM configuration'
        });
      }
    } catch (err: any) {
      setStatusMessage({
        type: 'error',
        text: err.message || 'Network error updating settings'
      });
    } finally {
      setIsSaving(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div style={{
      position: 'fixed',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      backgroundColor: 'rgba(5, 8, 16, 0.85)',
      backdropFilter: 'blur(8px)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      zIndex: 9999,
      padding: '20px'
    }}>
      <div style={{
        background: '#0f172a',
        border: '1px solid #1e293b',
        borderRadius: '14px',
        width: '100%',
        maxWidth: '520px',
        boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.7)',
        overflow: 'hidden'
      }}>
        {/* Header */}
        <div style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          padding: '16px 20px',
          background: 'linear-gradient(135deg, #1e293b, #0f172a)',
          borderBottom: '1px solid #334155'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <div style={{
              background: 'linear-gradient(135deg, #3b82f6, #8b5cf6)',
              padding: '6px',
              borderRadius: '8px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center'
            }}>
              <Sparkles size={18} color="#fff" />
            </div>
            <div>
              <h3 style={{ margin: 0, fontSize: '1rem', fontWeight: 800, color: '#f8fafc' }}>
                AI Commentary Engine
              </h3>
              <p style={{ margin: 0, fontSize: '0.72rem', color: '#94a3b8' }}>
                Configure Dynamic LLM or Offline Fallback Mode
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            style={{
              background: 'transparent',
              border: 'none',
              color: '#94a3b8',
              cursor: 'pointer',
              padding: '4px',
              borderRadius: '6px'
            }}
          >
            <X size={18} />
          </button>
        </div>

        {/* Modal Body */}
        <div style={{ padding: '20px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
          {/* Provider Option Cards */}
          <div>
            <label style={{ display: 'block', fontSize: '0.75rem', color: '#94a3b8', fontWeight: 700, marginBottom: '8px' }}>
              SELECT AI COMMENTARY PROVIDER:
            </label>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: '8px' }}>
              {/* Google Gemini Card */}
              <div
                onClick={() => handleProviderChange('google-gemini')}
                style={{
                  background: provider === 'google-gemini' ? 'rgba(59, 130, 246, 0.15)' : '#1e293b',
                  border: `1.5px solid ${provider === 'google-gemini' ? '#3b82f6' : '#334155'}`,
                  borderRadius: '8px',
                  padding: '12px 14px',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between'
                }}
              >
                <div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <span style={{ fontWeight: 800, fontSize: '0.85rem', color: '#60a5fa' }}>Google Gemini</span>
                    <span style={{ fontSize: '0.65rem', background: '#2563eb', color: '#fff', padding: '1px 6px', borderRadius: '4px', fontWeight: 700 }}>RECOMMENDED</span>
                  </div>
                  <div style={{ fontSize: '0.72rem', color: '#94a3b8', marginTop: '2px' }}>
                    Official Google GenAI SDK (gemini-2.0-flash / gemini-1.5-flash)
                  </div>
                </div>
                <div style={{
                  width: '18px',
                  height: '18px',
                  borderRadius: '50%',
                  border: `2px solid ${provider === 'google-gemini' ? '#3b82f6' : '#64748b'}`,
                  background: provider === 'google-gemini' ? '#3b82f6' : 'transparent',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center'
                }}>
                  {provider === 'google-gemini' && <Check size={12} color="#fff" />}
                </div>
              </div>

              {/* OpenAI Card */}
              <div
                onClick={() => handleProviderChange('openai')}
                style={{
                  background: provider === 'openai' ? 'rgba(16, 185, 129, 0.15)' : '#1e293b',
                  border: `1.5px solid ${provider === 'openai' ? '#10b981' : '#334155'}`,
                  borderRadius: '8px',
                  padding: '12px 14px',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between'
                }}
              >
                <div>
                  <div style={{ fontWeight: 800, fontSize: '0.85rem', color: '#34d399' }}>
                    OpenAI / Compatible API
                  </div>
                  <div style={{ fontSize: '0.72rem', color: '#94a3b8', marginTop: '2px' }}>
                    Standard OpenAI completions (gpt-4o-mini, gpt-4o)
                  </div>
                </div>
                <div style={{
                  width: '18px',
                  height: '18px',
                  borderRadius: '50%',
                  border: `2px solid ${provider === 'openai' ? '#10b981' : '#64748b'}`,
                  background: provider === 'openai' ? '#10b981' : 'transparent',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center'
                }}>
                  {provider === 'openai' && <Check size={12} color="#fff" />}
                </div>
              </div>

              {/* Local Fallback Card */}
              <div
                onClick={() => handleProviderChange('fallback')}
                style={{
                  background: provider === 'fallback' ? 'rgba(168, 85, 247, 0.15)' : '#1e293b',
                  border: `1.5px solid ${provider === 'fallback' ? '#a855f7' : '#334155'}`,
                  borderRadius: '8px',
                  padding: '12px 14px',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between'
                }}
              >
                <div>
                  <div style={{ fontWeight: 800, fontSize: '0.85rem', color: '#c084fc' }}>
                    Deterministic Local Fallback
                  </div>
                  <div style={{ fontSize: '0.72rem', color: '#94a3b8', marginTop: '2px' }}>
                    Offline template engine. Instant latency, zero API keys required.
                  </div>
                </div>
                <div style={{
                  width: '18px',
                  height: '18px',
                  borderRadius: '50%',
                  border: `2px solid ${provider === 'fallback' ? '#a855f7' : '#64748b'}`,
                  background: provider === 'fallback' ? '#a855f7' : 'transparent',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center'
                }}>
                  {provider === 'fallback' && <Check size={12} color="#fff" />}
                </div>
              </div>
            </div>
          </div>

          {/* Conditional API Key & Model inputs */}
          {provider !== 'fallback' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', background: '#0b1120', padding: '14px', borderRadius: '8px', border: '1px solid #1e293b' }}>
              <div>
                <label style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.75rem', color: '#94a3b8', fontWeight: 600, marginBottom: '6px' }}>
                  <Key size={13} color="#f59e0b" />
                  API KEY ({provider === 'google-gemini' ? 'GEMINI_API_KEY' : 'OPENAI_API_KEY'}):
                </label>
                <div style={{ position: 'relative' }}>
                  <input
                    type={showKey ? 'text' : 'password'}
                    value={apiKey}
                    onChange={(e) => setApiKey(e.target.value)}
                    placeholder={provider === 'google-gemini' ? 'AIzaSy...' : 'sk-...'}
                    style={{
                      width: '100%',
                      background: '#1e293b',
                      border: '1px solid #475569',
                      borderRadius: '6px',
                      color: '#f8fafc',
                      padding: '8px 36px 8px 12px',
                      fontSize: '0.85rem',
                      fontFamily: 'monospace',
                      boxSizing: 'border-box'
                    }}
                  />
                  <button
                    type="button"
                    onClick={() => setShowKey(!showKey)}
                    style={{
                      position: 'absolute',
                      right: '8px',
                      top: '50%',
                      transform: 'translateY(-50%)',
                      background: 'transparent',
                      border: 'none',
                      color: '#94a3b8',
                      cursor: 'pointer'
                    }}
                  >
                    {showKey ? <EyeOff size={15} /> : <Eye size={15} />}
                  </button>
                </div>
                <div style={{ fontSize: '0.68rem', color: '#64748b', marginTop: '4px' }}>
                  Leave blank to use environment variable if set in server backend.
                </div>
              </div>

              <div>
                <label style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.75rem', color: '#94a3b8', fontWeight: 600, marginBottom: '6px' }}>
                  <Cpu size={13} color="#60a5fa" />
                  MODEL IDENTIFIER:
                </label>
                <input
                  type="text"
                  value={modelName}
                  onChange={(e) => setModelName(e.target.value)}
                  placeholder={provider === 'google-gemini' ? 'gemini-2.0-flash' : 'gpt-4o-mini'}
                  style={{
                    width: '100%',
                    background: '#1e293b',
                    border: '1px solid #475569',
                    borderRadius: '6px',
                    color: '#f8fafc',
                    padding: '8px 12px',
                    fontSize: '0.85rem',
                    boxSizing: 'border-box'
                  }}
                />
              </div>
            </div>
          )}

          {/* Feedback message banner */}
          {statusMessage && (
            <div style={{
              background: statusMessage.type === 'success' ? 'rgba(16, 185, 129, 0.15)' : 'rgba(239, 68, 68, 0.15)',
              border: `1px solid ${statusMessage.type === 'success' ? '#10b981' : '#ef4444'}`,
              borderRadius: '6px',
              padding: '8px 12px',
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              fontSize: '0.8rem',
              color: statusMessage.type === 'success' ? '#6ee7b7' : '#fca5a5'
            }}>
              {statusMessage.type === 'success' ? <Check size={16} /> : <AlertCircle size={16} />}
              <span>{statusMessage.text}</span>
            </div>
          )}

          {/* Action buttons */}
          <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '10px', marginTop: '4px' }}>
            <button
              onClick={onClose}
              style={{
                background: '#1e293b',
                border: '1px solid #334155',
                color: '#94a3b8',
                padding: '8px 16px',
                borderRadius: '6px',
                fontSize: '0.85rem',
                fontWeight: 600,
                cursor: 'pointer'
              }}
            >
              Cancel
            </button>
            <button
              onClick={handleSave}
              disabled={isSaving}
              style={{
                background: 'linear-gradient(135deg, #2563eb, #1d4ed8)',
                border: 'none',
                color: '#ffffff',
                padding: '8px 20px',
                borderRadius: '6px',
                fontSize: '0.85rem',
                fontWeight: 700,
                cursor: isSaving ? 'not-allowed' : 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
                boxShadow: '0 4px 12px rgba(37, 99, 235, 0.3)'
              }}
            >
              {isSaving && <RefreshCw size={14} style={{ animation: 'spin 1s linear infinite' }} />}
              {isSaving ? 'Saving...' : 'Save & Apply'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
