export class AudioStreamPlayer {
  private ctx: AudioContext | null = null;
  private gainNode: GainNode | null = null;
  private queue: ArrayBuffer[] = [];
  private isPlaying: boolean = false;
  private currentSource: AudioBufferSourceNode | null = null;
  private isMuted: boolean = false;

  constructor() {
    // AudioContext will be initialized on first user interaction or playback
  }

  private initContext() {
    if (!this.ctx) {
      const AudioCtx = window.AudioContext || (window as any).webkitAudioContext;
      this.ctx = new AudioCtx();
      this.gainNode = this.ctx.createGain();
      this.gainNode.gain.value = this.isMuted ? 0.0 : 1.0;
      this.gainNode.connect(this.ctx.destination);
    }
    if (this.ctx.state === 'suspended') {
      this.ctx.resume();
    }
  }

  public setMuted(muted: boolean) {
    this.isMuted = muted;
    if (this.gainNode) {
      this.gainNode.gain.value = muted ? 0.0 : 1.0;
    }
  }

  public clearQueue() {
    this.queue = [];
    if (this.currentSource) {
      try {
        this.currentSource.stop();
      } catch (e) {
        // Ignore if already stopped
      }
      this.currentSource = null;
    }
    this.isPlaying = false;
  }

  public async playBase64Audio(b64Audio: string) {
    if (this.isMuted) return;
    this.initContext();

    try {
      const binaryString = window.atob(b64Audio);
      const len = binaryString.length;
      const bytes = new Uint8Array(len);
      for (let i = 0; i < len; i++) {
        bytes[i] = binaryString.charCodeAt(i);
      }

      this.queue.push(bytes.buffer);
      if (!this.isPlaying) {
        this.playNext();
      }
    } catch (err) {
      console.error('Failed to decode base64 audio:', err);
    }
  }

  private async playNext() {
    if (this.queue.length === 0 || !this.ctx || !this.gainNode) {
      this.isPlaying = false;
      return;
    }

    this.isPlaying = true;
    const bufferData = this.queue.shift()!;

    try {
      const audioBuffer = await this.ctx.decodeAudioData(bufferData);
      const source = this.ctx.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(this.gainNode);

      this.currentSource = source;
      source.onended = () => {
        this.currentSource = null;
        this.playNext();
      };

      source.start(0);
    } catch (err) {
      console.warn('Audio decoding/playback error, skipping chunk:', err);
      this.playNext();
    }
  }
}

export const audioPlayer = new AudioStreamPlayer();
