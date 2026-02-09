import { useEffect, useMemo, useRef, useState } from 'react';
import { Button } from '../../components/ui/Button';
import { GlassCard } from '../../components/ui/GlassCard';
import * as api from '../../services/api';
import './LogsSpeechPage.css';

const TARGET_SAMPLE_RATE = 16000;
const CHUNK_MS = 2400;
const MAX_PENDING_CHUNKS = 4;

type WebkitWindow = Window & { webkitAudioContext?: typeof AudioContext };

function resampleLinear(samples: Float32Array, fromRate: number, toRate: number): Float32Array {
  if (fromRate === toRate || samples.length === 0) {
    return samples;
  }

  const ratio = fromRate / toRate;
  const newLength = Math.floor(samples.length / ratio);
  const out = new Float32Array(newLength);

  for (let i = 0; i < newLength; i += 1) {
    const src = i * ratio;
    const left = Math.floor(src);
    const right = Math.min(left + 1, samples.length - 1);
    const alpha = src - left;
    out[i] = samples[left] * (1 - alpha) + samples[right] * alpha;
  }

  return out;
}

function concatFloat32(chunks: Float32Array[]): Float32Array {
  if (chunks.length === 0) {
    return new Float32Array(0);
  }

  let total = 0;
  for (const chunk of chunks) {
    total += chunk.length;
  }

  const merged = new Float32Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    merged.set(chunk, offset);
    offset += chunk.length;
  }

  return merged;
}

function mergeLogs(base: api.SpeechLogEntry[], incoming: api.SpeechLogEntry[]): api.SpeechLogEntry[] {
  const map = new Map<number, api.SpeechLogEntry>();
  for (const log of base) {
    map.set(log.id, log);
  }
  for (const log of incoming) {
    map.set(log.id, log);
  }
  return Array.from(map.values()).sort((a, b) => b.id - a.id);
}

function wsUrl(): string {
  const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
  const host = window.location.host;
  return `${protocol}://${host}/api/ws`;
}

function isGenericSpeakerLabel(value: string): boolean {
  return /^\s*(игрок|player|говорящий|speaker)\s+/i.test(value) || /^\s*(\?|неизвестный|unknown)\s*$/i.test(value);
}

function resolveSpeakerLabel(log: api.SpeechLogEntry, playerNames: Map<number, string>): string {
  const playerName = log.speaker_id != null ? playerNames.get(Number(log.speaker_id)) : undefined;
  if (playerName && !isGenericSpeakerLabel(playerName)) {
    return playerName;
  }

  if (log.speaker_label && !isGenericSpeakerLabel(log.speaker_label)) {
    return log.speaker_label;
  }

  if (log.speaker_name && !isGenericSpeakerLabel(log.speaker_name)) {
    return log.speaker_name;
  }

  if (log.speaker_id != null) {
    return `Игрок ${log.speaker_id}`;
  }

  return 'Неизвестный';
}

export function LogsSpeechPage() {
  const [logs, setLogs] = useState<api.SpeechLogEntry[]>([]);
  const [playerNames, setPlayerNames] = useState<Map<number, string>>(new Map());
  const [recording, setRecording] = useState(false);
  const [processing, setProcessing] = useState(false);
  const [gameRunning, setGameRunning] = useState<boolean | null>(null);
  const [error, setError] = useState('');
  const [asrWarning, setAsrWarning] = useState('');

  const streamRef = useRef<MediaStream | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  const audioContextRef = useRef<AudioContext | null>(null);
  const sourceNodeRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const processorNodeRef = useRef<ScriptProcessorNode | null>(null);
  const sampleRateRef = useRef<number>(TARGET_SAMPLE_RATE);
  const pcmChunksRef = useRef<Float32Array[]>([]);
  const pendingChunksRef = useRef<Array<{ samples: Float32Array; sampleRate: number }>>([]);
  const flushTimerRef = useRef<number | null>(null);

  const startInFlightRef = useRef(false);
  const stopInFlightRef = useRef(false);
  const processInFlightRef = useRef(false);

  const formattedLogs = useMemo(
    () =>
      logs.map((log) => {
        const label = resolveSpeakerLabel(log, playerNames);
        const kindLabel = (log.kind || '').toString().trim().toLowerCase() === 'gesture_transcript'
          ? 'транскрипция'
          : 'текст';
        return {
          ...log,
          speaker_label: label,
          line: `"${label}"(${kindLabel}): ${(log.text || '').trim() || '...'};`,
        };
      }),
    [logs, playerNames],
  );

  async function loadPlayersMap() {
    try {
      const response = await api.listPlayers();
      const map = new Map<number, string>();
      (response.players || []).forEach((player) => {
        const name = (player.name || '').trim();
        if (name) {
          map.set(player.id, name);
        }
      });
      setPlayerNames(map);
    } catch {
      // Optional fallback, logs continue to work without players map.
    }
  }

  function stopTracks() {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
  }

  async function loadLogs() {
    try {
      const response = await api.speechLogsList(400);
      if (!response.ok) {
        throw new Error(response.error || 'Не удалось загрузить speech logs');
      }
      setLogs(mergeLogs([], response.logs || []));
    } catch (loadError: any) {
      setError(loadError?.message || 'Ошибка загрузки logs_speech');
    }
  }

  async function processSamples(samples: Float32Array, sampleRate: number) {
    if (samples.length === 0 || processInFlightRef.current) {
      return;
    }

    try {
      processInFlightRef.current = true;
      setProcessing(true);

      const normalized = resampleLinear(samples, sampleRate, TARGET_SAMPLE_RATE);
      if (normalized.length < TARGET_SAMPLE_RATE / 2) {
        return;
      }

      const response = await api.speechRecognizeChunk(Array.from(normalized), TARGET_SAMPLE_RATE, true);
      if (!response.ok) {
        throw new Error(response.error || 'Ошибка распознавания речи');
      }

      if (response.asr_error) {
        setAsrWarning(response.asr_error);
      } else {
        setAsrWarning('');
      }

      if (response.entry) {
        setLogs((prev) => mergeLogs(prev, [response.entry!]));
      }
    } catch (chunkError: any) {
      setError(chunkError?.message || 'Ошибка обработки аудио чанка');
    } finally {
      processInFlightRef.current = false;
      setProcessing(false);
      if (pendingChunksRef.current.length > 0) {
        void drainPendingChunks();
      }
    }
  }

  async function drainPendingChunks() {
    if (processInFlightRef.current) {
      return;
    }

    const next = pendingChunksRef.current.shift();
    if (!next) {
      return;
    }

    await processSamples(next.samples, next.sampleRate);
  }

  async function flushBufferedSamples() {
    const merged = concatFloat32(pcmChunksRef.current);
    pcmChunksRef.current = [];

    if (merged.length < sampleRateRef.current / 2) {
      return;
    }

    pendingChunksRef.current.push({ samples: merged, sampleRate: sampleRateRef.current });
    if (pendingChunksRef.current.length > MAX_PENDING_CHUNKS) {
      const dropCount = pendingChunksRef.current.length - MAX_PENDING_CHUNKS;
      pendingChunksRef.current.splice(0, dropCount);
    }
    void drainPendingChunks();
  }

  async function startRecording() {
    if (startInFlightRef.current || stopInFlightRef.current) {
      return;
    }

    if (processorNodeRef.current && audioContextRef.current) {
      setRecording(true);
      return;
    }

    startInFlightRef.current = true;
    try {
      setError('');

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });

      const audioCtor = window.AudioContext || (window as WebkitWindow).webkitAudioContext;
      if (!audioCtor) {
        throw new Error('Ваш браузер не поддерживает WebAudio');
      }

      const context = new audioCtor();
      await context.resume().catch(() => undefined);

      const source = context.createMediaStreamSource(stream);
      const processor = context.createScriptProcessor(4096, 1, 1);

      processor.onaudioprocess = (event: AudioProcessingEvent) => {
        const input = event.inputBuffer.getChannelData(0);
        if (input.length > 0) {
          const copy = new Float32Array(input.length);
          copy.set(input);
          pcmChunksRef.current.push(copy);
        }

        const output = event.outputBuffer.getChannelData(0);
        output.fill(0);
      };

      source.connect(processor);
      processor.connect(context.destination);

      streamRef.current = stream;
      audioContextRef.current = context;
      sourceNodeRef.current = source;
      processorNodeRef.current = processor;
      sampleRateRef.current = context.sampleRate;
      pcmChunksRef.current = [];
      pendingChunksRef.current = [];

      if (flushTimerRef.current !== null) {
        window.clearInterval(flushTimerRef.current);
      }
      flushTimerRef.current = window.setInterval(() => {
        void flushBufferedSamples();
      }, CHUNK_MS);

      setRecording(true);
    } catch (startError: any) {
      setRecording(false);
      setError(startError?.message || 'Не удалось получить доступ к микрофону');
      await stopRecording();
    } finally {
      startInFlightRef.current = false;
    }
  }

  async function stopRecording() {
    if (stopInFlightRef.current) {
      return;
    }
    stopInFlightRef.current = true;

    try {
      if (flushTimerRef.current !== null) {
        window.clearInterval(flushTimerRef.current);
        flushTimerRef.current = null;
      }

      if (processorNodeRef.current) {
        processorNodeRef.current.onaudioprocess = null;
        processorNodeRef.current.disconnect();
        processorNodeRef.current = null;
      }

      if (sourceNodeRef.current) {
        sourceNodeRef.current.disconnect();
        sourceNodeRef.current = null;
      }

      await flushBufferedSamples();

      if (audioContextRef.current) {
        await audioContextRef.current.close().catch(() => undefined);
        audioContextRef.current = null;
      }

      pcmChunksRef.current = [];
      pendingChunksRef.current = [];
      stopTracks();
      setRecording(false);
    } finally {
      stopInFlightRef.current = false;
    }
  }

  async function clearLogs() {
    try {
      const response = await api.speechLogsClear();
      if (!response.ok) {
        throw new Error(response.error || 'Не удалось очистить logs_speech');
      }
      setLogs([]);
    } catch (clearError: any) {
      setError(clearError?.message || 'Ошибка очистки logs_speech');
    }
  }

  useEffect(() => {
    let disposed = false;

    async function syncRecordingWithGame() {
      try {
        const status = await api.getVideoStatus();
        if (disposed) {
          return;
        }

        const running = Boolean(status.running);
        setGameRunning(running);

        if (running) {
          await startRecording();
        } else {
          await stopRecording();
        }
      } catch {
        if (!disposed) {
          setGameRunning(null);
        }
      }
    }

    void loadPlayersMap();
    void loadLogs();
    void syncRecordingWithGame();

    const logsPoll = window.setInterval(() => {
      void loadLogs();
    }, 4000);

    const gamePoll = window.setInterval(() => {
      void syncRecordingWithGame();
    }, 2500);

    try {
      const ws = new WebSocket(wsUrl());
      wsRef.current = ws;
      ws.onmessage = (event) => {
        if (typeof event.data !== 'string') {
          return;
        }

        try {
          const payload = JSON.parse(event.data) as { type?: string; entry?: api.SpeechLogEntry };
          if (payload.type === 'speech.log' && payload.entry) {
            setLogs((prev) => mergeLogs(prev, [payload.entry!]));
          }
        } catch {
          // ignore non-json frames
        }
      };
    } catch {
      // optional realtime updates
    }

    return () => {
      disposed = true;
      window.clearInterval(logsPoll);
      window.clearInterval(gamePoll);
      void stopRecording();

      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="setup-shell">
      <div className="setup-container">
        <div className="setup-wizard speech-logs">
          <GlassCard className="speech-logs__header">
            <div>
              <h1 className="speech-logs__title">logs_speech</h1>
              <p className="speech-logs__subtitle">Логи распознавания речи и говорящего в реальном времени.</p>
            </div>
            <div className="speech-logs__status">
              <span className={`status-tag ${recording ? 'status-tag--success' : 'status-tag--warn'}`}>
                {recording ? 'микрофон: запись' : 'микрофон: остановлен'}
              </span>
              {processing && <span className="status-tag status-tag--warn">обработка...</span>}
            </div>
          </GlassCard>

          <div className="speech-logs__actions">
            <Button variant="secondary" onClick={() => void clearLogs()}>
              Очистить логи
            </Button>
          </div>

          {(error || asrWarning || gameRunning === false) && (
            <GlassCard className="speech-logs__notice">
              {gameRunning === false && (
                <div className="speech-logs__warn">Процесс игры не запущен. Логи начнут писаться автоматически после старта игры.</div>
              )}
              {error && <div className="speech-logs__error">{error}</div>}
              {asrWarning && <div className="speech-logs__warn">ASR: {asrWarning}</div>}
            </GlassCard>
          )}

          <GlassCard className="speech-logs__list-card">
            <div className="speech-logs__list-head">
              <h2>Логи звука</h2>
              <span>{formattedLogs.length}</span>
            </div>

            <div className="speech-logs__list">
              {formattedLogs.length === 0 && <div className="speech-logs__empty">Логов пока нет</div>}
              {formattedLogs.map((log) => (
                <div key={log.id} className="speech-logs__item">
                  <div className="speech-logs__line">{log.line}</div>
                  <div className="speech-logs__meta">
                    <span>ID: {log.id}</span>
                    <span>{new Date(log.timestamp * 1000).toLocaleTimeString()}</span>
                    <span>conf: {(log.confidence || 0).toFixed(2)}</span>
                  </div>
                </div>
              ))}
            </div>
          </GlassCard>
        </div>
      </div>
    </div>
  );
}
