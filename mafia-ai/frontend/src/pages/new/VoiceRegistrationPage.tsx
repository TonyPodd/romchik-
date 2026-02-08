import { useEffect, useMemo, useRef, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { SetupStageHeader } from '../../components/SetupStageHeader';
import { Button } from '../../components/ui/Button';
import { GlassCard } from '../../components/ui/GlassCard';
import * as api from '../../services/api';
import './VoiceRegistrationPage.css';

type Player = {
  id: number;
  name: string;
  registered: boolean;
  samplesCount: number;
};

type CaptureMode = 'register' | 'test';

type TestResult = {
  correct: boolean;
  expectedName: string | null;
  predictedName: string | null;
  confidence: number;
};

const REGISTER_DURATION_MS = 4200;
const TEST_DURATION_MS = 2800;
const TARGET_SAMPLE_RATE = 16000;

function buildPlayers(
  incomingPlayers: Array<{ id?: number; name?: string }>,
  fallbackCount: number,
): Player[] {
  if (incomingPlayers.length > 0) {
    return incomingPlayers.map((player, index) => {
      const id = Number(player.id) || index + 1;
      const name = player.name?.trim() || `Игрок ${id}`;
      return {
        id,
        name,
        registered: false,
        samplesCount: 0,
      };
    });
  }

  return Array.from({ length: fallbackCount }, (_, index) => ({
    id: index + 1,
    name: `Игрок ${index + 1}`,
    registered: false,
    samplesCount: 0,
  }));
}

function findNextPending(players: Player[], currentIndex: number): number {
  for (let i = currentIndex + 1; i < players.length; i += 1) {
    if (!players[i].registered) {
      return i;
    }
  }
  for (let i = 0; i < players.length; i += 1) {
    if (!players[i].registered) {
      return i;
    }
  }
  return currentIndex;
}

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

function splitForEnrollment(samples: Float32Array, parts: number = 3): number[][] {
  const minSegment = Math.floor(TARGET_SAMPLE_RATE * 0.85);
  const segmentLength = Math.floor(samples.length / parts);
  if (segmentLength < minSegment) {
    throw new Error('Запись слишком короткая. Говорите четко 3-4 секунды.');
  }

  const chunks: number[][] = [];
  for (let i = 0; i < parts; i += 1) {
    const start = i * segmentLength;
    const end = i === parts - 1 ? samples.length : (i + 1) * segmentLength;
    chunks.push(Array.from(samples.slice(start, end)));
  }
  return chunks;
}

export function VoiceRegistrationPage() {
  const navigate = useNavigate();
  const location = useLocation();

  const fallbackCount = Number(location.state?.playerCount) || 10;
  const incomingPlayers = Array.isArray(location.state?.players) ? location.state.players : [];

  const [players, setPlayers] = useState<Player[]>(() => buildPlayers(incomingPlayers, fallbackCount));
  const [currentIndex, setCurrentIndex] = useState(0);
  const [captureMode, setCaptureMode] = useState<CaptureMode | null>(null);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState('');
  const [testResult, setTestResult] = useState<TestResult | null>(null);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const progressIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const autoStopTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const isRecording = captureMode === 'register';
  const isTesting = captureMode === 'test';
  const isCapturing = captureMode !== null;

  const currentPlayer = players[currentIndex];
  const registeredCount = players.filter((player) => player.registered).length;
  const allRegistered = players.length > 0 && registeredCount === players.length;

  const playerCount = useMemo(() => players.length || fallbackCount, [players.length, fallbackCount]);

  useEffect(() => {
    void syncProfiles();
    return () => {
      cleanupCapture();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function clearCaptureTimers() {
    if (progressIntervalRef.current) {
      clearInterval(progressIntervalRef.current);
      progressIntervalRef.current = null;
    }
    if (autoStopTimeoutRef.current) {
      clearTimeout(autoStopTimeoutRef.current);
      autoStopTimeoutRef.current = null;
    }
  }

  function stopMicStream() {
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((track) => track.stop());
      mediaStreamRef.current = null;
    }
  }

  function cleanupCapture() {
    clearCaptureTimers();
    stopMicStream();
    mediaRecorderRef.current = null;
    chunksRef.current = [];
    setCaptureMode(null);
    setProgress(0);
  }

  async function syncProfiles() {
    try {
      const response = await api.voiceListProfiles();
      if (!response.ok) {
        throw new Error(response.error || 'Не удалось загрузить голосовые профили');
      }

      const byId = new Map<number, api.VoiceProfile>();
      response.profiles.forEach((profile) => byId.set(profile.player_id, profile));

      setPlayers((prev) =>
        prev.map((player) => {
          const profile = byId.get(player.id);
          return {
            ...player,
            registered: Boolean(profile),
            samplesCount: profile?.samples_count || 0,
          };
        }),
      );
    } catch (err: any) {
      setError(err?.message || 'Ошибка загрузки голосовых профилей');
    }
  }

  async function decodeToMono(blob: Blob): Promise<{ samples: Float32Array; sampleRate: number }> {
    const arrayBuffer = await blob.arrayBuffer();
    const context = new AudioContext();
    try {
      const decoded = await context.decodeAudioData(arrayBuffer);
      const channelData = decoded.getChannelData(0);
      const samples = new Float32Array(channelData.length);
      samples.set(channelData);
      return { samples, sampleRate: decoded.sampleRate };
    } finally {
      await context.close();
    }
  }

  async function handleRecordedBlob(mode: CaptureMode, blob: Blob) {
    try {
      const { samples, sampleRate } = await decodeToMono(blob);
      const normalized = resampleLinear(samples, sampleRate, TARGET_SAMPLE_RATE);

      if (mode === 'register') {
        const chunks = splitForEnrollment(normalized, 3);
        const response = await api.voiceRegister(
          currentPlayer.id,
          currentPlayer.name,
          chunks,
          TARGET_SAMPLE_RATE,
        );
        if (!response.ok) {
          throw new Error(response.error || 'Не удалось зарегистрировать голос');
        }

        setPlayers((prev) => {
          const updated = prev.map((player, index) =>
            index === currentIndex
              ? {
                  ...player,
                  registered: true,
                  samplesCount: response.samples_registered || Math.max(player.samplesCount, 3),
                }
              : player,
          );
          const next = findNextPending(updated, currentIndex);
          setCurrentIndex(next);
          return updated;
        });
        setTestResult(null);
        setError('');
      } else {
        const response = await api.voiceTestIdentify(
          currentPlayer.id,
          Array.from(normalized),
          TARGET_SAMPLE_RATE,
        );
        if (!response.ok) {
          throw new Error(response.error || 'Тест распознавания завершился с ошибкой');
        }

        setTestResult({
          correct: Boolean(response.correct),
          expectedName: response.expected_player_name || currentPlayer.name,
          predictedName: response.predicted_player_name || null,
          confidence: Number(response.confidence || 0),
        });
      }
    } catch (err: any) {
      setError(err?.message || 'Ошибка обработки аудио');
    }
  }

  async function startCapture(mode: CaptureMode) {
    if (isCapturing) {
      return;
    }

    try {
      setError('');
      if (mode === 'test') {
        setTestResult(null);
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });
      mediaStreamRef.current = stream;

      const preferredMime = 'audio/webm;codecs=opus';
      const options = MediaRecorder.isTypeSupported(preferredMime) ? { mimeType: preferredMime } : undefined;
      const recorder = new MediaRecorder(stream, options);
      mediaRecorderRef.current = recorder;
      chunksRef.current = [];
      setCaptureMode(mode);
      setProgress(0);

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      };

      recorder.onerror = () => {
        setError('Ошибка записи аудио');
      };

      recorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: recorder.mimeType || 'audio/webm' });
        void handleRecordedBlob(mode, blob).finally(() => {
          cleanupCapture();
        });
      };

      const duration = mode === 'register' ? REGISTER_DURATION_MS : TEST_DURATION_MS;
      const startedAt = Date.now();

      progressIntervalRef.current = setInterval(() => {
        const ratio = Math.min(1, (Date.now() - startedAt) / duration);
        setProgress(Math.round(ratio * 100));
      }, 100);

      autoStopTimeoutRef.current = setTimeout(() => {
        if (recorder.state === 'recording') {
          recorder.stop();
        }
      }, duration);

      recorder.start();
    } catch (err: any) {
      cleanupCapture();
      setError(err?.message || 'Не удалось получить доступ к микрофону');
    }
  }

  function stopCapture() {
    if (mediaRecorderRef.current?.state === 'recording') {
      mediaRecorderRef.current.stop();
    }
  }

  async function handleResetProfiles() {
    if (!confirm('Удалить все зарегистрированные голоса?')) {
      return;
    }
    try {
      const response = await api.voiceClearProfiles();
      if (!response.ok) {
        throw new Error(response.error || 'Не удалось очистить голосовые профили');
      }
      setPlayers((prev) => prev.map((player) => ({ ...player, registered: false, samplesCount: 0 })));
      setCurrentIndex(0);
      setError('');
      setTestResult(null);
    } catch (err: any) {
      setError(err?.message || 'Ошибка очистки профилей');
    }
  }

  function handleSkip() {
    if (currentIndex < players.length - 1) {
      setCurrentIndex(currentIndex + 1);
      setTestResult(null);
      setError('');
    }
  }

  return (
    <div className="setup-shell">
      <div className="setup-container setup-container--with-stage">
        <GlassCard className="setup-stage-shell">
          <SetupStageHeader
            current="voice"
            title="Регистрация голосов"
            subtitle="Сохраните голосовые образцы и проверьте корректность распознавания."
          />
        </GlassCard>

        <div className="setup-wizard">
          <GlassCard className="voice-page__summary">
            <div className="voice-page__summary-row">
              <span>Зарегистрировано голосов</span>
              <strong>{registeredCount} / {playerCount}</strong>
            </div>
            <div className="setup-progress">
              <progress className="setup-progress__native" value={registeredCount} max={playerCount} />
            </div>
          </GlassCard>

          <div className="setup-grid setup-grid--voice">
            <GlassCard className="voice-page__main">
              <div className="voice-page__header">
                <h2 className="voice-page__title">{currentPlayer?.name || 'Игрок'}</h2>
                <span className={`status-tag ${currentPlayer?.registered ? 'status-tag--success' : 'status-tag--warn'}`}>
                  {currentPlayer?.registered ? `Готово (${currentPlayer.samplesCount})` : 'Ожидает'}
                </span>
              </div>

              <p className="voice-page__prompt">
                Произнесите: "Я {currentPlayer?.name}, игрок номер {currentPlayer?.id}"
              </p>

              <div className="voice-stage">
                {isRecording && (
                  <div className="voice-page__recording">
                    <span className="voice-stage__pulse" />
                    <strong>Запись эталона... {Math.max(1, Math.ceil((100 - progress) / 24))}с</strong>
                  </div>
                )}
                {isTesting && (
                  <div className="voice-page__recording">
                    <span className="voice-stage__pulse voice-stage__pulse--test" />
                    <strong>Тест распознавания... {Math.max(1, Math.ceil((100 - progress) / 36))}с</strong>
                  </div>
                )}
                {!isCapturing && currentPlayer?.registered && (
                  <div className="voice-page__status status-tag status-tag--success">Голос зарегистрирован</div>
                )}
                {!isCapturing && !currentPlayer?.registered && (
                  <div className="voice-page__status">Готово к записи</div>
                )}
                {isCapturing && (
                  <div className="voice-page__capture-progress">
                    <progress className="setup-progress__native" value={progress} max={100} />
                  </div>
                )}
              </div>

              {error && <div className="voice-page__error">{error}</div>}

              {testResult && (
                <div className={`voice-page__test-result ${testResult.correct ? 'is-success' : 'is-fail'}`.trim()}>
                  <strong>{testResult.correct ? 'Тест пройден' : 'Тест не пройден'}</strong>
                  <span>
                    Ожидался: {testResult.expectedName || '—'} · Определен: {testResult.predictedName || 'не распознан'}
                  </span>
                  <span>Уверенность: {(testResult.confidence * 100).toFixed(1)}%</span>
                </div>
              )}

              <div className="voice-page__actions">
                {isCapturing ? (
                  <Button variant="danger" onClick={stopCapture} fullWidth>
                    Остановить
                  </Button>
                ) : (
                  <>
                    <Button variant="secondary" onClick={handleSkip} fullWidth>
                      Пропустить
                    </Button>
                    <Button onClick={() => void startCapture('register')} fullWidth>
                      {currentPlayer?.registered ? 'Перезаписать' : 'Записать 4с'}
                    </Button>
                    <Button
                      variant="secondary"
                      onClick={() => void startCapture('test')}
                      disabled={!currentPlayer?.registered}
                      fullWidth
                    >
                      Тест распознавания
                    </Button>
                  </>
                )}
              </div>
            </GlassCard>

            <GlassCard className="voice-page__list">
              <div className="voice-page__list-head">
                <h2 className="voice-page__title">Игровые места</h2>
                <Button variant="secondary" size="sm" onClick={() => void handleResetProfiles()}>
                  Сброс голосов
                </Button>
              </div>
              <div className="voice-page__players">
                {players.map((player, index) => (
                  <button
                    type="button"
                    key={player.id}
                    className={`voice-page__player ${index === currentIndex ? 'is-active' : ''} ${player.registered ? 'is-filled' : ''}`.trim()}
                    onClick={() => {
                      if (!isCapturing) {
                        setCurrentIndex(index);
                        setTestResult(null);
                        setError('');
                      }
                    }}
                  >
                    <span className="voice-page__player-id">{player.registered ? '✓' : player.id}</span>
                    <span className="voice-page__player-meta">
                      <strong>{player.name}</strong>
                      <small>
                        {player.registered
                          ? `Профиль записан (${player.samplesCount})`
                          : 'Ожидает запись'}
                      </small>
                    </span>
                  </button>
                ))}
              </div>
            </GlassCard>
          </div>

          <div className="setup-actions">
            <Button
              variant="secondary"
              size="lg"
              onClick={() =>
                navigate('/setup/face-registration', {
                  state: {
                    playerCount: playerCount,
                  },
                })
              }
            >
              Назад
            </Button>
            <Button
              size="lg"
              disabled={!allRegistered}
              onClick={() =>
                navigate('/setup/table-detection', {
                  state: {
                    playerCount: playerCount,
                    players: players.map((player) => ({
                      id: player.id,
                      name: player.name,
                    })),
                  },
                })
              }
            >
              {allRegistered ? 'Продолжить' : 'Запишите все голоса'}
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
