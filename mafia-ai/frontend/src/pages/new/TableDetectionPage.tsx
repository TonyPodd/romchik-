import { useEffect, useMemo, useRef, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { SetupStageHeader } from '../../components/SetupStageHeader';
import { Button } from '../../components/ui/Button';
import { GlassCard } from '../../components/ui/GlassCard';
import * as api from '../../services/api';
import './TableDetectionPage.css';

type DetectionState = 'idle' | 'detecting' | 'detected' | 'manual';
type Point = { x: number; y: number };

type ProcessRouteState = {
  playerCount?: number;
  players?: Array<{ id?: number; name?: string }>;
};

function toPoints(poly: api.TablePoint[] | null | undefined): Point[] | null {
  if (!Array.isArray(poly) || poly.length < 3) {
    return null;
  }
  return poly.map(([x, y]) => ({ x: Number(x), y: Number(y) }));
}

export function TableDetectionPage() {
  const navigate = useNavigate();
  const location = useLocation();
  const processState = ((location.state as ProcessRouteState | null) || {}) satisfies ProcessRouteState;

  const viewportRef = useRef<HTMLDivElement | null>(null);
  const scanIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const [videoRunning, setVideoRunning] = useState(false);
  const [detectionState, setDetectionState] = useState<DetectionState>('idle');
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState(0);
  const [poly, setPoly] = useState<Point[] | null>(null);
  const [drawing, setDrawing] = useState(false);
  const [draft, setDraft] = useState<Point[]>([]);
  const [error, setError] = useState('');
  const [message, setMessage] = useState('');

  const hasDetectedPolygon = Boolean(poly && poly.length >= 3);

  const activePolygon = useMemo(() => {
    if (drawing) {
      return draft;
    }
    return poly || [];
  }, [drawing, draft, poly]);

  useEffect(() => {
    let cancelled = false;

    async function bootstrap() {
      try {
        setError('');
        const status = await api.getVideoStatus();
        if (!status.running) {
          const start = await api.startVideo();
          if (!start.ok) {
            throw new Error(start.error || 'Не удалось запустить видео');
          }
        }
        if (cancelled) {
          return;
        }
        setVideoRunning(true);

        await api.tableBeginCalibration().catch(() => undefined);
        const tableStatus = await api.getTableStatus();
        if (cancelled) {
          return;
        }

        const existing = toPoints(tableStatus.poly_norm);
        if (existing) {
          setPoly(existing);
          setDetectionState('detected');
          setProgress(100);
          setMessage('Контур стола найден в текущей сессии.');
        } else {
          setDetectionState('idle');
          setProgress(0);
          setMessage('');
        }
      } catch (bootstrapError: any) {
        if (!cancelled) {
          setError(bootstrapError?.message || 'Не удалось подготовить калибровку стола');
        }
      }
    }

    void bootstrap();

    return () => {
      cancelled = true;
      stopProgressAnimation();
      void api.tableEndCalibration().catch(() => undefined);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function stopProgressAnimation() {
    if (scanIntervalRef.current) {
      clearInterval(scanIntervalRef.current);
      scanIntervalRef.current = null;
    }
  }

  function startProgressAnimation() {
    stopProgressAnimation();
    scanIntervalRef.current = setInterval(() => {
      setProgress((prev) => (prev >= 92 ? 92 : prev + 2));
    }, 120);
  }

  function normalizePoint(clientX: number, clientY: number): Point {
    const rect = viewportRef.current?.getBoundingClientRect();
    if (!rect || rect.width <= 0 || rect.height <= 0) {
      return { x: 0, y: 0 };
    }
    const x = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
    const y = Math.max(0, Math.min(1, (clientY - rect.top) / rect.height));
    return { x, y };
  }

  async function handleAutoDetect() {
    if (busy) {
      return;
    }

    setBusy(true);
    setError('');
    setMessage('');
    setDrawing(false);
    setDraft([]);
    setDetectionState('detecting');
    setProgress(8);
    startProgressAnimation();

    try {
      const response = await api.tableAutoDetect();
      const detected = toPoints(response.poly_norm);
      if (!response.ok || !detected) {
        throw new Error(response.error || 'Автоопределение не нашло стол. Используйте ручной режим.');
      }

      setPoly(detected);
      setDetectionState('detected');
      setProgress(100);
      setMessage('Контур стола определен автоматически.');
    } catch (detectError: any) {
      setDetectionState('idle');
      setProgress(0);
      setError(detectError?.message || 'Ошибка автоопределения стола');
    } finally {
      stopProgressAnimation();
      setBusy(false);
    }
  }

  function handleManualMode() {
    setError('');
    setMessage('Клик добавляет точку. Двойной клик завершает, правая кнопка удаляет последнюю точку.');
    setDetectionState('manual');
    setDrawing(true);
    setDraft(poly ? poly.map((point) => ({ ...point })) : []);
    setProgress(0);
  }

  function handleOverlayClick(event: React.MouseEvent<SVGSVGElement>) {
    if (!drawing) {
      return;
    }
    const point = normalizePoint(event.clientX, event.clientY);
    setDraft((prev) => [...prev, point]);
  }

  function handleOverlayDoubleClick(event: React.MouseEvent<SVGSVGElement>) {
    if (!drawing) {
      return;
    }
    event.preventDefault();
    if (draft.length >= 3) {
      setDrawing(false);
      setMessage('Контур готов. Нажмите "Сохранить контур".');
    }
  }

  function handleOverlayContextMenu(event: React.MouseEvent<SVGSVGElement>) {
    if (!drawing) {
      return;
    }
    event.preventDefault();
    setDraft((prev) => prev.slice(0, -1));
  }

  async function handleSaveManual() {
    if (busy || draft.length < 3) {
      return;
    }

    setBusy(true);
    setError('');
    setDetectionState('detecting');
    setProgress(10);
    startProgressAnimation();

    try {
      const payload: api.TablePoint[] = draft.map((point) => [
        Number(point.x.toFixed(5)),
        Number(point.y.toFixed(5)),
      ]);
      const response = await api.tableSetRoi(payload);
      const saved = toPoints(response.poly_norm);
      if (!response.ok || !saved) {
        throw new Error(response.error || 'Не удалось сохранить контур стола');
      }

      setPoly(saved);
      setDraft([]);
      setDrawing(false);
      setDetectionState('detected');
      setProgress(100);
      setMessage('Контур стола сохранен.');
    } catch (saveError: any) {
      setDetectionState('manual');
      setError(saveError?.message || 'Ошибка сохранения контура');
    } finally {
      stopProgressAnimation();
      setBusy(false);
    }
  }

  async function handleClear() {
    if (busy) {
      return;
    }
    setBusy(true);
    setError('');
    try {
      const response = await api.tableClearRoi();
      if (!response.ok) {
        throw new Error(response.error || 'Не удалось очистить контур');
      }
      setPoly(null);
      setDraft([]);
      setDrawing(false);
      setDetectionState('idle');
      setProgress(0);
      setMessage('Контур очищен.');
    } catch (clearError: any) {
      setError(clearError?.message || 'Ошибка очистки контура');
    } finally {
      setBusy(false);
    }
  }

  const polygonPoints = activePolygon.map((point) => `${point.x * 100},${point.y * 100}`).join(' ');

  return (
    <div className="setup-shell">
      <div className="setup-container setup-container--with-stage">
        <GlassCard className="setup-stage-shell">
          <SetupStageHeader
            current="table"
            title="Определение стола"
            subtitle="Проверьте контур игровой зоны: автоматически или вручную перед стартом партии."
          />
        </GlassCard>

        <div className="setup-wizard">
          <div className="setup-grid setup-grid--table">
            <GlassCard className="table-page__camera">
              <div className="table-page__header">
                <h2 className="table-page__title">Камера и контур</h2>
                <span
                  className={`status-tag ${
                    hasDetectedPolygon ? 'status-tag--success' : 'status-tag--warn'
                  }`}
                >
                  {hasDetectedPolygon ? 'Контур готов' : 'Контур не задан'}
                </span>
              </div>

              <div className="table-page__viewport table-stage" ref={viewportRef}>
                {videoRunning ? (
                  <img
                    className="table-page__stream"
                    src={api.getVideoStreamUrl()}
                    alt="Поток камеры"
                    draggable={false}
                  />
                ) : (
                  <div className="table-page__empty">Ожидание видеопотока...</div>
                )}

                <svg
                  className="table-page__overlay"
                  viewBox="0 0 100 100"
                  preserveAspectRatio="none"
                  onClick={handleOverlayClick}
                  onDoubleClick={handleOverlayDoubleClick}
                  onContextMenu={handleOverlayContextMenu}
                >
                  {activePolygon.length >= 2 && (
                    <polyline
                      points={polygonPoints}
                      fill={drawing && activePolygon.length >= 3 ? 'rgba(126, 245, 182, 0.10)' : 'rgba(56, 189, 248, 0.14)'}
                      stroke={drawing ? 'rgba(126, 245, 182, 0.88)' : 'rgba(56, 189, 248, 0.9)'}
                      strokeWidth={drawing ? 0.55 : 0.45}
                      strokeDasharray={drawing ? '1.1 1.1' : undefined}
                    />
                  )}
                  {activePolygon.map((point, index) => (
                    <circle
                      key={`${point.x}-${point.y}-${index}`}
                      cx={point.x * 100}
                      cy={point.y * 100}
                      r={drawing ? 0.9 : 0.65}
                      fill={drawing ? 'rgba(126, 245, 182, 0.95)' : 'rgba(185, 240, 255, 0.95)'}
                    />
                  ))}
                </svg>

                {detectionState === 'detecting' && <div className="table-page__scan-line" />}
              </div>

              <div className="setup-progress">
                <div className="setup-progress__row">
                  <span>Состояние калибровки</span>
                  <span>{Math.round(progress)}%</span>
                </div>
                <progress className="setup-progress__native" value={progress} max={100} />
              </div>
            </GlassCard>

            <GlassCard className="table-page__panel">
              <h2 className="table-page__title">Инструменты</h2>

              <div className="table-page__actions">
                <Button onClick={() => void handleAutoDetect()} disabled={busy} fullWidth>
                  Автоопределение
                </Button>
                <Button variant="secondary" onClick={handleManualMode} disabled={busy} fullWidth>
                  Ручная разметка
                </Button>
                <Button
                  variant="secondary"
                  onClick={() => void handleSaveManual()}
                  disabled={busy || draft.length < 3}
                  fullWidth
                >
                  Сохранить контур
                </Button>
                <Button variant="secondary" onClick={() => void handleClear()} disabled={busy || !hasDetectedPolygon} fullWidth>
                  Очистить контур
                </Button>
              </div>

              <ol className="table-page__steps">
                <li>Камера должна видеть весь стол без сильной засветки.</li>
                <li>Если автоопределение ошиблось, перейдите в ручную разметку.</li>
                <li>После сохранения контура можно запускать игровой процесс.</li>
              </ol>

              {drawing && (
                <div className="status-tag status-tag--warn table-page__hint">
                  Ручной режим: {draft.length} точек
                </div>
              )}

              {message && <div className="table-page__message">{message}</div>}
              {error && <div className="table-page__error">{error}</div>}
            </GlassCard>
          </div>

          <div className="setup-actions">
            <Button variant="secondary" size="lg" onClick={() => navigate('/setup/voice', { state: processState })}>
              Назад
            </Button>
            <Button
              size="lg"
              disabled={!hasDetectedPolygon}
              onClick={() =>
                navigate('/game/process', {
                  state: processState,
                })
              }
            >
              {hasDetectedPolygon ? 'Перейти к процессу игры' : 'Сначала задайте контур стола'}
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
