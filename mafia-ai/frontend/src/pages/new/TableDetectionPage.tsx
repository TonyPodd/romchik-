import { CSSProperties, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Button } from '../../components/ui/Button';
import { GlassCard } from '../../components/ui/GlassCard';
import './TableDetectionPage.css';

type DetectionState = 'idle' | 'detecting' | 'detected';

export function TableDetectionPage() {
  const navigate = useNavigate();
  const [detectionState, setDetectionState] = useState<DetectionState>('idle');
  const [progress, setProgress] = useState(0);

  function handleStartDetection() {
    setDetectionState('detecting');
    setProgress(0);

    const interval = setInterval(() => {
      setProgress((prev) => {
        if (prev >= 100) {
          clearInterval(interval);
          setDetectionState('detected');
          return 100;
        }
        return prev + 2;
      });
    }, 50);
  }

  function handleRetry() {
    setDetectionState('idle');
    setProgress(0);
  }

  return (
    <div className="setup-shell">
      <div className="setup-container">
        <div className="setup-hero">
          <h1 className="setup-title">Определение стола</h1>
          <p className="setup-subtitle">
            Система анализирует кадр и подбирает игровую область для этапов голосования.
          </p>
        </div>

        <div className="setup-grid setup-grid--two">
          <GlassCard className="table-page__camera">
            <h2 className="feature-card__title">Камера</h2>
            <div className="table-stage">
              <div className={`table-page__state table-page__state--${detectionState}`}>
                {detectionState === 'detected' ? '✓' : '⌁'}
              </div>
              {detectionState === 'detecting' && <div className="table-page__scan-line" />}
            </div>

            <div className="setup-progress">
              <div className="setup-progress__row">
                <span>Анализ изображения</span>
                <span>{Math.round(progress)}%</span>
              </div>
              <div className="setup-progress__bar">
                <div
                  className="setup-progress__fill"
                  style={{ '--progress-value': `${progress}%` } as CSSProperties}
                />
              </div>
            </div>

            <div className="table-page__actions">
              {detectionState === 'idle' && (
                <Button onClick={handleStartDetection} fullWidth>
                  Начать определение
                </Button>
              )}
              {detectionState === 'detecting' && (
                <Button variant="secondary" disabled fullWidth>
                  Идет обработка...
                </Button>
              )}
              {detectionState === 'detected' && (
                <Button variant="secondary" onClick={handleRetry} fullWidth>
                  Повторить
                </Button>
              )}
            </div>
          </GlassCard>

          <GlassCard className="table-page__info">
            <h2 className="feature-card__title">Проверка качества</h2>
            <ol className="table-page__steps">
              <li>Поставьте камеру так, чтобы стол занимал центр кадра.</li>
              <li>Избегайте сильной засветки и жестких теней на поверхности.</li>
              <li>После обнаружения проверьте корректность контура.</li>
            </ol>
            {detectionState === 'detected' && (
              <div className="status-tag status-tag--success table-page__ok">
                Стол определен и готов к следующему шагу.
              </div>
            )}
          </GlassCard>
        </div>

        <div className="setup-actions">
          <Button variant="secondary" size="lg" onClick={() => navigate('/setup/voice')}>
            Назад
          </Button>
          <Button
            size="lg"
            disabled={detectionState !== 'detected'}
            onClick={() => navigate('/')}
          >
            {detectionState === 'detected' ? 'Завершить настройку' : 'Определите стол'}
          </Button>
        </div>
      </div>
    </div>
  );
}
