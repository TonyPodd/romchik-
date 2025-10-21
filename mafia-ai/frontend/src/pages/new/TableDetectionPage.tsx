import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { GlassCard } from '../../components/ui/GlassCard';
import { Button } from '../../components/ui/Button';

type DetectionState = 'idle' | 'detecting' | 'detected' | 'error';

export function TableDetectionPage() {
  const navigate = useNavigate();
  const [detectionState, setDetectionState] = useState<DetectionState>('idle');
  const [progress, setProgress] = useState(0);

  const handleStartDetection = () => {
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
  };

  const handleRetry = () => {
    setDetectionState('idle');
    setProgress(0);
  };

  const handleContinue = () => {
    navigate('/');
  };

  return (
    <div style={{
      minHeight: '100vh',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      padding: '2rem',
      gap: '2rem',
    }}>
      <div style={{ maxWidth: '900px', width: '100%' }}>
        <h1 style={{
          fontSize: 'clamp(2rem, 5vw, 3rem)',
          fontWeight: 700,
          textAlign: 'center',
          marginBottom: '0.5rem',
        }}>
          Определение стола
        </h1>
        <p style={{
          color: '#94a3b8',
          textAlign: 'center',
          marginBottom: '3rem',
          fontSize: '1.125rem',
        }}>
          Направьте камеру на игровой стол для автоматического определения границ
        </p>

        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(350px, 1fr))',
          gap: '2rem',
          alignItems: 'start',
        }}>
          {/* Camera preview */}
          <GlassCard style={{
            padding: '2rem',
            display: 'flex',
            flexDirection: 'column',
            gap: '1.5rem',
          }}>
            <h2 style={{ fontSize: '1.25rem', fontWeight: 600 }}>
              Камера
            </h2>

            <div style={{
              position: 'relative',
              width: '100%',
              aspectRatio: '4/3',
              background: '#252938',
              borderRadius: '1rem',
              overflow: 'hidden',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}>
              {/* Camera placeholder */}
              <div style={{
                fontSize: '4rem',
                opacity: 0.3,
              }}>
                📹
              </div>

              {/* Scanning overlay */}
              {detectionState === 'detecting' && (
                <div style={{
                  position: 'absolute',
                  inset: 0,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  background: 'rgba(79,70,229,0.1)',
                }}>
                  <div style={{
                    width: '60px',
                    height: '60px',
                    border: '3px solid rgba(79,70,229,0.3)',
                    borderTop: '3px solid #4f46e5',
                    borderRadius: '50%',
                    animation: 'spin 1s linear infinite',
                  }} />
                </div>
              )}

              {/* Success overlay */}
              {detectionState === 'detected' && (
                <div style={{
                  position: 'absolute',
                  inset: 0,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  background: 'rgba(16, 185, 129, 0.2)',
                }}>
                  <div style={{ fontSize: '5rem', color: '#10b981' }}>✓</div>
                </div>
              )}
            </div>

            {/* Progress bar */}
            {detectionState === 'detecting' && (
              <div>
                <div style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  marginBottom: '0.5rem',
                }}>
                  <span style={{ color: '#94a3b8', fontSize: '0.875rem' }}>
                    Анализ изображения...
                  </span>
                  <span style={{ color: '#e2e8f0', fontSize: '0.875rem', fontWeight: 600 }}>
                    {Math.round(progress)}%
                  </span>
                </div>
                <div style={{
                  width: '100%',
                  height: '6px',
                  background: 'rgba(255, 255, 255, 0.1)',
                  borderRadius: '3px',
                  overflow: 'hidden',
                }}>
                  <div style={{
                    height: '100%',
                    background: '#4f46e5',
                    borderRadius: '3px',
                    width: `${progress}%`,
                    transition: 'width 0.1s ease',
                  }} />
                </div>
              </div>
            )}

            {/* Controls */}
            <div style={{ display: 'flex', gap: '1rem' }}>
              {detectionState === 'idle' && (
                <Button
                  variant="primary"
                  onClick={handleStartDetection}
                  fullWidth
                >
                  Начать определение
                </Button>
              )}
              {detectionState === 'detected' && (
                <Button
                  variant="secondary"
                  onClick={handleRetry}
                  fullWidth
                >
                  Повторить
                </Button>
              )}
            </div>
          </GlassCard>

          {/* Instructions */}
          <GlassCard style={{ padding: '2rem' }}>
            <h2 style={{
              fontSize: '1.25rem',
              fontWeight: 600,
              marginBottom: '1.5rem',
            }}>
              Инструкции
            </h2>

            <div style={{
              display: 'flex',
              flexDirection: 'column',
              gap: '1.5rem',
            }}>
              {[
                { title: 'Настройте камеру', text: 'Убедитесь что весь стол виден в кадре' },
                { title: 'Освещение', text: 'Обеспечьте хорошее освещение игровой зоны' },
                { title: 'Форма стола', text: 'Система распознает круглые и прямоугольные столы' },
                { title: 'Автоматическое определение', text: 'AI автоматически найдет границы стола для голосования' },
              ].map((item, index) => (
                <div
                  key={index}
                  style={{
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '0.5rem',
                  }}
                >
                  <div style={{ fontWeight: 600, fontSize: '0.9375rem' }}>
                    {index + 1}. {item.title}
                  </div>
                  <div style={{
                    color: '#94a3b8',
                    fontSize: '0.875rem',
                    lineHeight: 1.5,
                    paddingLeft: '1.25rem',
                  }}>
                    {item.text}
                  </div>
                </div>
              ))}
            </div>

            {detectionState === 'detected' && (
              <div style={{
                marginTop: '2rem',
                padding: '1rem',
                background: 'rgba(16, 185, 129, 0.1)',
                border: '1px solid rgba(16, 185, 129, 0.3)',
                borderRadius: '0.75rem',
                textAlign: 'center',
                color: '#10b981',
                fontWeight: 600,
              }}>
                Стол успешно определен!
              </div>
            )}
          </GlassCard>
        </div>

        {/* Navigation */}
        <div style={{
          display: 'flex',
          gap: '1rem',
          justifyContent: 'center',
          marginTop: '2rem',
        }}>
          <Button
            variant="secondary"
            size="lg"
            onClick={() => navigate('/setup/voice-registration')}
          >
            Назад
          </Button>
          <Button
            variant="success"
            size="lg"
            disabled={detectionState !== 'detected'}
            onClick={handleContinue}
            style={{ minWidth: '200px' }}
          >
            {detectionState === 'detected' ? 'Завершить настройку' : 'Определите стол'}
          </Button>
        </div>
      </div>
    </div>
  );
}
