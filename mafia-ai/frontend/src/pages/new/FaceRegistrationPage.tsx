import { useState, useEffect, useRef } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { GlassCard } from '../../components/ui/GlassCard';
import { Button } from '../../components/ui/Button';
import { Input } from '../../components/ui/Input';
import { FaceIDScanner } from '../../components/FaceIDScanner';
import * as api from '../../services/api';

type Player = {
  id: number;
  name: string;
  registered: boolean;
};

type ScanState = 'idle' | 'scanning' | 'success' | 'error';

export function FaceRegistrationPage() {
  const navigate = useNavigate();
  const location = useLocation();
  const playerCount = location.state?.playerCount || 10;

  const [players, setPlayers] = useState<Player[]>(() =>
    Array.from({ length: playerCount }, (_, i) => ({
      id: i + 1,
      name: '',
      registered: false,
    }))
  );
  const [currentIndex, setCurrentIndex] = useState(0);
  const [scanState, setScanState] = useState<ScanState>('idle');
  const [scanProgress, setScanProgress] = useState(0);
  const [videoRunning, setVideoRunning] = useState(false);
  const [error, setError] = useState('');
  const [hint, setHint] = useState('Смотрите прямо в камеру');
  const [registeredPlayers, setRegisteredPlayers] = useState<api.Player[]>([]);
  const [showPlayerList, setShowPlayerList] = useState(false);

  const scanIntervalRef = useRef<NodeJS.Timeout>();
  const currentPlayer = players[currentIndex];
  const allRegistered = players.every(p => p.registered);

  // Start video on mount (no cleanup - keep video running)
  useEffect(() => {
    startVideoStream();
    loadRegisteredPlayers();
    // Don't stop video on unmount - it should keep running
  }, []);

  const loadRegisteredPlayers = async () => {
    try {
      const response = await api.listPlayers();
      setRegisteredPlayers(response.players || []);
    } catch (err) {
      console.error('Failed to load players:', err);
    }
  };

  const startVideoStream = async () => {
    try {
      console.log('[Video] Starting video stream...');
      const response = await api.startVideo();
      console.log('[Video] Start response:', response);
      setVideoRunning(true);
      console.log('[Video] Video stream URL:', api.getVideoStreamUrl());
    } catch (err) {
      console.error('[Video] Failed to start video:', err);
      setError('Не удалось запустить видео');
    }
  };

  const stopVideoStream = async () => {
    try {
      await api.stopVideo();
      await api.enrollCancel();
    } catch (err) {
      console.error('Failed to stop video:', err);
    }
  };

  const handleNameChange = (value: string) => {
    setPlayers(prev => prev.map((p, i) =>
      i === currentIndex ? { ...p, name: value } : p
    ));
    setError('');
  };

  const handleStartScan = async () => {
    const player = players[currentIndex];
    if (!player.name.trim()) {
      setError('Введите имя игрока');
      return;
    }

    try {
      setScanState('scanning');
      setScanProgress(0);
      setError('');

      // Start enrollment session
      const startRes = await api.enrollStart(player.name, 12);
      if (!startRes.ok) {
        throw new Error(startRes.error || 'Failed to start enrollment');
      }

      // Start collecting samples
      scanIntervalRef.current = setInterval(async () => {
        try {
          const snapRes = await api.enrollSnap();
          if (snapRes.ok) {
            const statusRes = await api.enrollStatus();
            if (statusRes.ok && statusRes.progress !== undefined) {
              setScanProgress(statusRes.progress * 100);

              // Update hint from backend
              if (statusRes.hint) {
                setHint(statusRes.hint);
              }

              // Check if we have enough samples
              if (statusRes.count && statusRes.target && statusRes.count >= statusRes.target) {
                clearInterval(scanIntervalRef.current!);
                await finishEnrollment(player.name);
              }
            }
          }
        } catch (err) {
          console.error('Snap error:', err);
        }
      }, 200); // Snap every 200ms

    } catch (err: any) {
      console.error('Scan error:', err);
      setError(err.message || 'Ошибка сканирования');
      setScanState('error');
      if (scanIntervalRef.current) {
        clearInterval(scanIntervalRef.current);
      }
    }
  };

  const finishEnrollment = async (name: string) => {
    try {
      const finishRes = await api.enrollFinish(name);
      if (finishRes.ok) {
        setScanState('success');
        setScanProgress(100);

        // Mark player as registered
        setPlayers(prev => prev.map((p, i) =>
          i === currentIndex ? { ...p, registered: true } : p
        ));

        // Auto-advance to next player after 1 second
        setTimeout(() => {
          if (currentIndex < players.length - 1) {
            setCurrentIndex(currentIndex + 1);
            setScanState('idle');
            setScanProgress(0);
          } else {
            // All done
            setScanState('idle');
          }
        }, 1000);
      } else {
        throw new Error(finishRes.error || 'Failed to finish enrollment');
      }
    } catch (err: any) {
      console.error('Finish error:', err);
      setError(err.message || 'Ошибка завершения');
      setScanState('error');
    }
  };

  const handleCancelScan = () => {
    if (scanIntervalRef.current) {
      clearInterval(scanIntervalRef.current);
    }
    api.enrollCancel();
    setScanState('idle');
    setScanProgress(0);
  };

  const handlePlayerClick = (index: number) => {
    if (!players[index].registered && scanState === 'idle') {
      setCurrentIndex(index);
    }
  };

  const handleSelectExistingPlayer = (registeredPlayer: api.Player) => {
    setPlayers(prev => prev.map((p, i) =>
      i === currentIndex
        ? { ...p, name: registeredPlayer.name, registered: true }
        : p
    ));
    setShowPlayerList(false);
    setError('');
  };

  const handleDeletePlayer = async (playerId: number) => {
    try {
      await api.deletePlayer(playerId);
      await loadRegisteredPlayers();
    } catch (err) {
      console.error('Failed to delete player:', err);
    }
  };

  const handleResetDatabase = async () => {
    if (confirm('Вы уверены что хотите удалить всех зарегистрированных игроков?')) {
      try {
        await api.resetPlayers();
        await loadRegisteredPlayers();
      } catch (err) {
        console.error('Failed to reset players:', err);
      }
    }
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
      {/* Progress */}
      <div style={{ width: '100%', maxWidth: '900px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem', fontSize: '0.875rem' }}>
          <span style={{ color: '#94a3b8' }}>Регистрация лиц</span>
          <span style={{ color: '#e2e8f0', fontWeight: 600 }}>
            {players.filter(p => p.registered).length} / {playerCount}
          </span>
        </div>
        <div style={{ width: '100%', height: '6px', background: 'rgba(255,255,255,0.08)', borderRadius: '3px' }}>
          <div style={{
            height: '100%',
            background: '#4f46e5',
            borderRadius: '3px',
            width: `${(players.filter(p => p.registered).length / playerCount) * 100}%`,
            transition: 'width 0.3s ease',
          }} />
        </div>
      </div>

      {/* Main content */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(340px, 1fr))',
        gap: '2rem',
        maxWidth: '900px',
        width: '100%',
      }}>
        {/* Scanner */}
        <GlassCard style={{ padding: '2rem' }}>
          <h2 style={{ fontSize: '1.5rem', fontWeight: 600, marginBottom: '1.5rem' }}>
            Игрок {currentPlayer.id}
          </h2>

          {/* Name input */}
          {!currentPlayer.registered && (
            <Input
              label="Имя игрока"
              placeholder="Введите имя"
              value={currentPlayer.name}
              onChange={(e) => handleNameChange(e.target.value)}
              error={error}
              style={{ marginBottom: '1.5rem' }}
              disabled={scanState === 'scanning'}
            />
          )}

          {/* Face scanner */}
          <FaceIDScanner
            state={currentPlayer.registered ? 'success' : scanState}
            progress={scanProgress}
            videoUrl={videoRunning ? api.getVideoStreamUrl() : undefined}
          />

          {/* Hint */}
          {scanState === 'scanning' && (
            <div style={{
              marginTop: '1rem',
              padding: '0.75rem',
              background: 'rgba(79, 70, 229, 0.1)',
              border: '1px solid rgba(79, 70, 229, 0.3)',
              borderRadius: '0.5rem',
              textAlign: 'center',
              color: '#a5b4fc',
              fontSize: '0.875rem',
              fontWeight: 500,
            }}>
              {hint}
            </div>
          )}

          {/* Controls */}
          <div style={{ marginTop: '1.5rem', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            {scanState === 'idle' && !currentPlayer.registered && (
              <>
                <Button
                  onClick={handleStartScan}
                  fullWidth
                  disabled={!currentPlayer.name.trim()}
                >
                  Начать сканирование
                </Button>
                <Button
                  variant="secondary"
                  onClick={() => setShowPlayerList(true)}
                  fullWidth
                >
                  Выбрать из базы ({registeredPlayers.length})
                </Button>
              </>
            )}
            {scanState === 'scanning' && (
              <Button
                variant="secondary"
                onClick={handleCancelScan}
                fullWidth
              >
                Отменить
              </Button>
            )}
            {currentPlayer.registered && (
              <div style={{
                padding: '1rem',
                background: 'rgba(16, 185, 129, 0.1)',
                border: '1px solid rgba(16, 185, 129, 0.3)',
                borderRadius: '0.5rem',
                textAlign: 'center',
                color: '#10b981',
                fontWeight: 600,
                width: '100%',
              }}>
                {currentPlayer.name} зарегистрирован!
              </div>
            )}
          </div>

          {/* Database reset button */}
          {registeredPlayers.length > 0 && (
            <div style={{ marginTop: '1rem' }}>
              <Button
                variant="secondary"
                onClick={handleResetDatabase}
                fullWidth
                style={{
                  background: 'rgba(239, 68, 68, 0.1)',
                  borderColor: 'rgba(239, 68, 68, 0.3)',
                  color: '#f87171',
                }}
              >
                Очистить базу данных
              </Button>
            </div>
          )}
        </GlassCard>

        {/* Players list */}
        <GlassCard style={{ padding: '2rem' }}>
          <h3 style={{ fontSize: '1.25rem', fontWeight: 600, marginBottom: '1rem' }}>Игроки</h3>
          <div style={{
            display: 'flex',
            flexDirection: 'column',
            gap: '0.75rem',
            maxHeight: '500px',
            overflowY: 'auto',
          }}>
            {players.map((p, i) => (
              <div
                key={p.id}
                onClick={() => handlePlayerClick(i)}
                style={{
                  padding: '0.75rem 1rem',
                  background: i === currentIndex ? 'rgba(79,70,229,0.1)' : 'rgba(255,255,255,0.03)',
                  border: i === currentIndex ? '1px solid rgba(79,70,229,0.3)' : '1px solid rgba(255,255,255,0.05)',
                  borderRadius: '8px',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.75rem',
                  cursor: p.registered ? 'default' : 'pointer',
                  transition: 'all 0.15s ease',
                }}
              >
                <div style={{
                  width: '32px',
                  height: '32px',
                  borderRadius: '50%',
                  background: p.registered ? '#10b981' : 'rgba(255,255,255,0.08)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontSize: '0.875rem',
                  fontWeight: 600,
                  flexShrink: 0,
                }}>
                  {p.registered ? '✓' : p.id}
                </div>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{
                    fontWeight: 500,
                    fontSize: '0.9375rem',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    whiteSpace: 'nowrap',
                  }}>
                    {p.name || `Игрок ${p.id}`}
                  </div>
                  <div style={{ fontSize: '0.75rem', color: '#94a3b8' }}>
                    {p.registered ? 'Зарегистрирован' : i === currentIndex ? 'Текущий' : 'Ожидание'}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </GlassCard>
      </div>

      {/* Modal: Player list */}
      {showPlayerList && (
        <div style={{
          position: 'fixed',
          inset: 0,
          background: 'rgba(0, 0, 0, 0.7)',
          backdropFilter: 'blur(4px)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 1000,
          padding: '2rem',
        }} onClick={() => setShowPlayerList(false)}>
          <GlassCard style={{
            padding: '2rem',
            maxWidth: '500px',
            width: '100%',
            maxHeight: '80vh',
            overflowY: 'auto',
          }} onClick={(e) => e.stopPropagation()}>
            <h2 style={{ fontSize: '1.5rem', fontWeight: 600, marginBottom: '1.5rem' }}>
              Зарегистрированные игроки
            </h2>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
              {registeredPlayers.map((player) => (
                <div
                  key={player.id}
                  style={{
                    padding: '1rem',
                    background: 'rgba(255, 255, 255, 0.03)',
                    border: '1px solid rgba(255, 255, 255, 0.1)',
                    borderRadius: '8px',
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                  }}
                >
                  <span style={{ color: '#e2e8f0', fontWeight: 500 }}>{player.name}</span>
                  <div style={{ display: 'flex', gap: '0.5rem' }}>
                    <Button
                      onClick={() => handleSelectExistingPlayer(player)}
                      style={{ padding: '0.5rem 1rem' }}
                    >
                      Выбрать
                    </Button>
                    <Button
                      variant="secondary"
                      onClick={() => handleDeletePlayer(player.id)}
                      style={{
                        padding: '0.5rem 1rem',
                        background: 'rgba(239, 68, 68, 0.1)',
                        borderColor: 'rgba(239, 68, 68, 0.3)',
                        color: '#f87171',
                      }}
                    >
                      Удалить
                    </Button>
                  </div>
                </div>
              ))}
            </div>

            <Button
              variant="secondary"
              onClick={() => setShowPlayerList(false)}
              fullWidth
              style={{ marginTop: '1.5rem' }}
            >
              Закрыть
            </Button>
          </GlassCard>
        </div>
      )}

      {/* Actions */}
      <div style={{ display: 'flex', gap: '1rem' }}>
        <Button
          variant="secondary"
          size="lg"
          onClick={() => {
            stopVideoStream();
            navigate('/setup/players');
          }}
        >
          Назад
        </Button>
        <Button
          size="lg"
          disabled={!allRegistered}
          onClick={() => {
            stopVideoStream();
            navigate('/setup/voice-registration', { state: { playerCount, players } });
          }}
          style={{ minWidth: '180px' }}
        >
          {allRegistered ? 'Продолжить' : 'Завершите регистрацию'}
        </Button>
      </div>
    </div>
  );
}
