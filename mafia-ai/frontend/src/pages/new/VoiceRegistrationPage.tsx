import { useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { GlassCard } from '../../components/ui/GlassCard';
import { Button } from '../../components/ui/Button';

type Player = {
  id: number;
  name: string;
  registered: boolean;
};

export function VoiceRegistrationPage() {
  const navigate = useNavigate();
  const location = useLocation();
  const playerCount = location.state?.playerCount || 10;
  const incomingPlayers = location.state?.players || [];

  const [players, setPlayers] = useState<Player[]>(() =>
    incomingPlayers.length > 0
      ? incomingPlayers.map((p: Player) => ({ ...p, registered: false }))
      : Array.from({ length: playerCount }, (_, i) => ({
          id: i + 1,
          name: `Игрок ${i + 1}`,
          registered: false,
        }))
  );
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isRecording, setIsRecording] = useState(false);
  const [recordingProgress, setRecordingProgress] = useState(0);

  const currentPlayer = players[currentIndex];
  const allRegistered = players.every(p => p.registered);

  const handleStartRecording = () => {
    setIsRecording(true);
    setRecordingProgress(0);

    const interval = setInterval(() => {
      setRecordingProgress((prev) => {
        if (prev >= 100) {
          clearInterval(interval);
          setPlayers(prev => prev.map((p, i) =>
            i === currentIndex ? { ...p, registered: true } : p
          ));
          setIsRecording(false);
          setRecordingProgress(0);

          if (currentIndex < players.length - 1) {
            setTimeout(() => setCurrentIndex(currentIndex + 1), 300);
          }
          return 100;
        }
        return prev + 3.33; // 3 seconds recording
      });
    }, 100);
  };

  const handleSkip = () => {
    if (currentIndex < players.length - 1) {
      setCurrentIndex(currentIndex + 1);
      setRecordingProgress(0);
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
      <div style={{ width: '100%', maxWidth: '800px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem', fontSize: '0.875rem' }}>
          <span style={{ color: '#94a3b8' }}>Регистрация голосов</span>
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

      {/* Main */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))',
        gap: '2rem',
        maxWidth: '900px',
        width: '100%',
      }}>
        {/* Recorder */}
        <GlassCard style={{ padding: '2rem' }}>
          <h2 style={{ fontSize: '1.5rem', fontWeight: 600, marginBottom: '1.5rem' }}>
            {currentPlayer.name}
          </h2>

          <p style={{
            color: '#94a3b8',
            marginBottom: '1.5rem',
            fontSize: '0.9375rem',
          }}>
            Произнесите: "Я {currentPlayer.name}, игрок номер {currentPlayer.id}"
          </p>

          {/* Microphone area */}
          <div style={{
            width: '100%',
            aspectRatio: '1',
            background: '#252938',
            borderRadius: '12px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            marginBottom: '1.5rem',
            position: 'relative',
          }}>
            {/* Recording indicator */}
            {isRecording && (
              <div style={{
                position: 'absolute',
                inset: 0,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                flexDirection: 'column',
                gap: '1rem',
                background: 'rgba(79,70,229,0.1)',
                borderRadius: '12px',
              }}>
                <div style={{
                  width: '80px',
                  height: '80px',
                  borderRadius: '50%',
                  background: '#4f46e5',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontSize: '2.5rem',
                  animation: 'pulse 1.5s ease-in-out infinite',
                }}>
                  <div style={{
                    width: '20px',
                    height: '20px',
                    borderRadius: '50%',
                    background: '#fff',
                  }} />
                </div>
                <div style={{ color: '#e2e8f0', fontSize: '0.875rem' }}>
                  Запись... {Math.ceil((100 - recordingProgress) / 33)}с
                </div>
              </div>
            )}
            {!isRecording && currentPlayer.registered && (
              <div style={{ fontSize: '4rem', color: '#10b981' }}>✓</div>
            )}
            {!isRecording && !currentPlayer.registered && (
              <div style={{
                width: '80px',
                height: '80px',
                borderRadius: '50%',
                background: 'rgba(255,255,255,0.08)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}>
                <div style={{
                  width: '40px',
                  height: '40px',
                  borderRadius: '50%',
                  background: 'rgba(255,255,255,0.15)',
                }} />
              </div>
            )}
          </div>

          {!currentPlayer.registered && (
            <div style={{ display: 'flex', gap: '1rem' }}>
              <Button variant="secondary" onClick={handleSkip} style={{ flex: 1 }} disabled={isRecording}>
                Пропустить
              </Button>
              <Button onClick={handleStartRecording} style={{ flex: 1 }} disabled={isRecording} loading={isRecording}>
                Записать
              </Button>
            </div>
          )}
        </GlassCard>

        {/* List */}
        <GlassCard style={{ padding: '2rem' }}>
          <h3 style={{ fontSize: '1.25rem', fontWeight: 600, marginBottom: '1rem' }}>Игроки</h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', maxHeight: '400px', overflowY: 'auto' }}>
            {players.map((p, i) => (
              <div
                key={p.id}
                onClick={() => !p.registered && setCurrentIndex(i)}
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
                }}>
                  {p.registered ? '✓' : p.id}
                </div>
                <div style={{ flex: 1 }}>
                  <div style={{ fontWeight: 500, fontSize: '0.9375rem' }}>
                    {p.name}
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

      {/* Actions */}
      <div style={{ display: 'flex', gap: '1rem' }}>
        <Button variant="secondary" size="lg" onClick={() => navigate('/setup/face-registration', { state: { playerCount } })}>
          Назад
        </Button>
        <Button
          size="lg"
          disabled={!allRegistered}
          onClick={() => navigate('/setup/table-detection', { state: { playerCount, players } })}
          style={{ minWidth: '180px' }}
        >
          {allRegistered ? 'Продолжить' : 'Завершите регистрацию'}
        </Button>
      </div>
    </div>
  );
}
