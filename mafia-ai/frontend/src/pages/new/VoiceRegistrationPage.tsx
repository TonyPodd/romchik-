import { useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { SetupStageHeader } from '../../components/SetupStageHeader';
import { Button } from '../../components/ui/Button';
import { GlassCard } from '../../components/ui/GlassCard';
import './VoiceRegistrationPage.css';

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
      ? incomingPlayers.map((player: Player) => ({ ...player, registered: false }))
      : Array.from({ length: playerCount }, (_, index) => ({
          id: index + 1,
          name: `Игрок ${index + 1}`,
          registered: false,
        })),
  );
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isRecording, setIsRecording] = useState(false);
  const [recordingProgress, setRecordingProgress] = useState(0);

  const currentPlayer = players[currentIndex];
  const registeredCount = players.filter((player) => player.registered).length;
  const allRegistered = registeredCount === playerCount;

  function handleStartRecording() {
    setIsRecording(true);
    setRecordingProgress(0);

    const interval = setInterval(() => {
      setRecordingProgress((prev) => {
        if (prev >= 100) {
          clearInterval(interval);
          setPlayers((slots) =>
            slots.map((player, index) =>
              index === currentIndex ? { ...player, registered: true } : player,
            ),
          );
          setIsRecording(false);
          setRecordingProgress(0);

          if (currentIndex < players.length - 1) {
            setTimeout(() => setCurrentIndex(currentIndex + 1), 280);
          }
          return 100;
        }
        return prev + 3.33;
      });
    }, 100);
  }

  function handleSkip() {
    if (currentIndex < players.length - 1) {
      setCurrentIndex(currentIndex + 1);
      setRecordingProgress(0);
    }
  }

  return (
    <div className="setup-shell">
      <div className="setup-container setup-container--with-stage">
        <GlassCard className="setup-stage-shell">
          <SetupStageHeader
            current="voice"
            title="Регистрация голосов"
            subtitle="Сохраните короткий голосовой образец для каждого игрока."
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
                <h2 className="voice-page__title">{currentPlayer.name}</h2>
                <span className={`status-tag ${currentPlayer.registered ? 'status-tag--success' : 'status-tag--warn'}`}>
                  {currentPlayer.registered ? 'Готово' : 'В ожидании'}
                </span>
              </div>
              <p className="voice-page__prompt">
                Произнесите: "Я {currentPlayer.name}, игрок номер {currentPlayer.id}"
              </p>

              <div className="voice-stage">
                {isRecording && (
                  <div className="voice-page__recording">
                    <span className="voice-stage__pulse" />
                    <strong>Запись... {Math.ceil((100 - recordingProgress) / 33)}с</strong>
                  </div>
                )}
                {!isRecording && currentPlayer.registered && (
                  <div className="voice-page__status status-tag status-tag--success">Голос сохранен</div>
                )}
                {!isRecording && !currentPlayer.registered && (
                  <div className="voice-page__status">Готово к записи</div>
                )}
              </div>

              {!currentPlayer.registered && (
                <div className="voice-page__actions">
                  <Button variant="secondary" onClick={handleSkip} disabled={isRecording} fullWidth>
                    Пропустить
                  </Button>
                  <Button onClick={handleStartRecording} disabled={isRecording} loading={isRecording} fullWidth>
                    Записать
                  </Button>
                </div>
              )}
            </GlassCard>

            <GlassCard className="voice-page__list">
              <h2 className="voice-page__title">Игровые места</h2>
              <div className="voice-page__players">
                {players.map((player, index) => (
                  <button
                    type="button"
                    key={player.id}
                    className={`voice-page__player ${index === currentIndex ? 'is-active' : ''} ${player.registered ? 'is-filled' : ''}`.trim()}
                    onClick={() => !player.registered && setCurrentIndex(index)}
                  >
                    <span className="voice-page__player-id">{player.registered ? '✓' : player.id}</span>
                    <span className="voice-page__player-meta">
                      <strong>{player.name}</strong>
                      <small>{player.registered ? 'Зарегистрирован' : 'Ожидает запись'}</small>
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
              onClick={() => navigate('/setup/face-registration', { state: { playerCount } })}
            >
              Назад
            </Button>
            <Button
              size="lg"
              disabled={!allRegistered}
              onClick={() => navigate('/setup/table-detection', { state: { playerCount, players } })}
            >
              {allRegistered ? 'Продолжить' : 'Завершите регистрацию'}
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
