import { useState, useRef, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { GlassCard } from '../../components/ui/GlassCard';
import { Button } from '../../components/ui/Button';
import { enrollVoice, listPlayers } from '../../services/api';

type Player = {
  id: number;
  name: string;
  registered: boolean;
  hasVoice?: boolean;
};

export function VoiceRegistrationPage() {
  const navigate = useNavigate();
  const location = useLocation();
  const playerCount = location.state?.playerCount || 10;
  const incomingPlayers = location.state?.players || [];

  const [players, setPlayers] = useState<Player[]>([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isRecording, setIsRecording] = useState(false);
  const [recordingProgress, setRecordingProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const progressIntervalRef = useRef<number | null>(null);

  const currentPlayer = players[currentIndex];
  const allRegistered = players.every(p => p.registered);

  // Загрузка игроков из БД при монтировании
  useEffect(() => {
    loadPlayers();
  }, []);

  const loadPlayers = async () => {
    try {
      const response = await listPlayers();
      if (response.players && response.players.length > 0) {
        // Используем реальных игроков из БД
        setPlayers(
          response.players.map((p: any) => ({
            id: p.id,
            name: p.name || `Игрок ${p.id}`,
            registered: false,
            hasVoice: !!p.voice_embedding,
          }))
        );
      } else {
        // Fallback на моковые данные
        setPlayers(
          Array.from({ length: playerCount }, (_, i) => ({
            id: i + 1,
            name: `Игрок ${i + 1}`,
            registered: false,
          }))
        );
      }
    } catch (err) {
      console.error('[Voice] Failed to load players:', err);
      setError('Не удалось загрузить список игроков');
      // Fallback на моковые данные
      setPlayers(
        Array.from({ length: playerCount }, (_, i) => ({
          id: i + 1,
          name: `Игрок ${i + 1}`,
          registered: false,
        }))
      );
    }
  };

  const handleStartRecording = async () => {
    try {
      setError(null);
      
      // Запрашиваем доступ к микрофону
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          channelCount: 1,
          sampleRate: 16000,
          echoCancellation: true,
          noiseSuppression: true,
        } 
      });

      // Создаем MediaRecorder
      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus'
      });
      
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      // Собираем аудио данные
      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      // Обработка окончания записи
      mediaRecorder.onstop = async () => {
        // Останавливаем все треки
        stream.getTracks().forEach(track => track.stop());
        
        // Создаем Blob из записанных данных
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        
        // Конвертируем в WAV (требуется для backend)
        await processAndUploadAudio(audioBlob);
      };

      // Запускаем запись
      mediaRecorder.start();
      setIsRecording(true);
      setRecordingProgress(0);

      // Прогресс бар (3 секунды записи)
      const duration = 3000; // 3 секунды
      const startTime = Date.now();
      
      progressIntervalRef.current = window.setInterval(() => {
        const elapsed = Date.now() - startTime;
        const progress = Math.min((elapsed / duration) * 100, 100);
        setRecordingProgress(progress);

        if (progress >= 100) {
          stopRecording();
        }
      }, 100);

    } catch (err) {
      console.error('[Voice] Error starting recording:', err);
      setError('Не удалось получить доступ к микрофону');
      setIsRecording(false);
    }
  };

  const stopRecording = () => {
    if (progressIntervalRef.current) {
      clearInterval(progressIntervalRef.current);
      progressIntervalRef.current = null;
    }

    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
    }

    setIsRecording(false);
  };

  const processAndUploadAudio = async (audioBlob: Blob) => {
    try {
      setError(null);
      console.log('[Voice] Uploading audio for player:', currentPlayer.name);

      // Отправляем на backend
      const response = await enrollVoice(currentPlayer.id, audioBlob);

      if (response.ok) {
        console.log('[Voice] ✅ Voice enrolled successfully:', response);
        
        // Отмечаем игрока как зарегистрированного
        setPlayers(prev => prev.map((p, i) =>
          i === currentIndex ? { ...p, registered: true, hasVoice: true } : p
        ));

        // Переходим к следующему игроку
        if (currentIndex < players.length - 1) {
          setTimeout(() => {
            setCurrentIndex(currentIndex + 1);
            setRecordingProgress(0);
          }, 300);
        }
      } else {
        console.error('[Voice] Error:', response.error);
        setError(response.message || 'Ошибка при регистрации голоса');
      }
    } catch (err) {
      console.error('[Voice] Failed to upload audio:', err);
      setError('Не удалось отправить аудио на сервер');
    }
  };

  const handleSkip = () => {
    if (currentIndex < players.length - 1) {
      setCurrentIndex(currentIndex + 1);
      setRecordingProgress(0);
      setError(null);
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
            {players.filter(p => p.registered).length} / {players.length}
          </span>
        </div>
        <div style={{ width: '100%', height: '6px', background: 'rgba(255,255,255,0.08)', borderRadius: '3px' }}>
          <div style={{
            height: '100%',
            background: '#4f46e5',
            borderRadius: '3px',
            width: `${(players.filter(p => p.registered).length / players.length) * 100}%`,
            transition: 'width 0.3s ease',
          }} />
        </div>
      </div>

      {/* Error Message */}
      {error && (
        <div style={{
          background: 'rgba(239,68,68,0.1)',
          border: '1px solid rgba(239,68,68,0.3)',
          color: '#ef4444',
          padding: '1rem',
          borderRadius: '8px',
          maxWidth: '800px',
          width: '100%',
        }}>
          ⚠️ {error}
        </div>
      )}

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
            {currentPlayer?.name || 'Загрузка...'}
          </h2>

          <p style={{
            color: '#94a3b8',
            marginBottom: '1.5rem',
            fontSize: '0.9375rem',
          }}>
            Произнесите: "Я {currentPlayer?.name}, игрок номер {currentPlayer?.id}"
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
            {!isRecording && currentPlayer?.registered && (
              <div style={{ fontSize: '4rem', color: '#10b981' }}>✓</div>
            )}
            {!isRecording && currentPlayer && !currentPlayer.registered && (
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

          {currentPlayer && !currentPlayer.registered && (
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
