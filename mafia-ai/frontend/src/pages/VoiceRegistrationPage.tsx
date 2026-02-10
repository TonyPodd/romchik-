// src/pages/VoiceRegistrationPage.tsx
import React, { useState, useRef, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { GlassButton } from '../components/GlassButton';

interface VoiceProfile {
  player_id: number;
  player_name: string;
  samples_count: number;
  created_at: number;
}

interface ActiveSpeaker {
  player_id: number;
  player_name: string;
  confidence: number;
}

export const VoiceRegistrationPage: React.FC = () => {
  const navigate = useNavigate();
  const [profiles, setProfiles] = useState<VoiceProfile[]>([]);
  const [activeSpeakers, setActiveSpeakers] = useState<ActiveSpeaker[]>([]);
  const [isRecording, setIsRecording] = useState(false);
  const [recordingForPlayer, setRecordingForPlayer] = useState<string>('');
  const [isListening, setIsListening] = useState(false);
  const [error, setError] = useState<string>('');
  const [showTesting, setShowTesting] = useState(false);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const audioContextRef = useRef<AudioContext | null>(null);
  const micStreamRef = useRef<MediaStream | null>(null);
  const identifyIntervalRef = useRef<number | null>(null);
  const audioBufferRef = useRef<Float32Array>(new Float32Array(0));
  const scriptProcessorRef = useRef<ScriptProcessorNode | null>(null);

  // Load profiles on mount
  useEffect(() => {
    loadProfiles();
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopListening();
      if (micStreamRef.current) {
        micStreamRef.current.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  const loadProfiles = async () => {
    try {
      const response = await fetch('http://localhost:8000/voice/profiles');
      const data = await response.json();
      if (data.ok) {
        setProfiles(data.profiles);
      }
    } catch (err) {
      console.error('Failed to load profiles:', err);
    }
  };

  const startRecording = async (playerName: string) => {
    if (!playerName.trim()) {
      setError('Введите имя игрока');
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      micStreamRef.current = stream;

      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        await registerVoice(playerName, audioBlob);
        stream.getTracks().forEach(track => track.stop());
        micStreamRef.current = null;
      };

      mediaRecorder.start();
      setIsRecording(true);
      setRecordingForPlayer(playerName);
      setError('');
    } catch (err) {
      setError('Не удалось получить доступ к микрофону');
      console.error(err);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      setRecordingForPlayer('');
    }
  };

  // Simple linear interpolation resampler
  const resampleAudio = (audioData: Float32Array, originalRate: number, targetRate: number): Float32Array => {
    if (originalRate === targetRate) {
      return audioData;
    }

    const ratio = originalRate / targetRate;
    const newLength = Math.floor(audioData.length / ratio);
    const result = new Float32Array(newLength);

    for (let i = 0; i < newLength; i++) {
      const srcIndex = i * ratio;
      const srcIndexFloor = Math.floor(srcIndex);
      const srcIndexCeil = Math.min(srcIndexFloor + 1, audioData.length - 1);
      const fraction = srcIndex - srcIndexFloor;

      // Linear interpolation
      result[i] = audioData[srcIndexFloor] * (1 - fraction) + audioData[srcIndexCeil] * fraction;
    }

    return result;
  };

  const registerVoice = async (playerName: string, audioBlob: Blob) => {
    try {
      // Convert blob to audio samples
      const audioBuffer = await audioBlob.arrayBuffer();
      const audioContext = new AudioContext();
      const decodedData = await audioContext.decodeAudioData(audioBuffer);

      // Extract audio samples (mono channel)
      const audioData = decodedData.getChannelData(0);
      const originalSampleRate = decodedData.sampleRate;

      // Resample to 16kHz
      const targetSampleRate = 16000;
      const resampledData = resampleAudio(audioData, originalSampleRate, targetSampleRate);
      const samples = Array.from(resampledData);

      // Create multiple samples by splitting the audio
      const sampleSize = Math.floor(samples.length / 3);
      const audioSamples = [
        samples.slice(0, sampleSize),
        samples.slice(sampleSize, sampleSize * 2),
        samples.slice(sampleSize * 2)
      ];

      console.log('Регистрация голоса:', {
        playerName,
        originalSampleRate,
        targetSampleRate,
        originalLength: audioData.length,
        resampledLength: resampledData.length,
        samplesPerSegment: sampleSize
      });

      const response = await fetch('http://localhost:8000/voice/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          player_id: Date.now(),
          player_name: playerName,
          audio_samples: audioSamples,
          sample_rate: targetSampleRate
        })
      });

      const data = await response.json();
      if (data.ok) {
        await loadProfiles();
        setError('');
        setRecordingForPlayer('');
      } else {
        setError(`Ошибка регистрации: ${data.error || 'Неизвестная ошибка'}`);
      }

      audioContext.close();
    } catch (err) {
      setError(`Ошибка регистрации: ${err}`);
      console.error(err);
    }
  };

  const startListening = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      micStreamRef.current = stream;

      // Create audio context
      const audioContext = new AudioContext();
      audioContextRef.current = audioContext;

      const source = audioContext.createMediaStreamSource(stream);

      // Use ScriptProcessorNode to capture audio data
      const bufferSize = 4096;
      const scriptProcessor = audioContext.createScriptProcessor(bufferSize, 1, 1);
      scriptProcessorRef.current = scriptProcessor;

      // Buffer to accumulate audio (1 second worth)
      const targetBufferSize = Math.floor(audioContext.sampleRate);
      audioBufferRef.current = new Float32Array(targetBufferSize);
      let writePosition = 0;

      scriptProcessor.onaudioprocess = (e) => {
        const inputData = e.inputBuffer.getChannelData(0);

        // Accumulate audio data
        for (let i = 0; i < inputData.length; i++) {
          audioBufferRef.current[writePosition] = inputData[i];
          writePosition = (writePosition + 1) % targetBufferSize;
        }
      };

      source.connect(scriptProcessor);
      scriptProcessor.connect(audioContext.destination);

      setIsListening(true);
      setError('');

      console.log('Начато прослушивание с sample rate:', audioContext.sampleRate);

      // Start periodic speaker identification
      identifyIntervalRef.current = window.setInterval(async () => {
        await identifySpeaker();
      }, 1000);
    } catch (err) {
      setError('Не удалось получить доступ к микрофону');
      console.error(err);
    }
  };

  const stopListening = () => {
    if (identifyIntervalRef.current) {
      clearInterval(identifyIntervalRef.current);
      identifyIntervalRef.current = null;
    }

    if (scriptProcessorRef.current) {
      scriptProcessorRef.current.disconnect();
      scriptProcessorRef.current = null;
    }

    if (micStreamRef.current) {
      micStreamRef.current.getTracks().forEach(track => track.stop());
      micStreamRef.current = null;
    }

    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }

    setIsListening(false);
    setActiveSpeakers([]);
  };

  const identifySpeaker = async () => {
    if (!audioContextRef.current || audioBufferRef.current.length === 0) return;

    try {
      // Get accumulated audio data
      const audioData = new Float32Array(audioBufferRef.current);
      const originalSampleRate = audioContextRef.current.sampleRate;

      // Resample to 16kHz
      const targetSampleRate = 16000;
      const resampledData = resampleAudio(audioData, originalSampleRate, targetSampleRate);
      const audioSamples = Array.from(resampledData);

      const response = await fetch('http://localhost:8000/voice/identify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          audio: audioSamples,
          sample_rate: targetSampleRate
        })
      });

      const data = await response.json();
      if (data.ok && data.player_id) {
        setActiveSpeakers([{
          player_id: data.player_id,
          player_name: data.player_name,
          confidence: data.confidence
        }]);
      } else {
        setActiveSpeakers([]);
      }
    } catch (err) {
      console.error('Ошибка идентификации:', err);
    }
  };

  const clearAllProfiles = async () => {
    if (!confirm('Вы уверены что хотите удалить все голосовые профили?')) return;

    try {
      const response = await fetch('http://localhost:8000/voice/clear', {
        method: 'POST'
      });
      const data = await response.json();
      if (data.ok) {
        await loadProfiles();
      }
    } catch (err) {
      setError('Не удалось очистить профили');
      console.error(err);
    }
  };

  const progress = profiles.length > 0 ? (profiles.length / 10) * 100 : 0;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="setup-page"
    >
      <div className="card glass large">
        <div className="card-header">
          <h2 className="card-title">Регистрация голосов</h2>
          <div className="voice-progress" style={{ marginTop: '1rem' }}>
            <div className="progress-text">
              <span className="recorded-count">{profiles.length}</span>
              <span className="separator">/</span>
              <span className="total-count">10</span>
            </div>
            <div className="progress-bar glass">
              <motion.div
                className="progress-fill"
                style={{ width: `${progress}%` }}
                initial={{ width: 0 }}
                animate={{ width: `${progress}%` }}
                transition={{ duration: 0.5, ease: "easeOut" }}
              />
            </div>
          </div>
        </div>

        <div className="card-body">
          {error && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              style={{
                padding: '1rem',
                background: 'rgba(239, 68, 68, 0.1)',
                border: '1px solid rgba(239, 68, 68, 0.3)',
                borderRadius: '0.5rem',
                color: '#ef4444',
                marginBottom: '1rem'
              }}
            >
              {error}
            </motion.div>
          )}

          {/* Voice Registration Section */}
          <div style={{ marginBottom: '2rem' }}>
            <h3 style={{ marginBottom: '1rem', opacity: 0.9 }}>Записать голос</h3>
            <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap', alignItems: 'flex-end' }}>
              <div style={{ flex: 1, minWidth: '200px' }}>
                <input
                  type="text"
                  className="glass-input"
                  placeholder="Имя игрока"
                  value={recordingForPlayer}
                  onChange={(e) => setRecordingForPlayer(e.target.value)}
                  disabled={isRecording}
                  style={{ width: '100%' }}
                />
              </div>
              <GlassButton
                onClick={() => isRecording ? stopRecording() : startRecording(recordingForPlayer)}
                variant={isRecording ? 'danger' : 'primary'}
              >
                {isRecording ? '⏹️ Остановить' : '🎤 Начать запись'}
              </GlassButton>
            </div>
            {isRecording && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                style={{
                  marginTop: '1rem',
                  padding: '0.75rem',
                  background: 'rgba(16, 185, 129, 0.1)',
                  borderRadius: '0.5rem',
                  textAlign: 'center'
                }}
              >
                Запись {recordingForPlayer}... Говорите в течение 3-5 секунд
              </motion.div>
            )}
          </div>

          {/* Registered Profiles */}
          <div style={{ marginBottom: '2rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
              <h3 style={{ opacity: 0.9 }}>Зарегистрированные голоса ({profiles.length})</h3>
              {profiles.length > 0 && (
                <GlassButton onClick={clearAllProfiles} variant="ghost" size="small">
                  Очистить все
                </GlassButton>
              )}
            </div>
            {profiles.length === 0 ? (
              <div style={{ padding: '2rem', textAlign: 'center', opacity: 0.5 }}>
                Нет зарегистрированных голосов
              </div>
            ) : (
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: '1rem' }}>
                {profiles.map((profile) => (
                  <motion.div
                    key={profile.player_id}
                    className="glass"
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    style={{
                      padding: '1rem',
                      borderRadius: '0.5rem'
                    }}
                  >
                    <div style={{ fontWeight: 600, marginBottom: '0.5rem' }}>
                      {profile.player_name}
                    </div>
                    <div style={{ fontSize: '0.85rem', opacity: 0.7 }}>
                      {profile.samples_count} сэмпла
                    </div>
                  </motion.div>
                ))}
              </div>
            )}
          </div>

          {/* Testing Section Toggle */}
          {profiles.length > 0 && (
            <div style={{ marginBottom: '1rem' }}>
              <GlassButton
                onClick={() => setShowTesting(!showTesting)}
                variant="ghost"
                style={{ width: '100%' }}
              >
                {showTesting ? '▼ Скрыть тестирование' : '▶ Показать тестирование определения голосов'}
              </GlassButton>
            </div>
          )}

          {/* Real-time Speaker Detection */}
          {showTesting && profiles.length > 0 && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
            >
              <div style={{ borderTop: '1px solid rgba(255,255,255,0.1)', paddingTop: '2rem', marginTop: '1rem' }}>
                <h3 style={{ marginBottom: '1rem', opacity: 0.9 }}>Тестирование определения</h3>
                <div style={{ display: 'flex', gap: '1rem', alignItems: 'center', marginBottom: '1rem' }}>
                  <GlassButton
                    onClick={() => isListening ? stopListening() : startListening()}
                    variant={isListening ? 'danger' : 'primary'}
                  >
                    {isListening ? '⏹️ Остановить' : '👂 Начать прослушивание'}
                  </GlassButton>
                </div>

                {/* Active Speakers Display */}
                {isListening && (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="glass"
                    style={{
                      padding: '2rem',
                      borderRadius: '0.75rem',
                      minHeight: '150px',
                      display: 'flex',
                      flexDirection: 'column',
                      alignItems: 'center',
                      justifyContent: 'center'
                    }}
                  >
                    <div style={{
                      fontSize: '3rem',
                      marginBottom: '1rem',
                      animation: 'pulse 2s infinite'
                    }}>
                      👂
                    </div>
                    <AnimatePresence mode="wait">
                      {activeSpeakers.length === 0 ? (
                        <motion.div
                          key="waiting"
                          initial={{ opacity: 0 }}
                          animate={{ opacity: 1 }}
                          exit={{ opacity: 0 }}
                          style={{ textAlign: 'center', opacity: 0.6 }}
                        >
                          Слушаю...
                        </motion.div>
                      ) : (
                        <motion.div
                          key="speaking"
                          initial={{ opacity: 0, scale: 0.8 }}
                          animate={{ opacity: 1, scale: 1 }}
                          exit={{ opacity: 0, scale: 0.8 }}
                          style={{ textAlign: 'center' }}
                        >
                          <div style={{ fontSize: '1.5rem', fontWeight: 600, marginBottom: '0.5rem' }}>
                            {activeSpeakers[0].player_name} говорит
                          </div>
                          <div style={{ opacity: 0.7 }}>
                            Уверенность: {(activeSpeakers[0].confidence * 100).toFixed(1)}%
                          </div>
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </motion.div>
                )}
              </div>
            </motion.div>
          )}
        </div>

        <div className="card-footer">
          <GlassButton onClick={() => navigate('/setup/players')} variant="ghost">
            ← Назад
          </GlassButton>
          <GlassButton
            onClick={() => navigate('/game/live')}
            disabled={profiles.length === 0}
          >
            К игре →
          </GlassButton>
        </div>
      </div>

      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.5; }
        }

        .glass-input {
          padding: 0.75rem;
          background: rgba(255, 255, 255, 0.05);
          border: 1px solid rgba(255, 255, 255, 0.1);
          border-radius: 0.5rem;
          color: white;
          font-size: 1rem;
          transition: all 0.3s;
        }

        .glass-input:focus {
          outline: none;
          background: rgba(255, 255, 255, 0.1);
          border-color: rgba(115, 194, 255, 0.5);
          box-shadow: 0 0 20px rgba(115, 194, 255, 0.3);
        }

        .glass-input::placeholder {
          color: rgba(255, 255, 255, 0.4);
        }

        .voice-progress {
          width: 100%;
        }

        .progress-text {
          display: flex;
          justify-content: center;
          align-items: baseline;
          gap: 0.25rem;
          margin-bottom: 0.5rem;
          font-size: 1.2rem;
          font-weight: 600;
        }

        .recorded-count {
          color: #10b981;
          font-size: 1.5rem;
        }

        .separator {
          opacity: 0.5;
        }

        .total-count {
          opacity: 0.7;
        }

        .progress-bar {
          width: 100%;
          height: 8px;
          border-radius: 4px;
          overflow: hidden;
          background: rgba(255, 255, 255, 0.05);
        }

        .progress-fill {
          height: 100%;
          background: linear-gradient(90deg, #10b981, #059669);
          border-radius: 4px;
          transition: width 0.5s ease;
        }
      `}</style>
    </motion.div>
  );
};
