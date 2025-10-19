// src/pages/VoiceRegistrationPage.tsx
import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import { GlassButton } from "../components/GlassButton";

export const VoiceRegistrationPage: React.FC = () => {
  const navigate = useNavigate();
  const [isRecording, setIsRecording] = useState(false);
  const [recordedPlayers, setRecordedPlayers] = useState<number[]>([]);
  const [currentPlayer, setCurrentPlayer] = useState(1);

  const handleRecord = () => {
    setIsRecording(true);
    // Simulate recording
    setTimeout(() => {
      setIsRecording(false);
      setRecordedPlayers([...recordedPlayers, currentPlayer]);
      if (currentPlayer < 10) {
        setCurrentPlayer(currentPlayer + 1);
      }
    }, 3000);
  };

  const allRecorded = recordedPlayers.length === 10;

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="setup-page"
    >
      <div className="card glass large">
        <div className="card-header">
          <h2 className="card-title">Регистрация голосов</h2>
          <p className="card-subtitle">
            Записано: {recordedPlayers.length} / 10
          </p>
        </div>

        <div className="card-body">
          <AnimatePresence mode="wait">
            {!allRecorded ? (
              <motion.div
                key="recording"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="voice-recording"
              >
                {/* Waveform visualization */}
                <div className="waveform-container">
                  <div className="waveform glass">
                    {Array.from({ length: 50 }).map((_, i) => (
                      <motion.div
                        key={i}
                        className="wave-bar"
                        animate={{
                          scaleY: isRecording
                            ? [1, Math.random() * 3 + 1, 1]
                            : 1,
                        }}
                        transition={{
                          duration: 0.3,
                          delay: i * 0.02,
                          repeat: isRecording ? Infinity : 0,
                        }}
                      />
                    ))}
                  </div>
                  {isRecording && (
                    <motion.div
                      className="recording-indicator"
                      animate={{ opacity: [1, 0.5, 1] }}
                      transition={{ duration: 1, repeat: Infinity }}
                    >
                      ⏺ Запись...
                    </motion.div>
                  )}
                </div>

                {/* Current player */}
                <div className="current-player-info">
                  <h3>Игрок {currentPlayer}</h3>
                  <p className="instruction">
                    Произнесите любое слово для регистрации голоса
                  </p>
                  <GlassButton
                    onClick={handleRecord}
                    disabled={isRecording}
                    size="large"
                  >
                    {isRecording ? "Запись..." : "🎙️ Начать запись"}
                  </GlassButton>
                </div>

                {/* Progress grid */}
                <div className="players-progress">
                  {Array.from({ length: 10 }).map((_, i) => (
                    <motion.div
                      key={i}
                      className={`player-dot ${
                        recordedPlayers.includes(i + 1) ? "recorded" : ""
                      } ${i + 1 === currentPlayer ? "active" : ""}`}
                      whileHover={{ scale: 1.2 }}
                    >
                      {recordedPlayers.includes(i + 1) && "✓"}
                    </motion.div>
                  ))}
                </div>
              </motion.div>
            ) : (
              <motion.div
                key="complete"
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0 }}
                className="completion-view"
              >
                <div className="success-icon large">✓</div>
                <h3>Все голоса записаны!</h3>
                <p>Система готова к началу игры</p>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        <div className="card-footer">
          <GlassButton
            onClick={() => navigate("/setup/players")}
            variant="ghost"
          >
            ← Назад
          </GlassButton>
          <GlassButton
            onClick={() => navigate("/game/setup")}
            disabled={!allRecorded}
          >
            К игре →
          </GlassButton>
        </div>
      </div>
    </motion.div>
  );
};
