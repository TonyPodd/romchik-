// src/pages/VoiceRegistrationPage.tsx
import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import { GlassButton } from "../components/GlassButton";

export const VoiceRegistrationPage: React.FC = () => {
  const navigate = useNavigate();
  const [isRecording, setIsRecording] = useState(false);
  const [recordedPlayers, setRecordedPlayers] = useState<number[]>([]);
  const [currentPlayer, setCurrentPlayer] = useState(1);
  const [recordProgress, setRecordProgress] = useState(0);

  const handleRecord = () => {
    setIsRecording(true);
    setRecordProgress(0);
  };

  // Recording progress animation
  useEffect(() => {
    if (isRecording) {
      const interval = setInterval(() => {
        setRecordProgress((prev) => {
          if (prev >= 100) {
            clearInterval(interval);
            setIsRecording(false);
            setRecordedPlayers([...recordedPlayers, currentPlayer]);
            if (currentPlayer < 10) {
              setCurrentPlayer(currentPlayer + 1);
            }
            return 100;
          }
          return prev + 3.33; // 100% in 3 seconds
        });
      }, 100);
      return () => clearInterval(interval);
    }
  }, [isRecording, currentPlayer, recordedPlayers]);

  const allRecorded = recordedPlayers.length === 10;
  const progress = (recordedPlayers.length / 10) * 100;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      className="setup-page"
    >
      <div className="card glass large">
        {/* Enhanced header with progress */}
        <div className="card-header">
          <motion.h2
            className="card-title"
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
          >
            Регистрация голосов
          </motion.h2>
          <motion.div
            className="voice-progress"
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
          >
            <div className="progress-text">
              <span className="recorded-count">{recordedPlayers.length}</span>
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
          </motion.div>
        </div>

        <div className="card-body">
          <AnimatePresence mode="wait">
            {!allRecorded ? (
              <motion.div
                key="recording"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 20 }}
                transition={{ duration: 0.3 }}
                className="voice-recording"
              >
                {/* Enhanced waveform visualization */}
                <motion.div
                  className="waveform-container glass"
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: 0.2 }}
                >
                  {/* Circular audio visualizer */}
                  <svg className="audio-circle" viewBox="0 0 200 200">
                    <defs>
                      <radialGradient id="audioGradient">
                        <stop offset="0%" stopColor="rgba(115, 194, 255, 0.8)" />
                        <stop offset="100%" stopColor="rgba(167, 139, 250, 0.4)" />
                      </radialGradient>
                    </defs>
                    {/* Center circle */}
                    <motion.circle
                      cx="100"
                      cy="100"
                      r="30"
                      fill="url(#audioGradient)"
                      animate={
                        isRecording
                          ? {
                              r: [30, 40, 30],
                              opacity: [0.8, 1, 0.8],
                            }
                          : {}
                      }
                      transition={{
                        duration: 1.5,
                        repeat: isRecording ? Infinity : 0,
                        ease: "easeInOut",
                      }}
                    />
                    {/* Audio bars in circle */}
                    {Array.from({ length: 60 }).map((_, i) => {
                      const angle = (i * 6) * (Math.PI / 180);
                      const baseRadius = 50;
                      const barHeight = isRecording ? Math.random() * 20 + 10 : 10;
                      const x1 = 100 + baseRadius * Math.cos(angle);
                      const y1 = 100 + baseRadius * Math.sin(angle);
                      const x2 = 100 + (baseRadius + barHeight) * Math.cos(angle);
                      const y2 = 100 + (baseRadius + barHeight) * Math.sin(angle);

                      return (
                        <motion.line
                          key={i}
                          x1={x1}
                          y1={y1}
                          x2={x2}
                          y2={y2}
                          stroke="rgba(115, 194, 255, 0.6)"
                          strokeWidth="2"
                          strokeLinecap="round"
                          animate={
                            isRecording
                              ? {
                                  x2: [x2, 100 + (baseRadius + Math.random() * 30 + 10) * Math.cos(angle)],
                                  y2: [y2, 100 + (baseRadius + Math.random() * 30 + 10) * Math.sin(angle)],
                                  opacity: [0.3, 0.8, 0.3],
                                }
                              : {}
                          }
                          transition={{
                            duration: 0.2 + Math.random() * 0.3,
                            repeat: isRecording ? Infinity : 0,
                            delay: i * 0.01,
                          }}
                        />
                      );
                    })}
                  </svg>

                  {/* Recording indicator */}
                  {isRecording && (
                    <motion.div
                      className="recording-indicator glass"
                      initial={{ opacity: 0, scale: 0.8 }}
                      animate={{ opacity: 1, scale: 1 }}
                      exit={{ opacity: 0, scale: 0.8 }}
                    >
                      <motion.div
                        className="rec-dot"
                        animate={{
                          scale: [1, 1.3, 1],
                          boxShadow: [
                            "0 0 10px rgba(239, 68, 68, 0.5)",
                            "0 0 20px rgba(239, 68, 68, 0.8)",
                            "0 0 10px rgba(239, 68, 68, 0.5)",
                          ],
                        }}
                        transition={{ duration: 1, repeat: Infinity }}
                      />
                      <span>Запись...</span>
                      <span className="time">{Math.round(recordProgress / 100 * 3)}s</span>
                    </motion.div>
                  )}

                  {/* Progress ring */}
                  {isRecording && (
                    <svg className="progress-ring" viewBox="0 0 200 200">
                      <motion.circle
                        cx="100"
                        cy="100"
                        r="90"
                        fill="none"
                        stroke="rgba(34, 197, 94, 0.8)"
                        strokeWidth="3"
                        strokeDasharray="565.48"
                        strokeDashoffset={565.48 - (565.48 * recordProgress) / 100}
                        strokeLinecap="round"
                        style={{ transformOrigin: "50% 50%", transform: "rotate(-90deg)" }}
                        transition={{ duration: 0.1 }}
                      />
                    </svg>
                  )}
                </motion.div>

                {/* Current player info */}
                <motion.div
                  className="current-player-info glass"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.3 }}
                >
                  <motion.h3
                    className="player-title"
                    animate={
                      isRecording
                        ? {}
                        : {
                            textShadow: [
                              "0 0 0px rgba(115, 194, 255, 0)",
                              "0 0 10px rgba(115, 194, 255, 0.5)",
                              "0 0 0px rgba(115, 194, 255, 0)",
                            ],
                          }
                    }
                    transition={{ duration: 2, repeat: Infinity }}
                  >
                    Игрок {currentPlayer}
                  </motion.h3>
                  <p className="instruction">
                    Произнесите любое слово для регистрации голоса
                  </p>
                  <GlassButton
                    onClick={handleRecord}
                    disabled={isRecording}
                    size="large"
                  >
                    {isRecording ? "⏺ Запись..." : "🎙️ Начать запись"}
                  </GlassButton>
                </motion.div>

                {/* Enhanced progress grid */}
                <motion.div
                  className="players-progress"
                  variants={{
                    hidden: { opacity: 0 },
                    show: {
                      opacity: 1,
                      transition: {
                        staggerChildren: 0.05,
                      },
                    },
                  }}
                  initial="hidden"
                  animate="show"
                >
                  {Array.from({ length: 10 }).map((_, i) => (
                    <motion.div
                      key={i}
                      className={`player-dot glass ${
                        recordedPlayers.includes(i + 1) ? "recorded" : ""
                      } ${i + 1 === currentPlayer ? "active" : ""}`}
                      variants={{
                        hidden: { opacity: 0, scale: 0 },
                        show: { opacity: 1, scale: 1 },
                      }}
                      whileHover={{ scale: 1.3 }}
                      animate={
                        i + 1 === currentPlayer && !recordedPlayers.includes(i + 1)
                          ? {
                              scale: [1, 1.2, 1],
                              boxShadow: [
                                "0 0 0px rgba(115, 194, 255, 0)",
                                "0 0 15px rgba(115, 194, 255, 0.6)",
                                "0 0 0px rgba(115, 194, 255, 0)",
                              ],
                            }
                          : {}
                      }
                      transition={{ duration: 1.5, repeat: Infinity }}
                    >
                      <span className="dot-number">{i + 1}</span>
                      {recordedPlayers.includes(i + 1) && (
                        <motion.div
                          className="check-overlay"
                          initial={{ scale: 0, rotate: -180 }}
                          animate={{ scale: 1, rotate: 0 }}
                          transition={{ type: "spring", damping: 10 }}
                        >
                          ✓
                        </motion.div>
                      )}
                    </motion.div>
                  ))}
                </motion.div>
              </motion.div>
            ) : (
              <motion.div
                key="complete"
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 1.1 }}
                className="completion-view glass"
              >
                <motion.div
                  className="success-icon large"
                  animate={{
                    scale: [1, 1.2, 1],
                    rotate: [0, 360],
                  }}
                  transition={{
                    scale: { duration: 1, repeat: Infinity, repeatDelay: 1 },
                    rotate: { duration: 0.8, ease: "easeOut" },
                  }}
                >
                  ✓
                </motion.div>
                <motion.h3
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.2 }}
                >
                  Все голоса записаны!
                </motion.h3>
                <motion.p
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.3 }}
                >
                  Система готова к началу игры
                </motion.p>
                {/* Sound waves */}
                {Array.from({ length: 3 }).map((_, i) => (
                  <motion.div
                    key={i}
                    className="sound-wave"
                    animate={{
                      scale: [1, 2, 1],
                      opacity: [0.5, 0, 0.5],
                    }}
                    transition={{
                      duration: 2,
                      repeat: Infinity,
                      delay: i * 0.5,
                    }}
                  />
                ))}
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
