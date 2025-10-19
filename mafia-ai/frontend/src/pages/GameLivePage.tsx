// src/pages/GameLivePage.tsx
import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import { GlassButton } from "../components/GlassButton";

type GamePhase = "night" | "day" | "voting" | "finished";

export const GameLivePage: React.FC = () => {
  const navigate = useNavigate();
  const [phase, setPhase] = useState<GamePhase>("night");
  const [day, setDay] = useState(1);
  const [currentSpeaker, setCurrentSpeaker] = useState<number | null>(null);
  const [timer, setTimer] = useState(60);
  const [fouls, setFouls] = useState<Array<{ player: number; type: string }>>([]);

  useEffect(() => {
    if (currentSpeaker && timer > 0) {
      const interval = setInterval(() => setTimer((t) => t - 1), 1000);
      return () => clearInterval(interval);
    }
  }, [currentSpeaker, timer]);

  const startTurn = (playerId: number) => {
    setCurrentSpeaker(playerId);
    setTimer(60);
  };

  const addFoul = (player: number, type: string) => {
    setFouls([...fouls, { player, type }]);
  };

  return (
    <div className="game-live-page">
      <div className="game-header glass">
        <div className="game-info">
          <h2>День {day}</h2>
          <div className="phase-badge">{phase === "night" ? "🌙 Ночь" : phase === "day" ? "☀️ День" : "🗳️ Голосование"}</div>
        </div>
        <div className="game-controls">
          <GlassButton onClick={() => navigate("/game/stats")} variant="ghost">Завершить</GlassButton>
        </div>
      </div>

      <div className="game-content">
        {/* Video feed */}
        <div className="video-section glass">
          <div className="video-placeholder">
            <p>📹 Видеопоток</p>
          </div>
        </div>

        {/* Game state */}
        <div className="game-state glass">
          <h3>Текущий ход</h3>
          {currentSpeaker ? (
            <div className="speaker-info">
              <div className="speaker-name">Игрок {currentSpeaker}</div>
              <div className="timer">
                <div className="timer-ring" style={{ '--progress': timer / 60 } as any}>
                  <span>{timer}s</span>
                </div>
              </div>
              <GlassButton onClick={() => setCurrentSpeaker(null)} variant="ghost">
                Завершить ход
              </GlassButton>
            </div>
          ) : (
            <div className="player-select">
              <p>Выберите игрока:</p>
              <div className="player-buttons">
                {Array.from({ length: 10 }).map((_, i) => (
                  <button
                    key={i}
                    className="player-btn glass-btn"
                    onClick={() => startTurn(i + 1)}
                  >
                    {i + 1}
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Fouls monitor */}
        <div className="fouls-monitor glass">
          <h3>Фолы ({fouls.length})</h3>
          <div className="fouls-list">
            <AnimatePresence>
              {fouls.map((foul, i) => (
                <motion.div
                  key={i}
                  className="foul-item"
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: 20 }}
                >
                  <span className="foul-player">Игрок {foul.player}</span>
                  <span className="foul-type">{foul.type}</span>
                </motion.div>
              ))}
            </AnimatePresence>
          </div>
          {currentSpeaker && (
            <GlassButton
              onClick={() => addFoul(currentSpeaker, "Речь не в свой ход")}
              variant="ghost"
              size="small"
            >
              + Фол
            </GlassButton>
          )}
        </div>
      </div>
    </div>
  );
};
