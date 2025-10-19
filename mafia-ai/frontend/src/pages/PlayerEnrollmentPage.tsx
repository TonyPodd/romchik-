// src/pages/PlayerEnrollmentPage.tsx
import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import { FaceIDScanner } from "../components/FaceIDScanner";
import { GlassButton } from "../components/GlassButton";

interface Player {
  id: number;
  name: string;
  enrolled: boolean;
}

export const PlayerEnrollmentPage: React.FC = () => {
  const navigate = useNavigate();
  const [currentPlayer, setCurrentPlayer] = useState(0);
  const [playerName, setPlayerName] = useState("");
  const [showScanner, setShowScanner] = useState(false);
  const [players, setPlayers] = useState<Player[]>(
    Array.from({ length: 10 }, (_, i) => ({
      id: i + 1,
      name: `Игрок ${i + 1}`,
      enrolled: false,
    }))
  );

  const handleStartScan = () => {
    if (playerName.trim()) {
      setShowScanner(true);
    }
  };

  const handleScanComplete = () => {
    // Update player with name
    setPlayers((prev) =>
      prev.map((p) =>
        p.id === currentPlayer + 1
          ? { ...p, name: playerName, enrolled: true }
          : p
      )
    );

    setTimeout(() => {
      setShowScanner(false);
      setPlayerName("");

      // Move to next player
      if (currentPlayer < 9) {
        setCurrentPlayer(currentPlayer + 1);
      }
    }, 2000);
  };

  const enrolledCount = players.filter((p) => p.enrolled).length;
  const allEnrolled = enrolledCount === 10;

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="setup-page"
    >
      <div className="card glass large">
        <div className="card-header">
          <h2 className="card-title">Регистрация игроков</h2>
          <p className="card-subtitle">
            Зарегистрировано: {enrolledCount} / 10
          </p>
        </div>

        <div className="card-body">
          <AnimatePresence mode="wait">
            {!showScanner ? (
              <motion.div
                key="input"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 20 }}
                className="enrollment-input"
              >
                {/* Player list */}
                <div className="players-grid">
                  {players.map((player, index) => (
                    <motion.div
                      key={player.id}
                      className={`player-card glass ${
                        player.enrolled ? "enrolled" : ""
                      } ${index === currentPlayer ? "active" : ""}`}
                      onClick={() => !player.enrolled && setCurrentPlayer(index)}
                      whileHover={{ scale: 1.02 }}
                      whileTap={{ scale: 0.98 }}
                    >
                      <div className="player-number">{player.id}</div>
                      <div className="player-info">
                        <div className="player-name">{player.name}</div>
                        {player.enrolled && (
                          <div className="enrolled-badge">✓ Готов</div>
                        )}
                      </div>
                    </motion.div>
                  ))}
                </div>

                {/* Name input */}
                {!allEnrolled && (
                  <motion.div
                    className="name-input-section"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                  >
                    <h3 className="section-title">
                      Игрок {currentPlayer + 1}
                    </h3>
                    <input
                      type="text"
                      className="glass-input"
                      placeholder="Введите имя игрока"
                      value={playerName}
                      onChange={(e) => setPlayerName(e.target.value)}
                      onKeyPress={(e) => {
                        if (e.key === "Enter" && playerName.trim()) {
                          handleStartScan();
                        }
                      }}
                      autoFocus
                    />
                    <GlassButton
                      onClick={handleStartScan}
                      disabled={!playerName.trim()}
                      size="large"
                    >
                      📸 Сканировать лицо
                    </GlassButton>
                  </motion.div>
                )}

                {allEnrolled && (
                  <motion.div
                    className="completion-message"
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                  >
                    <div className="success-icon">✓</div>
                    <h3>Все игроки зарегистрированы!</h3>
                    <p>Можно переходить к следующему шагу</p>
                  </motion.div>
                )}
              </motion.div>
            ) : (
              <motion.div
                key="scanner"
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.9 }}
              >
                <FaceIDScanner
                  playerName={playerName}
                  onComplete={handleScanComplete}
                  autoStart
                />
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        <div className="card-footer">
          <GlassButton
            onClick={() => navigate("/setup/table")}
            variant="ghost"
          >
            ← Назад
          </GlassButton>
          <GlassButton
            onClick={() => navigate("/setup/voice")}
            disabled={!allEnrolled}
          >
            Далее →
          </GlassButton>
        </div>
      </div>
    </motion.div>
  );
};
