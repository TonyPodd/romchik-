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
  const progress = (enrolledCount / 10) * 100;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      className="setup-page"
    >
      <div className="card glass large">
        {/* Enhanced header with progress bar */}
        <div className="card-header">
          <motion.h2
            className="card-title"
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
          >
            Регистрация игроков
          </motion.h2>
          <motion.div
            className="enrollment-progress"
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
          >
            <div className="progress-text">
              <span className="enrolled-count">{enrolledCount}</span>
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
            {!showScanner ? (
              <motion.div
                key="input"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 20 }}
                transition={{ duration: 0.3 }}
                className="enrollment-input"
              >
                {/* Enhanced player grid */}
                <motion.div
                  className="players-grid"
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
                  {players.map((player, index) => (
                    <motion.div
                      key={player.id}
                      className={`player-card glass ${
                        player.enrolled ? "enrolled" : ""
                      } ${index === currentPlayer ? "active" : ""}`}
                      variants={{
                        hidden: { opacity: 0, scale: 0.8 },
                        show: { opacity: 1, scale: 1 },
                      }}
                      onClick={() => !player.enrolled && setCurrentPlayer(index)}
                      whileHover={{
                        scale: 1.05,
                        backgroundColor: "rgba(255, 255, 255, 0.1)",
                        boxShadow: "0 8px 32px rgba(115, 194, 255, 0.3)",
                      }}
                      whileTap={{ scale: 0.95 }}
                      transition={{ type: "spring", damping: 15, stiffness: 300 }}
                    >
                      <motion.div
                        className="player-number"
                        animate={
                          index === currentPlayer && !player.enrolled
                            ? {
                                scale: [1, 1.2, 1],
                                boxShadow: [
                                  "0 0 0px rgba(115, 194, 255, 0)",
                                  "0 0 20px rgba(115, 194, 255, 0.6)",
                                  "0 0 0px rgba(115, 194, 255, 0)",
                                ],
                              }
                            : {}
                        }
                        transition={{ duration: 2, repeat: Infinity }}
                      >
                        {player.id}
                      </motion.div>
                      <div className="player-info">
                        <div className="player-name">{player.name}</div>
                        {player.enrolled && (
                          <motion.div
                            className="enrolled-badge glass"
                            initial={{ scale: 0, rotate: -180 }}
                            animate={{ scale: 1, rotate: 0 }}
                            transition={{ type: "spring", damping: 10 }}
                          >
                            <span className="check-icon">✓</span>
                            <span>Готов</span>
                          </motion.div>
                        )}
                      </div>
                      {/* Active indicator ring */}
                      {index === currentPlayer && !player.enrolled && (
                        <motion.div
                          className="active-ring"
                          layoutId="activePlayer"
                          transition={{ type: "spring", damping: 20, stiffness: 300 }}
                        />
                      )}
                    </motion.div>
                  ))}
                </motion.div>

                {/* Enhanced name input section */}
                {!allEnrolled && (
                  <motion.div
                    className="name-input-section glass"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.2 }}
                  >
                    <motion.h3
                      className="section-title"
                      animate={{
                        textShadow: [
                          "0 0 0px rgba(115, 194, 255, 0)",
                          "0 0 10px rgba(115, 194, 255, 0.5)",
                          "0 0 0px rgba(115, 194, 255, 0)",
                        ],
                      }}
                      transition={{ duration: 2, repeat: Infinity }}
                    >
                      Игрок {currentPlayer + 1}
                    </motion.h3>
                    <motion.input
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
                      whileFocus={{
                        scale: 1.02,
                        boxShadow: "0 8px 32px rgba(115, 194, 255, 0.4)",
                      }}
                      transition={{ type: "spring", damping: 15 }}
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

                {/* Enhanced completion message */}
                {allEnrolled && (
                  <motion.div
                    className="completion-message glass"
                    initial={{ opacity: 0, scale: 0.8, y: 20 }}
                    animate={{ opacity: 1, scale: 1, y: 0 }}
                    transition={{ type: "spring", damping: 15 }}
                  >
                    <motion.div
                      className="success-icon"
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
                      Все игроки зарегистрированы!
                    </motion.h3>
                    <motion.p
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: 0.3 }}
                    >
                      Можно переходить к следующему шагу
                    </motion.p>
                    {/* Celebration particles */}
                    {Array.from({ length: 20 }).map((_, i) => (
                      <motion.div
                        key={i}
                        className="celebration-particle"
                        initial={{
                          x: 0,
                          y: 0,
                          opacity: 1,
                          scale: Math.random() * 0.5 + 0.5,
                        }}
                        animate={{
                          x: (Math.random() - 0.5) * 200,
                          y: (Math.random() - 0.5) * 200,
                          opacity: 0,
                        }}
                        transition={{
                          duration: 1.5,
                          delay: i * 0.05,
                          ease: "easeOut",
                        }}
                      />
                    ))}
                  </motion.div>
                )}
              </motion.div>
            ) : (
              <motion.div
                key="scanner"
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.9 }}
                transition={{ duration: 0.4, ease: "easeOut" }}
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
