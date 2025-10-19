// src/pages/TableDetectionPage.tsx
import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import { GlassButton } from "../components/GlassButton";

export const TableDetectionPage: React.FC = () => {
  const navigate = useNavigate();
  const [isDetecting, setIsDetecting] = useState(false);
  const [isDetected, setIsDetected] = useState(false);

  const handleAutoDetect = () => {
    setIsDetecting(true);
    // Simulate detection
    setTimeout(() => {
      setIsDetecting(false);
      setIsDetected(true);
    }, 3000);
  };

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="setup-page"
    >
      <div className="card glass large">
        <div className="card-header">
          <h2 className="card-title">Определение стола</h2>
          <p className="card-subtitle">
            Настройте область видимости игрового стола
          </p>
        </div>

        <div className="card-body">
          {/* Video preview */}
          <div className="video-preview glass">
            <div className="preview-placeholder">
              {!isDetected && !isDetecting && (
                <motion.div
                  className="placeholder-content"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                >
                  <div className="placeholder-icon">📹</div>
                  <p>Камера не активна</p>
                </motion.div>
              )}

              {isDetecting && (
                <motion.div
                  className="detecting-overlay"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                >
                  <div className="scanner-grid">
                    {Array.from({ length: 100 }).map((_, i) => (
                      <motion.div
                        key={i}
                        className="grid-cell"
                        animate={{
                          opacity: [0.1, 0.5, 0.1],
                          scale: [0.8, 1, 0.8],
                        }}
                        transition={{
                          duration: 2,
                          delay: i * 0.02,
                          repeat: Infinity,
                        }}
                      />
                    ))}
                  </div>
                  <p className="scanning-text">Сканирование стола...</p>
                </motion.div>
              )}

              {isDetected && (
                <motion.div
                  className="detected-overlay"
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                >
                  <div className="table-outline">
                    <svg viewBox="0 0 400 300" className="table-svg">
                      <motion.ellipse
                        cx="200"
                        cy="150"
                        rx="160"
                        ry="120"
                        fill="none"
                        stroke="rgba(34, 197, 94, 0.8)"
                        strokeWidth="3"
                        strokeDasharray="10 5"
                        initial={{ pathLength: 0 }}
                        animate={{ pathLength: 1 }}
                        transition={{ duration: 1 }}
                      />
                    </svg>
                    {/* Player positions */}
                    {Array.from({ length: 10 }).map((_, i) => {
                      const angle = (i * 36 - 90) * (Math.PI / 180);
                      const x = 200 + 160 * Math.cos(angle);
                      const y = 150 + 120 * Math.sin(angle);
                      return (
                        <motion.div
                          key={i}
                          className="player-marker"
                          style={{
                            position: "absolute",
                            left: `${(x / 400) * 100}%`,
                            top: `${(y / 300) * 100}%`,
                            transform: "translate(-50%, -50%)",
                          }}
                          initial={{ scale: 0 }}
                          animate={{ scale: 1 }}
                          transition={{ delay: 0.5 + i * 0.1 }}
                        >
                          <div className="marker-dot" />
                          <div className="marker-label">{i + 1}</div>
                        </motion.div>
                      );
                    })}
                  </div>
                  <motion.p
                    className="detected-text"
                    initial={{ y: 10, opacity: 0 }}
                    animate={{ y: 0, opacity: 1 }}
                    transition={{ delay: 1.5 }}
                  >
                    ✓ Стол обнаружен
                  </motion.p>
                </motion.div>
              )}
            </div>
          </div>

          {/* Controls */}
          <div className="table-controls">
            {!isDetected && (
              <GlassButton
                onClick={handleAutoDetect}
                disabled={isDetecting}
                size="large"
              >
                {isDetecting ? "Обнаружение..." : "🔍 Автоопределение"}
              </GlassButton>
            )}

            {isDetected && (
              <motion.div
                className="success-message"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
              >
                <div className="success-icon">✓</div>
                <p>Стол успешно настроен</p>
                <p className="success-detail">10 позиций определено</p>
              </motion.div>
            )}
          </div>
        </div>

        <div className="card-footer">
          <GlassButton onClick={() => navigate("/")} variant="ghost">
            ← Назад
          </GlassButton>
          <GlassButton
            onClick={() => navigate("/setup/players")}
            disabled={!isDetected}
          >
            Далее →
          </GlassButton>
        </div>
      </div>
    </motion.div>
  );
};
