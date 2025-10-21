// src/pages/TableDetectionPage.tsx
import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import { GlassButton } from "../components/GlassButton";

type TableShape = "circle" | "oval" | "rectangle";

export const TableDetectionPage: React.FC = () => {
  const navigate = useNavigate();
  const [isDetecting, setIsDetecting] = useState(false);
  const [isDetected, setIsDetected] = useState(false);
  const [scanProgress, setScanProgress] = useState(0);
  const [tableShape, setTableShape] = useState<TableShape>("circle");

  const handleAutoDetect = () => {
    setIsDetecting(true);
    setScanProgress(0);
  };

  // Smooth progress animation
  useEffect(() => {
    if (isDetecting) {
      const interval = setInterval(() => {
        setScanProgress((prev) => {
          if (prev >= 100) {
            clearInterval(interval);
            setIsDetecting(false);
            setIsDetected(true);
            // Randomly detect table shape for demo
            const shapes: TableShape[] = ["circle", "oval", "rectangle"];
            setTableShape(shapes[Math.floor(Math.random() * shapes.length)]);
            return 100;
          }
          return prev + 2;
        });
      }, 50);
      return () => clearInterval(interval);
    }
  }, [isDetecting]);

  // Calculate player positions based on table shape
  const playerPositions = Array.from({ length: 10 }).map((_, i) => {
    if (tableShape === "circle") {
      // Circle: evenly distributed
      const angle = (i * 36) * (Math.PI / 180);
      return {
        x: 50 + 40 * Math.cos(angle - Math.PI / 2),
        y: 50 + 40 * Math.sin(angle - Math.PI / 2),
      };
    } else if (tableShape === "oval") {
      // Oval: ellipse distribution
      const angle = (i * 36) * (Math.PI / 180);
      return {
        x: 50 + 45 * Math.cos(angle - Math.PI / 2),
        y: 50 + 35 * Math.sin(angle - Math.PI / 2),
      };
    } else {
      // Rectangle: positions along perimeter
      const perimeter = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
      const corners = [
        // Top side
        { x: 20, y: 20 },
        { x: 35, y: 15 },
        { x: 50, y: 15 },
        { x: 65, y: 15 },
        { x: 80, y: 20 },
        // Bottom side
        { x: 80, y: 80 },
        { x: 65, y: 85 },
        { x: 50, y: 85 },
        { x: 35, y: 85 },
        { x: 20, y: 80 },
      ];
      return corners[i];
    }
  });

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      className="setup-page"
    >
      <div className="card glass large">
        <div className="card-header">
          <motion.h2
            className="card-title"
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
          >
            Определение стола
          </motion.h2>
          <motion.p
            className="card-subtitle"
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
          >
            Настройте область видимости игрового стола
          </motion.p>
        </div>

        <div className="card-body">
          {/* Enhanced video preview with glass effect */}
          <div className="video-preview glass">
            <AnimatePresence mode="wait">
              {/* Idle state */}
              {!isDetected && !isDetecting && (
                <motion.div
                  key="idle"
                  className="preview-placeholder"
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 1.05 }}
                  transition={{ duration: 0.3 }}
                >
                  <motion.div
                    className="placeholder-icon"
                    animate={{
                      scale: [1, 1.1, 1],
                      rotate: [0, 5, -5, 0],
                    }}
                    transition={{
                      duration: 3,
                      repeat: Infinity,
                      ease: "easeInOut",
                    }}
                  >
                    📹
                  </motion.div>
                  <motion.p
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.2 }}
                  >
                    Камера не активна
                  </motion.p>
                </motion.div>
              )}

              {/* Scanning state with symmetric animation */}
              {isDetecting && (
                <motion.div
                  key="scanning"
                  className="detecting-overlay"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                >
                  {/* Scanning rings (symmetric) */}
                  <div className="scan-rings">
                    {[0, 1, 2, 3].map((i) => (
                      <motion.div
                        key={i}
                        className="scan-ring"
                        style={{
                          width: `${30 + i * 20}%`,
                          height: `${30 + i * 20}%`,
                        }}
                        animate={{
                          scale: [1, 1.05, 1],
                          opacity: [0.3, 0.6, 0.3],
                          rotate: i % 2 === 0 ? [0, 360] : [360, 0],
                        }}
                        transition={{
                          duration: 3 + i * 0.5,
                          repeat: Infinity,
                          ease: "linear",
                          delay: i * 0.2,
                        }}
                      />
                    ))}
                  </div>

                  {/* Center scanning dot */}
                  <motion.div
                    className="scan-center"
                    animate={{
                      scale: [1, 1.3, 1],
                      boxShadow: [
                        "0 0 20px rgba(115, 194, 255, 0.5)",
                        "0 0 40px rgba(115, 194, 255, 0.8)",
                        "0 0 20px rgba(115, 194, 255, 0.5)",
                      ],
                    }}
                    transition={{
                      duration: 2,
                      repeat: Infinity,
                      ease: "easeInOut",
                    }}
                  />

                  {/* Radial scan lines (symmetric) */}
                  <svg className="scan-lines" viewBox="0 0 100 100">
                    {Array.from({ length: 12 }).map((_, i) => {
                      const angle = (i * 30) * (Math.PI / 180);
                      const x2 = 50 + 45 * Math.cos(angle);
                      const y2 = 50 + 45 * Math.sin(angle);
                      return (
                        <motion.line
                          key={i}
                          x1="50"
                          y1="50"
                          x2={x2}
                          y2={y2}
                          stroke="rgba(115, 194, 255, 0.3)"
                          strokeWidth="0.5"
                          initial={{ pathLength: 0, opacity: 0 }}
                          animate={{
                            pathLength: [0, 1, 0],
                            opacity: [0, 0.6, 0],
                          }}
                          transition={{
                            duration: 2,
                            repeat: Infinity,
                            delay: i * 0.15,
                            ease: "easeInOut",
                          }}
                        />
                      );
                    })}
                  </svg>

                  {/* Progress text */}
                  <motion.div
                    className="scan-progress"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                  >
                    <p className="scan-text">Сканирование стола...</p>
                    <div className="progress-bar glass">
                      <motion.div
                        className="progress-fill"
                        style={{ width: `${scanProgress}%` }}
                        transition={{ duration: 0.3 }}
                      />
                    </div>
                    <p className="scan-percent">{Math.round(scanProgress)}%</p>
                  </motion.div>
                </motion.div>
              )}

              {/* Detected state with adaptive table shape */}
              {isDetected && (
                <motion.div
                  key="detected"
                  className="detected-overlay"
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 1.1 }}
                >
                  {/* Table outline - adapts to detected shape */}
                  <svg className="table-svg" viewBox="0 0 100 100">
                    {tableShape === "circle" && (
                      <>
                        {/* Outer glow */}
                        <motion.circle
                          cx="50"
                          cy="50"
                          r="42"
                          fill="none"
                          stroke="rgba(34, 197, 94, 0.2)"
                          strokeWidth="8"
                          initial={{ pathLength: 0, opacity: 0 }}
                          animate={{ pathLength: 1, opacity: 1 }}
                          transition={{ duration: 1, ease: "easeOut" }}
                        />
                        {/* Main circle */}
                        <motion.circle
                          cx="50"
                          cy="50"
                          r="40"
                          fill="none"
                          stroke="rgba(34, 197, 94, 0.8)"
                          strokeWidth="2"
                          strokeDasharray="5 3"
                          initial={{ pathLength: 0, rotate: 0 }}
                          animate={{
                            pathLength: 1,
                            rotate: 360,
                          }}
                          transition={{
                            pathLength: { duration: 1.5, ease: "easeOut" },
                            rotate: { duration: 20, repeat: Infinity, ease: "linear" },
                          }}
                          style={{ transformOrigin: "50% 50%" }}
                        />
                      </>
                    )}
                    {tableShape === "oval" && (
                      <>
                        {/* Outer glow */}
                        <motion.ellipse
                          cx="50"
                          cy="50"
                          rx="47"
                          ry="37"
                          fill="none"
                          stroke="rgba(34, 197, 94, 0.2)"
                          strokeWidth="8"
                          initial={{ pathLength: 0, opacity: 0 }}
                          animate={{ pathLength: 1, opacity: 1 }}
                          transition={{ duration: 1, ease: "easeOut" }}
                        />
                        {/* Main oval */}
                        <motion.ellipse
                          cx="50"
                          cy="50"
                          rx="45"
                          ry="35"
                          fill="none"
                          stroke="rgba(34, 197, 94, 0.8)"
                          strokeWidth="2"
                          strokeDasharray="5 3"
                          initial={{ pathLength: 0, rotate: 0 }}
                          animate={{
                            pathLength: 1,
                            rotate: 360,
                          }}
                          transition={{
                            pathLength: { duration: 1.5, ease: "easeOut" },
                            rotate: { duration: 20, repeat: Infinity, ease: "linear" },
                          }}
                          style={{ transformOrigin: "50% 50%" }}
                        />
                      </>
                    )}
                    {tableShape === "rectangle" && (
                      <>
                        {/* Outer glow */}
                        <motion.rect
                          x="8"
                          y="13"
                          width="84"
                          height="74"
                          rx="8"
                          fill="none"
                          stroke="rgba(34, 197, 94, 0.2)"
                          strokeWidth="8"
                          initial={{ pathLength: 0, opacity: 0 }}
                          animate={{ pathLength: 1, opacity: 1 }}
                          transition={{ duration: 1, ease: "easeOut" }}
                        />
                        {/* Main rectangle */}
                        <motion.rect
                          x="10"
                          y="15"
                          width="80"
                          height="70"
                          rx="6"
                          fill="none"
                          stroke="rgba(34, 197, 94, 0.8)"
                          strokeWidth="2"
                          strokeDasharray="5 3"
                          initial={{ pathLength: 0 }}
                          animate={{ pathLength: 1 }}
                          transition={{ duration: 1.5, ease: "easeOut" }}
                        />
                      </>
                    )}
                  </svg>

                  {/* Player markers in perfect circle */}
                  {playerPositions.map((pos, i) => (
                    <motion.div
                      key={i}
                      className="player-marker glass"
                      style={{
                        position: "absolute",
                        left: `${pos.x}%`,
                        top: `${pos.y}%`,
                        transform: "translate(-50%, -50%)",
                      }}
                      initial={{ scale: 0, opacity: 0 }}
                      animate={{
                        scale: 1,
                        opacity: 1,
                      }}
                      transition={{
                        delay: 1 + i * 0.08,
                        type: "spring",
                        damping: 12,
                        stiffness: 200,
                      }}
                      whileHover={{
                        scale: 1.3,
                        zIndex: 10,
                        boxShadow: "0 0 20px rgba(115, 194, 255, 0.6)",
                      }}
                    >
                      <div className="marker-dot" />
                      <div className="marker-label">{i + 1}</div>
                      <motion.div
                        className="marker-pulse"
                        animate={{
                          scale: [1, 1.5, 1],
                          opacity: [0.5, 0, 0.5],
                        }}
                        transition={{
                          duration: 2,
                          repeat: Infinity,
                          delay: i * 0.2,
                        }}
                      />
                    </motion.div>
                  ))}

                  {/* Success message with table type */}
                  <motion.div
                    className="detected-badge glass"
                    initial={{ y: 20, opacity: 0 }}
                    animate={{ y: 0, opacity: 1 }}
                    transition={{ delay: 2, type: "spring" }}
                  >
                    <motion.div
                      className="success-icon"
                      animate={{ rotate: [0, 360] }}
                      transition={{ duration: 0.6, ease: "easeOut" }}
                    >
                      ✓
                    </motion.div>
                    <span>
                      {tableShape === "circle" && "Круглый стол"}
                      {tableShape === "oval" && "Овальный стол"}
                      {tableShape === "rectangle" && "Прямоугольный стол"}
                      {" • 10 позиций"}
                    </span>
                  </motion.div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* Controls */}
          <motion.div
            className="table-controls"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
          >
            {!isDetected && !isDetecting && (
              <GlassButton
                onClick={handleAutoDetect}
                size="large"
              >
                🔍 Автоопределение
              </GlassButton>
            )}

            {isDetected && (
              <motion.div
                className="success-message glass"
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 2.5 }}
              >
                <div className="success-icon">✓</div>
                <div>
                  <p className="success-title">Стол успешно настроен</p>
                  <p className="success-detail">Все позиции определены</p>
                </div>
              </motion.div>
            )}
          </motion.div>
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
