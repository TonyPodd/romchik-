// src/components/FaceIDScanner.tsx
import React, { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import "./FaceIDScanner.css";

interface FaceIDScannerProps {
  onComplete?: () => void;
  playerName?: string;
  autoStart?: boolean;
}

type ScanState = "idle" | "scanning" | "processing" | "success" | "error";

export const FaceIDScanner: React.FC<FaceIDScannerProps> = ({
  onComplete,
  playerName = "Игрок",
  autoStart = false,
}) => {
  const [state, setState] = useState<ScanState>("idle");
  const [progress, setProgress] = useState(0);
  const [scanAngle, setScanAngle] = useState(0);

  // Auto start scanning
  useEffect(() => {
    if (autoStart) {
      startScan();
    }
  }, [autoStart]);

  // Scanning animation
  useEffect(() => {
    if (state === "scanning") {
      const interval = setInterval(() => {
        setProgress((prev) => {
          if (prev >= 100) {
            clearInterval(interval);
            setState("processing");
            return 100;
          }
          return prev + 2;
        });

        setScanAngle((prev) => (prev + 10) % 360);
      }, 50);

      return () => clearInterval(interval);
    }
  }, [state]);

  // Processing -> Success
  useEffect(() => {
    if (state === "processing") {
      const timeout = setTimeout(() => {
        setState("success");
        onComplete?.();
      }, 1500);

      return () => clearTimeout(timeout);
    }
  }, [state, onComplete]);

  const startScan = () => {
    setProgress(0);
    setScanAngle(0);
    setState("scanning");
  };

  const reset = () => {
    setProgress(0);
    setScanAngle(0);
    setState("idle");
  };

  return (
    <div className="faceid-scanner">
      {/* Animated rings */}
      <div className="scanner-rings">
        {/* Outer glow ring */}
        <motion.div
          className="ring ring-glow"
          animate={{
            scale: state === "scanning" ? [1, 1.1, 1] : 1,
            opacity: state === "scanning" ? [0.3, 0.5, 0.3] : 0.2,
          }}
          transition={{
            duration: 2,
            repeat: state === "scanning" ? Infinity : 0,
            ease: "easeInOut",
          }}
        />

        {/* Main scanning ring */}
        <motion.div
          className={`ring ring-main state-${state}`}
          animate={{
            rotate: state === "scanning" ? scanAngle : 0,
          }}
          transition={{
            duration: 0.05,
            ease: "linear",
          }}
        >
          {/* Scanning line */}
          {state === "scanning" && (
            <motion.div
              className="scan-line"
              initial={{ scaleY: 0 }}
              animate={{ scaleY: 1 }}
              transition={{ duration: 0.3 }}
            />
          )}

          {/* Progress arcs */}
          <svg className="progress-svg" viewBox="0 0 200 200">
            <circle
              className="progress-bg"
              cx="100"
              cy="100"
              r="90"
              fill="none"
              stroke="currentColor"
              strokeWidth="3"
              opacity="0.1"
            />
            <motion.circle
              className="progress-fg"
              cx="100"
              cy="100"
              r="90"
              fill="none"
              stroke="currentColor"
              strokeWidth="3"
              strokeDasharray={`${2 * Math.PI * 90}`}
              initial={{ strokeDashoffset: 2 * Math.PI * 90 }}
              animate={{
                strokeDashoffset: 2 * Math.PI * 90 * (1 - progress / 100),
              }}
              transition={{ duration: 0.1 }}
            />
          </svg>

          {/* Center face icon */}
          <div className="face-icon-container">
            <AnimatePresence mode="wait">
              {state === "idle" && (
                <motion.div
                  key="idle"
                  className="face-icon"
                  initial={{ scale: 0, opacity: 0 }}
                  animate={{ scale: 1, opacity: 1 }}
                  exit={{ scale: 0, opacity: 0 }}
                  transition={{ duration: 0.3 }}
                >
                  <svg
                    viewBox="0 0 100 100"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="3"
                  >
                    <circle cx="50" cy="50" r="35" />
                    <circle cx="38" cy="42" r="3" fill="currentColor" />
                    <circle cx="62" cy="42" r="3" fill="currentColor" />
                    <path d="M 35 60 Q 50 70 65 60" strokeLinecap="round" />
                  </svg>
                </motion.div>
              )}

              {state === "scanning" && (
                <motion.div
                  key="scanning"
                  className="face-icon scanning"
                  initial={{ scale: 0, opacity: 0 }}
                  animate={{ scale: 1, opacity: 1 }}
                  exit={{ scale: 0, opacity: 0 }}
                >
                  <div className="grid-overlay">
                    {Array.from({ length: 25 }).map((_, i) => (
                      <motion.div
                        key={i}
                        className="grid-dot"
                        animate={{
                          scale: [0, 1, 0],
                          opacity: [0, 1, 0],
                        }}
                        transition={{
                          duration: 2,
                          delay: i * 0.05,
                          repeat: Infinity,
                          repeatDelay: 1,
                        }}
                      />
                    ))}
                  </div>
                </motion.div>
              )}

              {state === "processing" && (
                <motion.div
                  key="processing"
                  className="face-icon processing"
                  initial={{ scale: 0, rotate: -180 }}
                  animate={{ scale: 1, rotate: 0 }}
                  exit={{ scale: 0 }}
                  transition={{ duration: 0.5, type: "spring" }}
                >
                  <motion.div
                    className="spinner"
                    animate={{ rotate: 360 }}
                    transition={{
                      duration: 1,
                      repeat: Infinity,
                      ease: "linear",
                    }}
                  >
                    <svg
                      viewBox="0 0 100 100"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="4"
                    >
                      <circle
                        cx="50"
                        cy="50"
                        r="30"
                        strokeDasharray="60 150"
                        strokeLinecap="round"
                      />
                    </svg>
                  </motion.div>
                </motion.div>
              )}

              {state === "success" && (
                <motion.div
                  key="success"
                  className="face-icon success"
                  initial={{ scale: 0 }}
                  animate={{ scale: [0, 1.2, 1] }}
                  exit={{ scale: 0 }}
                  transition={{ duration: 0.5, type: "spring" }}
                >
                  <motion.svg
                    viewBox="0 0 100 100"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="6"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  >
                    <motion.path
                      d="M 20 50 L 40 70 L 80 30"
                      initial={{ pathLength: 0 }}
                      animate={{ pathLength: 1 }}
                      transition={{ duration: 0.5, delay: 0.2 }}
                    />
                  </motion.svg>
                </motion.div>
              )}

              {state === "error" && (
                <motion.div
                  key="error"
                  className="face-icon error"
                  initial={{ scale: 0 }}
                  animate={{ scale: [0, 1.1, 1] }}
                  exit={{ scale: 0 }}
                >
                  <svg
                    viewBox="0 0 100 100"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="6"
                    strokeLinecap="round"
                  >
                    <path d="M 30 30 L 70 70" />
                    <path d="M 70 30 L 30 70" />
                  </svg>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </motion.div>

        {/* Inner highlight ring */}
        <motion.div
          className="ring ring-highlight"
          animate={{
            scale: state === "success" ? [1, 1.1, 1] : 1,
            opacity: state === "success" ? [0, 0.8, 0] : 0,
          }}
          transition={{
            duration: 0.6,
          }}
        />
      </div>

      {/* Status text */}
      <motion.div
        className="scanner-status"
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
      >
        <AnimatePresence mode="wait">
          {state === "idle" && (
            <motion.div
              key="idle"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
            >
              <p className="status-title">Готов к сканированию</p>
              <p className="status-subtitle">Посмотрите в камеру</p>
            </motion.div>
          )}

          {state === "scanning" && (
            <motion.div
              key="scanning"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
            >
              <p className="status-title">Сканирование лица...</p>
              <p className="status-subtitle">{progress}%</p>
            </motion.div>
          )}

          {state === "processing" && (
            <motion.div
              key="processing"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
            >
              <p className="status-title">Обработка...</p>
              <p className="status-subtitle">Создание профиля</p>
            </motion.div>
          )}

          {state === "success" && (
            <motion.div
              key="success"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
            >
              <p className="status-title success">✓ {playerName} добавлен!</p>
              <p className="status-subtitle">Лицо успешно зарегистрировано</p>
            </motion.div>
          )}

          {state === "error" && (
            <motion.div
              key="error"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
            >
              <p className="status-title error">Ошибка сканирования</p>
              <p className="status-subtitle">Попробуйте еще раз</p>
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>

      {/* Action button */}
      {state === "idle" && (
        <motion.button
          className="scan-button glass-btn"
          onClick={startScan}
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          Начать сканирование
        </motion.button>
      )}

      {state === "success" && (
        <motion.button
          className="scan-button glass-btn"
          onClick={reset}
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          Сканировать еще
        </motion.button>
      )}
    </div>
  );
};
