import { motion, AnimatePresence } from 'framer-motion';
import { useEffect, useState } from 'react';

type ScannerState = 'idle' | 'scanning' | 'processing' | 'success' | 'error';

interface FaceIDScannerProps {
  state: ScannerState;
  progress?: number; // 0-100
  message?: string;
  size?: number;
}

export function FaceIDScanner({
  state,
  progress = 0,
  message,
  size = 300,
}: FaceIDScannerProps) {
  const [dots, setDots] = useState<Array<{ x: number; y: number; delay: number }>>([]);

  // Generate grid of dots for animation
  useEffect(() => {
    const gridSize = 10;
    const newDots = [];
    for (let i = 0; i < gridSize; i++) {
      for (let j = 0; j < gridSize; j++) {
        newDots.push({
          x: (i / (gridSize - 1)) * 100,
          y: (j / (gridSize - 1)) * 100,
          delay: (i + j) * 0.05,
        });
      }
    }
    setDots(newDots);
  }, []);

  const stateColors = {
    idle: 'rgba(255, 255, 255, 0.3)',
    scanning: '#6366f1',
    processing: '#8b5cf6',
    success: '#10b981',
    error: '#ef4444',
  };

  const stateIcons = {
    idle: '👤',
    scanning: '🔍',
    processing: '⚙️',
    success: '✓',
    error: '✗',
  };

  const color = stateColors[state];
  const icon = stateIcons[state];

  return (
    <div style={{
      position: 'relative',
      width: `${size}px`,
      height: `${size}px`,
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
    }}>
      {/* Outer scanning rings */}
      <AnimatePresence>
        {state === 'scanning' && (
          <>
            {[0, 1, 2].map((i) => (
              <motion.div
                key={i}
                style={{
                  position: 'absolute',
                  width: `${size}px`,
                  height: `${size}px`,
                  borderRadius: '50%',
                  border: `2px solid ${color}`,
                  opacity: 0,
                }}
                initial={{ scale: 0.8, opacity: 0 }}
                animate={{
                  scale: [0.8, 1.3],
                  opacity: [0.6, 0],
                }}
                exit={{ opacity: 0 }}
                transition={{
                  duration: 2,
                  repeat: Infinity,
                  delay: i * 0.4,
                  ease: 'easeOut',
                }}
              />
            ))}
          </>
        )}
      </AnimatePresence>

      {/* Progress ring */}
      <svg
        width={size}
        height={size}
        style={{ position: 'absolute' }}
      >
        {/* Background circle */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={(size / 2) - 10}
          fill="none"
          stroke="rgba(255, 255, 255, 0.1)"
          strokeWidth="4"
        />
        {/* Progress circle */}
        <motion.circle
          cx={size / 2}
          cy={size / 2}
          r={(size / 2) - 10}
          fill="none"
          stroke={color}
          strokeWidth="4"
          strokeLinecap="round"
          strokeDasharray={`${2 * Math.PI * ((size / 2) - 10)}`}
          initial={{ strokeDashoffset: 2 * Math.PI * ((size / 2) - 10) }}
          animate={{
            strokeDashoffset: (1 - progress / 100) * 2 * Math.PI * ((size / 2) - 10),
          }}
          transition={{ duration: 0.5 }}
          style={{
            transform: 'rotate(-90deg)',
            transformOrigin: '50% 50%',
            filter: state === 'scanning' ? 'drop-shadow(0 0 10px currentColor)' : 'none',
          }}
        />
      </svg>

      {/* Center content */}
      <div style={{
        position: 'relative',
        width: `${size * 0.7}px`,
        height: `${size * 0.7}px`,
        borderRadius: '50%',
        background: 'rgba(255, 255, 255, 0.05)',
        backdropFilter: 'blur(10px)',
        border: `2px solid ${color}`,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        overflow: 'hidden',
        boxShadow: state === 'scanning' ? `0 0 40px ${color}` : 'none',
        transition: 'all 0.5s ease',
      }}>
        {/* Animated dots grid */}
        <AnimatePresence>
          {state === 'scanning' && (
            <div style={{
              position: 'absolute',
              width: '80%',
              height: '80%',
            }}>
              {dots.map((dot, i) => (
                <motion.div
                  key={i}
                  initial={{ scale: 0, opacity: 0 }}
                  animate={{
                    scale: [0, 1, 0.8],
                    opacity: [0, 1, 0.6],
                  }}
                  exit={{ scale: 0, opacity: 0 }}
                  transition={{
                    duration: 1.5,
                    delay: dot.delay,
                    repeat: Infinity,
                    repeatDelay: 1,
                  }}
                  style={{
                    position: 'absolute',
                    left: `${dot.x}%`,
                    top: `${dot.y}%`,
                    width: '6px',
                    height: '6px',
                    borderRadius: '50%',
                    background: color,
                    transform: 'translate(-50%, -50%)',
                    boxShadow: `0 0 10px ${color}`,
                  }}
                />
              ))}
            </div>
          )}
        </AnimatePresence>

        {/* Icon */}
        <motion.div
          key={state}
          initial={{ scale: 0, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          exit={{ scale: 0, opacity: 0 }}
          transition={{ type: 'spring', stiffness: 300, damping: 20 }}
          style={{
            fontSize: size * 0.2,
            filter: `drop-shadow(0 0 20px ${color})`,
          }}
        >
          {icon}
        </motion.div>

        {/* Scanning line */}
        <AnimatePresence>
          {state === 'scanning' && (
            <motion.div
              initial={{ y: -size * 0.35 }}
              animate={{ y: size * 0.35 }}
              exit={{ opacity: 0 }}
              transition={{
                duration: 2,
                repeat: Infinity,
                ease: 'linear',
              }}
              style={{
                position: 'absolute',
                width: '100%',
                height: '2px',
                background: `linear-gradient(90deg, transparent, ${color}, transparent)`,
                boxShadow: `0 0 20px ${color}`,
              }}
            />
          )}
        </AnimatePresence>

        {/* Processing spinner */}
        <AnimatePresence>
          {state === 'processing' && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1, rotate: 360 }}
              exit={{ opacity: 0 }}
              transition={{
                rotate: {
                  duration: 1,
                  repeat: Infinity,
                  ease: 'linear',
                },
              }}
              style={{
                position: 'absolute',
                width: '60%',
                height: '60%',
                border: `3px solid transparent`,
                borderTopColor: color,
                borderRightColor: color,
                borderRadius: '50%',
              }}
            />
          )}
        </AnimatePresence>

        {/* Success/Error pulse */}
        <AnimatePresence>
          {(state === 'success' || state === 'error') && (
            <motion.div
              initial={{ scale: 1, opacity: 0.6 }}
              animate={{ scale: 1.5, opacity: 0 }}
              transition={{ duration: 0.6 }}
              style={{
                position: 'absolute',
                width: '100%',
                height: '100%',
                borderRadius: '50%',
                background: color,
              }}
            />
          )}
        </AnimatePresence>
      </div>

      {/* Message */}
      <AnimatePresence>
        {message && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            style={{
              position: 'absolute',
              bottom: -40,
              left: '50%',
              transform: 'translateX(-50%)',
              whiteSpace: 'nowrap',
              color: color,
              fontSize: '0.875rem',
              fontWeight: 600,
              textAlign: 'center',
            }}
          >
            {message}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
