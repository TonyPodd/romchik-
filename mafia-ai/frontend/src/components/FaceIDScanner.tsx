import { useEffect, useState, useRef } from 'react';

type ScanState = 'idle' | 'scanning' | 'success' | 'error';

interface FaceIDScannerProps {
  state: ScanState;
  progress?: number; // 0-100
  videoUrl?: string;
}

export function FaceIDScanner({ state, progress = 0, videoUrl }: FaceIDScannerProps) {
  const imgRef = useRef<HTMLImageElement>(null);
  const [streamUrl, setStreamUrl] = useState<string>('');

  // MJPEG stream - add timestamp to force browser to load
  useEffect(() => {
    if (videoUrl) {
      // Add timestamp to bypass browser caching and force stream load
      const url = `${videoUrl}?t=${Date.now()}`;
      console.log('[FaceIDScanner] Setting stream URL:', url);
      setStreamUrl(url);
    } else {
      setStreamUrl('');
    }
  }, [videoUrl]);

  const getStateColor = () => {
    switch (state) {
      case 'scanning':
        return '#10b981'; // зеленый для сканирования
      case 'success':
        return '#10b981';
      case 'error':
        return '#ef4444';
      default:
        return '#64748b';
    }
  };

  return (
    <div style={{
      position: 'relative',
      width: '100%',
      background: '#0f1117',
      borderRadius: '1rem',
      overflow: 'visible',
    }}>
      {/* Video stream */}
      {streamUrl ? (
        <img
          ref={imgRef}
          src={streamUrl}
          alt="Camera stream"
          style={{
            display: 'block',
            width: '100%',
            height: 'auto',
            borderRadius: '1rem',
          }}
          onLoad={() => console.log('[FaceIDScanner] Stream loaded')}
          onError={(e) => console.error('[FaceIDScanner] Stream error:', e)}
        />
      ) : (
        <div style={{
          fontSize: '4rem',
          opacity: 0.2,
        }}>
          👤
        </div>
      )}

      {/* Minimal overlay */}
      <div style={{
        position: 'absolute',
        inset: 0,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
      }}>
        {/* Center progress circle - large and dynamic */}
        {state === 'scanning' && (
          <div style={{
            position: 'relative',
            width: '200px',
            height: '200px',
          }}>
            {/* Pulsating glow */}
            <div
              style={{
                position: 'absolute',
                inset: 0,
                borderRadius: '50%',
                background: `radial-gradient(circle, ${getStateColor()}40, transparent 70%)`,
                animation: 'pulse 2s ease-in-out infinite',
              }}
            />
            <svg
              width="200"
              height="200"
              style={{
                position: 'absolute',
                top: 0,
                left: 0,
                transform: 'rotate(-90deg)',
              }}
            >
              {/* Background circle */}
              <circle
                cx="100"
                cy="100"
                r="95"
                fill="none"
                stroke="rgba(255, 255, 255, 0.1)"
                strokeWidth="3"
              />
              {/* Progress circle with glow */}
              <circle
                cx="100"
                cy="100"
                r="95"
                fill="none"
                stroke={getStateColor()}
                strokeWidth="4"
                strokeLinecap="round"
                strokeDasharray={`${2 * Math.PI * 95}`}
                strokeDashoffset={`${2 * Math.PI * 95 * (1 - progress / 100)}`}
                style={{
                  transition: 'stroke-dashoffset 0.3s ease',
                  filter: `drop-shadow(0 0 8px ${getStateColor()})`,
                }}
              />
            </svg>

            {/* Center content */}
            <div style={{
              position: 'absolute',
              inset: 0,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              flexDirection: 'column',
              gap: '0.5rem',
            }}>
              {/* Animated scanning icon */}
              <div style={{
                fontSize: '2rem',
                color: getStateColor(),
                animation: 'breathe 2s ease-in-out infinite',
              }}>
                👤
              </div>
              {/* Progress percentage */}
              <div style={{
                color: getStateColor(),
                fontSize: '1.25rem',
                fontWeight: 600,
                textShadow: `0 0 10px ${getStateColor()}80`,
              }}>
                {Math.round(progress)}%
              </div>
            </div>
          </div>
        )}

        {/* Success/Error icons */}
        {state === 'success' && (
          <div style={{
            fontSize: '5rem',
            color: '#10b981',
          }}>✓</div>
        )}
        {state === 'error' && (
          <div style={{
            fontSize: '5rem',
            color: '#ef4444',
          }}>✕</div>
        )}
      </div>

    </div>
  );
}
