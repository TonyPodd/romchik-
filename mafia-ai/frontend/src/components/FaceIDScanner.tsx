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
        return '#4f46e5';
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
        {/* Simple corner brackets - only when scanning */}
        {state === 'scanning' && (
          <>
            <div style={{
              position: 'absolute',
              top: '15%',
              left: '15%',
              width: '50px',
              height: '50px',
              borderTop: `2px solid ${getStateColor()}`,
              borderLeft: `2px solid ${getStateColor()}`,
              opacity: 0.4,
            }} />
            <div style={{
              position: 'absolute',
              top: '15%',
              right: '15%',
              width: '50px',
              height: '50px',
              borderTop: `2px solid ${getStateColor()}`,
              borderRight: `2px solid ${getStateColor()}`,
              opacity: 0.4,
            }} />
            <div style={{
              position: 'absolute',
              bottom: '15%',
              left: '15%',
              width: '50px',
              height: '50px',
              borderBottom: `2px solid ${getStateColor()}`,
              borderLeft: `2px solid ${getStateColor()}`,
              opacity: 0.4,
            }} />
            <div style={{
              position: 'absolute',
              bottom: '15%',
              right: '15%',
              width: '50px',
              height: '50px',
              borderBottom: `2px solid ${getStateColor()}`,
              borderRight: `2px solid ${getStateColor()}`,
              opacity: 0.4,
            }} />
          </>
        )}

        {/* Center progress circle - minimal */}
        {state === 'scanning' && (
          <div style={{
            position: 'relative',
            width: '80px',
            height: '80px',
          }}>
            <svg
              width="80"
              height="80"
              style={{
                position: 'absolute',
                top: 0,
                left: 0,
                transform: 'rotate(-90deg)',
              }}
            >
              <circle
                cx="40"
                cy="40"
                r="35"
                fill="none"
                stroke="rgba(255, 255, 255, 0.1)"
                strokeWidth="2"
              />
              <circle
                cx="40"
                cy="40"
                r="35"
                fill="none"
                stroke={getStateColor()}
                strokeWidth="2"
                strokeLinecap="round"
                strokeDasharray={`${2 * Math.PI * 35}`}
                strokeDashoffset={`${2 * Math.PI * 35 * (1 - progress / 100)}`}
                style={{
                  transition: 'stroke-dashoffset 0.3s ease',
                }}
              />
            </svg>

            {/* Center dot */}
            <div style={{
              position: 'absolute',
              inset: 0,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}>
              <div style={{
                width: '8px',
                height: '8px',
                background: getStateColor(),
                borderRadius: '50%',
                opacity: 0.6,
              }} />
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

      {/* Progress text */}
      {state === 'scanning' && (
        <div style={{
          position: 'absolute',
          bottom: '1rem',
          left: 0,
          right: 0,
          textAlign: 'center',
          color: '#94a3b8',
          fontSize: '0.875rem',
          fontWeight: 500,
        }}>
          {Math.round(progress)}%
        </div>
      )}
    </div>
  );
}
