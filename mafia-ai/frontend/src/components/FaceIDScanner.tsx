import { CSSProperties, useEffect, useState } from 'react';
import './FaceIDScanner.css';

type ScanState = 'idle' | 'scanning' | 'success' | 'error';

interface FaceIDScannerProps {
  state: ScanState;
  progress?: number; // 0-100
  videoUrl?: string;
}

export function FaceIDScanner({ state, progress = 0, videoUrl }: FaceIDScannerProps) {
  const [streamUrl, setStreamUrl] = useState<string>('');

  useEffect(() => {
    if (!videoUrl) {
      setStreamUrl('');
      return;
    }
    setStreamUrl(`${videoUrl}?t=${Date.now()}`);
  }, [videoUrl]);

  const progressValue = Math.max(0, Math.min(100, Math.round(progress)));
  const stateLabel =
    state === 'success' ? 'Лицо сохранено' :
    state === 'error' ? 'Ошибка сканирования' :
    state === 'scanning' ? 'Идет сканирование' :
    'Подготовка';

  const scannerStyle = {
    '--scan-progress': `${progressValue}%`,
  } as CSSProperties;

  return (
    <div className={`face-scanner face-scanner--${state}`} style={scannerStyle}>
      {streamUrl ? (
        <img className="face-scanner__video" src={streamUrl} alt="Camera stream" />
      ) : (
        <div className="face-scanner__empty">Видеопоток недоступен</div>
      )}

      <div className="face-scanner__veil" />
      <div className="face-scanner__center">
        <div className="face-scanner__halo" />
        <div className="face-scanner__frame">
          <div className="face-scanner__grid" />
          {state === 'scanning' && <div className="face-scanner__scan-beam" />}
        </div>
        {(state === 'success' || state === 'error') && (
          <div className="face-scanner__result">
            {state === 'success' ? '✓' : '✕'}
          </div>
        )}
      </div>

      <div className="face-scanner__bottom">
        <div className="face-scanner__meta">
          <span className={`status-tag ${state === 'success' ? 'status-tag--success' : ''} ${state === 'error' ? 'status-tag--danger' : ''}`}>
            {stateLabel}
          </span>
          <span>{progressValue}%</span>
        </div>
        <progress className="face-scanner__progress" value={progressValue} max={100} />
      </div>
    </div>
  );
}
