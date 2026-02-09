import { useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { SetupStageHeader } from '../../components/SetupStageHeader';
import { Button } from '../../components/ui/Button';
import { GlassCard } from '../../components/ui/GlassCard';
import './TableDetectionPage.css';

type DetectionState = 'idle' | 'detecting' | 'detected';
type ProcessRouteState = {
  playerCount?: number;
  players?: Array<{ id?: number; name?: string }>;
};

export function TableDetectionPage() {
  const navigate = useNavigate();
  const location = useLocation();
  const [detectionState, setDetectionState] = useState<DetectionState>('idle');
  const [progress, setProgress] = useState(0);
  const processState = ((location.state as ProcessRouteState | null) || {}) satisfies ProcessRouteState;

  function handleStartDetection() {
    setDetectionState('detecting');
    setProgress(0);

    const interval = setInterval(() => {
      setProgress((prev) => {
        if (prev >= 100) {
          clearInterval(interval);
          setDetectionState('detected');
          return 100;
        }
        return prev + 2;
      });
    }, 50);
  }

  function handleRetry() {
    setDetectionState('idle');
    setProgress(0);
  }

  return (
    <div className="setup-shell">
      <div className="setup-container setup-container--with-stage">
        <GlassCard className="setup-stage-shell">
          <SetupStageHeader
            current="table"
            title="РћРїСЂРµРґРµР»РµРЅРёРµ СЃС‚РѕР»Р°"
            subtitle="РџСЂРѕРІРµСЂСЊС‚Рµ РєР°РґСЂ Рё РїРѕРґС‚РІРµСЂРґРёС‚Рµ РёРіСЂРѕРІСѓСЋ РѕР±Р»Р°СЃС‚СЊ РїРµСЂРµРґ РЅР°С‡Р°Р»РѕРј РїР°СЂС‚РёРё."
          />
        </GlassCard>

        <div className="setup-wizard">
          <div className="setup-grid setup-grid--table">
            <GlassCard className="table-page__camera">
              <h2 className="table-page__title">РљР°РјРµСЂР°</h2>
              <div className="table-stage">
                <div className={`table-page__state table-page__state--${detectionState}`}>
                  {detectionState === 'detected' ? 'вњ“' : 'вЊѓ'}
                </div>
                {detectionState === 'detecting' && <div className="table-page__scan-line" />}
              </div>

              <div className="setup-progress">
                <div className="setup-progress__row">
                  <span>РђРЅР°Р»РёР· РёР·РѕР±СЂР°Р¶РµРЅРёСЏ</span>
                  <span>{Math.round(progress)}%</span>
                </div>
                <progress className="setup-progress__native" value={progress} max={100} />
              </div>

              <div className="table-page__actions">
                {detectionState === 'idle' && (
                  <Button onClick={handleStartDetection} fullWidth>
                    РќР°С‡Р°С‚СЊ РѕРїСЂРµРґРµР»РµРЅРёРµ
                  </Button>
                )}
                {detectionState === 'detecting' && (
                  <Button variant="secondary" disabled fullWidth>
                    РРґРµС‚ РѕР±СЂР°Р±РѕС‚РєР°...
                  </Button>
                )}
                {detectionState === 'detected' && (
                  <Button variant="secondary" onClick={handleRetry} fullWidth>
                    РџРѕРІС‚РѕСЂРёС‚СЊ
                  </Button>
                )}
              </div>
            </GlassCard>

            <GlassCard className="table-page__info">
              <h2 className="table-page__title">РџСЂРѕРІРµСЂРєР° РєР°С‡РµСЃС‚РІР°</h2>
              <ol className="table-page__steps">
                <li>РџРѕСЃС‚Р°РІСЊС‚Рµ РєР°РјРµСЂСѓ С‚Р°Рє, С‡С‚РѕР±С‹ СЃС‚РѕР» Р·Р°РЅРёРјР°Р» С†РµРЅС‚СЂ РєР°РґСЂР°.</li>
                <li>РР·Р±РµРіР°Р№С‚Рµ СЃРёР»СЊРЅРѕР№ Р·Р°СЃРІРµС‚РєРё Рё Р¶РµСЃС‚РєРёС… С‚РµРЅРµР№ РЅР° РїРѕРІРµСЂС…РЅРѕСЃС‚Рё.</li>
                <li>РџРѕСЃР»Рµ РѕР±РЅР°СЂСѓР¶РµРЅРёСЏ РїСЂРѕРІРµСЂСЊС‚Рµ РєРѕСЂСЂРµРєС‚РЅРѕСЃС‚СЊ РєРѕРЅС‚СѓСЂР°.</li>
              </ol>
              {detectionState === 'detected' && (
                <div className="status-tag status-tag--success table-page__ok">
                  РЎС‚РѕР» РѕРїСЂРµРґРµР»РµРЅ Рё РіРѕС‚РѕРІ Рє СЃР»РµРґСѓСЋС‰РµРјСѓ С€Р°РіСѓ.
                </div>
              )}
            </GlassCard>
          </div>

          <div className="setup-actions">
            <Button variant="secondary" size="lg" onClick={() => navigate('/setup/voice')}>
              РќР°Р·Р°Рґ
            </Button>
            <Button
              size="lg"
              disabled={detectionState !== 'detected'}
              onClick={() =>
                navigate('/game/process', {
                  state: processState,
                })
              }
            >
              {detectionState === 'detected' ? 'РџРµСЂРµР№С‚Рё Рє РїСЂРѕС†РµСЃСЃСѓ' : 'РћРїСЂРµРґРµР»РёС‚Рµ СЃС‚РѕР»'}
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}

