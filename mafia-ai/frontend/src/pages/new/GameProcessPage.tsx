import { useEffect, useMemo, useRef, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { Button } from '../../components/ui/Button';
import { GlassCard } from '../../components/ui/GlassCard';
import * as api from '../../services/api';
import './GameProcessPage.css';

type IncomingPlayer = { id?: number; name?: string };
type ProcessRouteState = {
  playerCount?: number;
  players?: IncomingPlayer[];
};

type KnownPlayer = {
  id: number;
  name: string;
};

type WsStatus = 'connecting' | 'open' | 'closed' | 'error';

type GestureWsHand = {
  track_id?: number;
  owner_id?: number | null;
  label?: string;
  gesture?: string;
  handedness?: string;
  fingers?: number;
  count?: number;
  center?: [number, number];
};

type GestureWsPayload = {
  type?: string;
  faces?: Array<{ id?: number | null; name?: string | null }>;
  hands?: GestureWsHand[];
};

type ActiveGesture = {
  key: string;
  ownerId: number | null;
  ownerName: string;
  label: string;
  handedness: string;
  fingers: number;
  centerX: number;
};

function normalizePlayers(incoming: IncomingPlayer[] | undefined, fallbackCount: number): KnownPlayer[] {
  if (Array.isArray(incoming) && incoming.length > 0) {
    return incoming.map((player, index) => {
      const id = Number(player.id) || index + 1;
      const name = player.name?.trim() || `Игрок ${id}`;
      return { id, name };
    });
  }

  return Array.from({ length: fallbackCount }, (_, index) => ({
    id: index + 1,
    name: `Игрок ${index + 1}`,
  }));
}

function getWsUrl(): string {
  const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
  const host = window.location.host;
  return `${protocol}://${host}/api/ws`;
}

function wsStatusText(status: WsStatus): string {
  if (status === 'open') {
    return 'Подключен';
  }
  if (status === 'connecting') {
    return 'Подключение';
  }
  if (status === 'error') {
    return 'Ошибка соединения';
  }
  return 'Отключен';
}

function wsStatusClass(status: WsStatus): string {
  if (status === 'open') {
    return 'status-tag--success';
  }
  if (status === 'connecting') {
    return 'status-tag--warn';
  }
  return 'status-tag--danger';
}

function displayGestureLabel(raw: string): string {
  const value = raw.trim().toLowerCase();
  if (!value) {
    return 'неизвестно';
  }

  const map: Record<string, string> = {
    thumb_up: 'мирный',
    like: 'мирный',
    thumb_down: 'мафия',
    dislike: 'мафия',
    ok: 'OK',
    ok_sign: 'OK',
    jambo: 'Если',
    call_me: 'Если',
    call: 'Если',
  };

  return map[value] || raw;
}

export function GameProcessPage() {
  const navigate = useNavigate();
  const location = useLocation();

  const routeState = (location.state as ProcessRouteState | null) || {};
  const fallbackCount = Number(routeState.playerCount) || 10;

  const [players, setPlayers] = useState<KnownPlayer[]>(() =>
    normalizePlayers(routeState.players, fallbackCount),
  );
  const [videoRunning, setVideoRunning] = useState(false);
  const [streamUrl, setStreamUrl] = useState('');
  const [wsStatus, setWsStatus] = useState<WsStatus>('connecting');
  const [activePlayerIds, setActivePlayerIds] = useState<number[]>([]);
  const [activeGestures, setActiveGestures] = useState<ActiveGesture[]>([]);
  const [unknownFaces, setUnknownFaces] = useState(0);
  const [error, setError] = useState('');

  const wsRef = useRef<WebSocket | null>(null);
  const playerNameByIdRef = useRef<Map<number, string>>(new Map());

  const activeIdSet = useMemo(() => new Set(activePlayerIds), [activePlayerIds]);

  const activePlayers = useMemo(() => {
    const byId = new Map(players.map((player) => [player.id, player.name]));
    return activePlayerIds.map((id) => ({
      id,
      name: byId.get(id) || `Игрок ${id}`,
    }));
  }, [activePlayerIds, players]);

  useEffect(() => {
    playerNameByIdRef.current = new Map(players.map((player) => [player.id, player.name]));
  }, [players]);

  async function reloadPlayers() {
    try {
      const response = await api.listPlayers();
      const loaded = (response.players || []).map((player) => ({
        id: player.id,
        name: player.name?.trim() || `Игрок ${player.id}`,
      }));
      if (loaded.length > 0) {
        setPlayers(loaded);
      }
    } catch (loadError) {
      console.error('Failed to load players:', loadError);
    }
  }

  async function restartVideo() {
    setError('');
    setActivePlayerIds([]);
    setActiveGestures([]);
    setUnknownFaces(0);

    try {
      await api.stopVideo().catch(() => undefined);
      const startRes = await api.startVideo();
      if (!startRes.ok) {
        throw new Error(startRes.error || 'Не удалось перезапустить видео');
      }

      setVideoRunning(true);
      setStreamUrl(`${api.getVideoStreamUrl()}?t=${Date.now()}`);
      await api.setVideoGestures(true).catch(() => undefined);
    } catch (restartError: any) {
      setVideoRunning(false);
      setError(restartError?.message || 'Не удалось запустить видеопоток');
    }
  }

  useEffect(() => {
    let disposed = false;

    function connectWs() {
      try {
        const ws = new WebSocket(getWsUrl());
        wsRef.current = ws;
        setWsStatus('connecting');

        ws.onopen = () => {
          if (!disposed) {
            setWsStatus('open');
          }
        };

        ws.onerror = () => {
          if (!disposed) {
            setWsStatus('error');
          }
        };

        ws.onclose = () => {
          if (!disposed) {
            setWsStatus((prev) => (prev === 'error' ? 'error' : 'closed'));
          }
        };

        ws.onmessage = (event) => {
          if (typeof event.data !== 'string') {
            return;
          }

          let payload: GestureWsPayload;
          try {
            payload = JSON.parse(event.data) as GestureWsPayload;
          } catch {
            return;
          }

          if (payload.type !== 'gesture') {
            return;
          }

          const faces = Array.isArray(payload.faces) ? payload.faces : [];
          const ids = new Set<number>();
          let unknown = 0;

          faces.forEach((face) => {
            if (typeof face.id === 'number' && Number.isFinite(face.id)) {
              ids.add(face.id);
              return;
            }
            unknown += 1;
          });

          setActivePlayerIds(Array.from(ids).sort((a, b) => a - b));
          setUnknownFaces(unknown);

          const hands = Array.isArray(payload.hands) ? payload.hands : [];
          const parsedHands = hands
            .map((hand, index): ActiveGesture => {
              const ownerId = typeof hand.owner_id === 'number' && Number.isFinite(hand.owner_id) ? hand.owner_id : null;
              const ownerName = ownerId
                ? playerNameByIdRef.current.get(ownerId) || `Игрок ${ownerId}`
                : 'не привязан';
              const rawLabel = (hand.gesture || hand.label || '').toString().trim();
              const label = displayGestureLabel(rawLabel);
              const handednessRaw = (hand.handedness || '').toString().trim().toLowerCase();
              const handedness = handednessRaw === 'left' ? 'левая' : handednessRaw === 'right' ? 'правая' : 'рука';
              const fingers = Number.isFinite(Number(hand.fingers))
                ? Number(hand.fingers)
                : Number.isFinite(Number(hand.count))
                  ? Number(hand.count)
                  : 0;
              const centerX = Array.isArray(hand.center) ? Number(hand.center[0] || 0) : index * 10;
              const trackId = typeof hand.track_id === 'number' ? hand.track_id : index;
              return {
                key: `${trackId}:${ownerId ?? 'u'}:${label}:${handedness}:${index}`,
                ownerId,
                ownerName,
                label,
                handedness,
                fingers,
                centerX,
              };
            })
            .sort((a, b) => a.centerX - b.centerX);

          setActiveGestures(parsedHands);
        };
      } catch (wsError) {
        console.error('WS init failed:', wsError);
        setWsStatus('error');
      }
    }

    async function bootstrap() {
      await reloadPlayers();

      try {
        const response = await api.startVideo();
        if (!response.ok) {
          throw new Error(response.error || 'Не удалось запустить видео');
        }

        if (disposed) {
          return;
        }

        setVideoRunning(true);
        setStreamUrl(`${api.getVideoStreamUrl()}?t=${Date.now()}`);
        await api.setVideoGestures(true).catch(() => undefined);
      } catch (startError: any) {
        if (!disposed) {
          setVideoRunning(false);
          setError(startError?.message || 'Не удалось запустить видеопоток');
        }
      }

      if (!disposed) {
        connectWs();
      }
    }

    void bootstrap();

    return () => {
      disposed = true;
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
      void api.setVideoGestures(true).catch(() => undefined);
      void api.stopVideo().catch(() => undefined);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="setup-shell">
      <div className="setup-container">
        <div className="setup-wizard">
          <GlassCard className="process-page__header">
            <div className="process-page__hero">
              <h1 className="process-page__heading">Игровой процесс</h1>
              <p className="process-page__subtitle">
                Видеопоток с камеры и распознанные игроки в реальном времени.
              </p>
            </div>
            <div className="process-page__status-row">
              <span className={`status-tag ${videoRunning ? 'status-tag--success' : 'status-tag--danger'}`}>
                Видео: {videoRunning ? 'запущено' : 'остановлено'}
              </span>
              <span className={`status-tag ${wsStatusClass(wsStatus)}`}>
                WS: {wsStatusText(wsStatus)}
              </span>
              <span className="status-tag status-tag--warn">Жестов: {activeGestures.length}</span>
              {unknownFaces > 0 && (
                <span className="status-tag status-tag--warn">Неизвестных лиц: {unknownFaces}</span>
              )}
            </div>
          </GlassCard>

          <div className="process-page__grid">
            <GlassCard className="process-page__camera-card">
              <div className="process-page__card-head">
                <h2 className="process-page__title">Камера</h2>
                <span className="process-page__muted">Поток с подсветкой распознанных игроков</span>
              </div>
              <div className="process-page__camera-frame">
                {streamUrl ? (
                  <img className="process-page__camera-image" src={streamUrl} alt="Camera stream" />
                ) : (
                  <div className="process-page__empty">Видеопоток недоступен</div>
                )}
              </div>
            </GlassCard>

            <GlassCard className="process-page__people-card">
              <div className="process-page__card-head">
                <h2 className="process-page__title">Кто в кадре сейчас</h2>
                <span className="process-page__muted">{activePlayers.length} активных</span>
              </div>

              <div className="process-page__active-chips">
                {activePlayers.length > 0 ? (
                  activePlayers.map((player) => (
                    <div key={player.id} className="process-page__chip">
                      <strong>{player.name}</strong>
                      <span>ID {player.id}</span>
                    </div>
                  ))
                ) : (
                  <div className="process-page__empty-small">Сейчас никого не распознано</div>
                )}
              </div>

              <div className="process-page__card-head">
                <h3 className="process-page__title">Жесты в кадре</h3>
                <span className="process-page__muted">{activeGestures.length}</span>
              </div>

              <div className="process-page__gestures">
                {activeGestures.length > 0 ? (
                  activeGestures.map((gesture) => (
                    <div key={gesture.key} className="process-page__gesture-item">
                      <div className="process-page__gesture-main">
                        <strong>{gesture.label}</strong>
                        <span>{gesture.handedness}</span>
                      </div>
                      <div className="process-page__gesture-meta">
                        <span>{gesture.ownerName}</span>
                        <span>пальцев: {gesture.fingers}</span>
                      </div>
                    </div>
                  ))
                ) : (
                  <div className="process-page__empty-small">Жесты пока не обнаружены</div>
                )}
              </div>

              <div className="process-page__roster">
                {players.map((player) => {
                  const isActive = activeIdSet.has(player.id);
                  return (
                    <div
                      key={player.id}
                      className={`process-page__roster-item ${isActive ? 'is-active' : ''}`.trim()}
                    >
                      <div className="process-page__roster-meta">
                        <strong>{player.name}</strong>
                        <span>ID {player.id}</span>
                      </div>
                      <span className={`status-tag ${isActive ? 'status-tag--success' : ''}`.trim()}>
                        {isActive ? 'в кадре' : 'вне кадра'}
                      </span>
                    </div>
                  );
                })}
              </div>
            </GlassCard>
          </div>

          {error && <div className="process-page__error">{error}</div>}

          <div className="setup-actions">
            <Button
              variant="secondary"
              size="lg"
              onClick={() =>
                navigate('/setup/table-detection', {
                  state: routeState,
                })
              }
            >
              Назад к калибровке
            </Button>
            <Button variant="secondary" size="lg" onClick={() => void restartVideo()}>
              Перезапустить видео
            </Button>
            <Button size="lg" onClick={() => void reloadPlayers()}>
              Обновить игроков
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}

