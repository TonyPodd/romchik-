import { useEffect, useMemo, useRef, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { FaceIDScanner } from '../../components/FaceIDScanner';
import { SetupStageHeader } from '../../components/SetupStageHeader';
import { Button } from '../../components/ui/Button';
import { GlassCard } from '../../components/ui/GlassCard';
import { Input } from '../../components/ui/Input';
import * as api from '../../services/api';
import './FaceRegistrationPage.css';

type ScanState = 'idle' | 'scanning' | 'success' | 'error';

type PlayerSlot = {
  id: number;
  name: string;
  registered: boolean;
  thumbUrl?: string;
  profileId?: number;
  source?: 'database' | 'new';
};

const DEFAULT_HINT = 'Смотрите прямо в камеру';

function createSlots(playerCount: number): PlayerSlot[] {
  return Array.from({ length: playerCount }, (_, index) => ({
    id: index + 1,
    name: '',
    registered: false,
  }));
}

function findNextPending(players: PlayerSlot[], currentIndex: number): number {
  for (let i = currentIndex + 1; i < players.length; i += 1) {
    if (!players[i].registered) {
      return i;
    }
  }
  for (let i = 0; i < players.length; i += 1) {
    if (!players[i].registered) {
      return i;
    }
  }
  return currentIndex;
}

export function FaceRegistrationPage() {
  const navigate = useNavigate();
  const location = useLocation();
  const playerCount = location.state?.playerCount || 10;

  const [players, setPlayers] = useState<PlayerSlot[]>(() => createSlots(playerCount));
  const [currentIndex, setCurrentIndex] = useState(0);
  const [scanState, setScanState] = useState<ScanState>('idle');
  const [scanProgress, setScanProgress] = useState(0);
  const [videoRunning, setVideoRunning] = useState(false);
  const [hint, setHint] = useState(DEFAULT_HINT);
  const [error, setError] = useState('');
  const [registeredPlayers, setRegisteredPlayers] = useState<api.Player[]>([]);
  const [showPlayerList, setShowPlayerList] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');

  const scanIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const snapBusyRef = useRef(false);

  const currentPlayer = players[currentIndex];
  const registeredCount = players.filter((player) => player.registered).length;
  const allRegistered = registeredCount === playerCount;

  const usedProfileIds = useMemo(() => {
    const ids = new Set<number>();
    players.forEach((player) => {
      if (typeof player.profileId === 'number') {
        ids.add(player.profileId);
      }
    });
    return ids;
  }, [players]);

  const filteredRegisteredPlayers = useMemo(() => {
    const query = searchQuery.trim().toLowerCase();
    if (!query) {
      return registeredPlayers;
    }
    return registeredPlayers.filter((player) => {
      const displayName = player.name?.trim() || `Игрок ${player.id}`;
      return displayName.toLowerCase().includes(query);
    });
  }, [registeredPlayers, searchQuery]);

  useEffect(() => {
    void startVideoStream();
    void loadRegisteredPlayers();

    return () => {
      clearScanLoop();
      void api.enrollCancel().catch(() => undefined);
      void api.setVideoGestures(true).catch(() => undefined);
      void api.setVideoFaceMatch(true).catch(() => undefined);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function clearScanLoop() {
    if (scanIntervalRef.current) {
      clearInterval(scanIntervalRef.current);
      scanIntervalRef.current = null;
    }
    snapBusyRef.current = false;
  }

  async function loadRegisteredPlayers() {
    try {
      const response = await api.listPlayers();
      setRegisteredPlayers(response.players || []);
    } catch (err) {
      console.error('Failed to load players:', err);
    }
  }

  async function startVideoStream() {
    try {
      const response = await api.startVideo();
      if (!response.ok) {
        throw new Error(response.error || 'Не удалось запустить видео');
      }
      setVideoRunning(true);
      await api.setVideoGestures(false);
      await api.setVideoFaceMatch(true);
    } catch (err: any) {
      console.error('Failed to start video:', err);
      setError(err?.message || 'Не удалось запустить видео');
    }
  }

  async function stopVideoStream() {
    try {
      clearScanLoop();
      await api.enrollCancel();
      await api.setVideoGestures(true);
      await api.setVideoFaceMatch(true);
      await api.stopVideo();
      setVideoRunning(false);
    } catch (err) {
      console.error('Failed to stop video:', err);
    }
  }

  function handleNameChange(value: string) {
    setPlayers((prev) =>
      prev.map((player, index) =>
        index === currentIndex ? { ...player, name: value } : player,
      ),
    );
    setError('');
  }

  async function handleOpenDatabase() {
    await loadRegisteredPlayers();
    setSearchQuery('');
    setShowPlayerList(true);
  }

  function assignFromDatabase(selected: api.Player) {
    const inUseByAnotherSlot = players.some(
      (player, index) => player.profileId === selected.id && index !== currentIndex,
    );
    if (inUseByAnotherSlot) {
      setError('Этот профиль уже назначен другому месту.');
      return;
    }

    const selectedName = selected.name?.trim() || `Игрок ${selected.id}`;
    const thumbUrl = api.getPlayerThumbUrl(selected);

    setPlayers((prev) => {
      const updated = prev.map((player, index) =>
        index === currentIndex
          ? {
              ...player,
              name: selectedName,
              registered: true,
              thumbUrl,
              profileId: selected.id,
              source: 'database',
            }
          : player,
      );
      const next = findNextPending(updated, currentIndex);
      setCurrentIndex(next);
      return updated;
    });

    setScanState('idle');
    setScanProgress(100);
    setHint(DEFAULT_HINT);
    setShowPlayerList(false);
    setError('');
  }

  function clearSlot(index: number) {
    setPlayers((prev) =>
      prev.map((player, slotIndex) =>
        slotIndex === index
          ? {
              ...player,
              registered: false,
              source: undefined,
              profileId: undefined,
              thumbUrl: undefined,
            }
          : player,
      ),
    );
    setCurrentIndex(index);
    setScanState('idle');
    setScanProgress(0);
    setHint(DEFAULT_HINT);
    setError('');
  }

  async function finishEnrollment(name: string) {
    try {
      const finishRes = await api.enrollFinish(name);
      if (!finishRes.ok) {
        const details = finishRes.details;
        const detailError =
          (details && typeof details.error === 'string' && details.error) ||
          (details && Array.isArray(details.sample_errors) && details.sample_errors[0]) ||
          '';
        const detailStats =
          details && typeof details.added === 'number' && typeof details.total === 'number'
            ? ` (${details.added}/${details.total})`
            : '';
        const message = detailError
          ? `Не удалось зарегистрировать лицо в CompreFace: ${detailError}${detailStats}`
          : finishRes.error || 'Не удалось завершить регистрацию';
        throw new Error(message);
      }

      const enrolledProfile = finishRes.player as Partial<api.Player> | undefined;
      const thumbUrl = enrolledProfile?.thumb
        ? api.getPlayerThumbUrl({ thumb: enrolledProfile.thumb, rev: Date.now() })
        : undefined;

      setPlayers((prev) => {
        const updated = prev.map((player, index) =>
          index === currentIndex
            ? {
                ...player,
                registered: true,
                source: 'new',
                profileId: enrolledProfile?.id,
                thumbUrl,
              }
            : player,
        );

        const next = findNextPending(updated, currentIndex);
        setTimeout(() => {
          setCurrentIndex(next);
          setScanState('idle');
          setScanProgress(0);
          setHint(DEFAULT_HINT);
        }, 450);
        return updated;
      });

      setScanState('success');
      setScanProgress(100);
      await loadRegisteredPlayers();
      await api.setVideoFaceMatch(true);
    } catch (err: any) {
      setScanState('error');
      setError(err?.message || 'Ошибка завершения регистрации');
      await api.setVideoFaceMatch(true);
    }
  }

  async function handleStartScan() {
    if (scanState === 'scanning') {
      return;
    }

    const name = currentPlayer.name.trim();
    if (!name) {
      setError('Введите имя игрока');
      return;
    }

    try {
      setScanState('scanning');
      setScanProgress(0);
      setHint(DEFAULT_HINT);
      setError('');
      await api.setVideoFaceMatch(false);

      const startRes = await api.enrollStart(name, 12);
      if (!startRes.ok) {
        throw new Error(startRes.error || 'Не удалось начать регистрацию');
      }

      clearScanLoop();
      scanIntervalRef.current = setInterval(async () => {
        if (snapBusyRef.current) {
          return;
        }
        snapBusyRef.current = true;
        try {
          const snapRes = await api.enrollSnap();
          if (!snapRes.ok) {
            return;
          }

          const statusRes = await api.enrollStatus();
          if (!statusRes.ok || statusRes.progress === undefined) {
            return;
          }

          setScanProgress(Math.round(statusRes.progress * 100));
          setHint(statusRes.hint || DEFAULT_HINT);

          if (
            typeof statusRes.count === 'number' &&
            typeof statusRes.target === 'number' &&
            statusRes.count >= statusRes.target
          ) {
            clearScanLoop();
            await finishEnrollment(name);
          }
        } catch (scanErr) {
          console.error('Snap error:', scanErr);
        } finally {
          snapBusyRef.current = false;
        }
      }, 220);
    } catch (err: any) {
      clearScanLoop();
      setScanState('error');
      setError(err?.message || 'Ошибка сканирования');
      await api.setVideoFaceMatch(true);
    }
  }

  async function handleCancelScan() {
    clearScanLoop();
    await api.enrollCancel();
    await api.setVideoFaceMatch(true);
    setScanState('idle');
    setScanProgress(0);
    setHint(DEFAULT_HINT);
  }

  async function handleDeletePlayer(playerId: number) {
    try {
      await api.deletePlayer(playerId);
      setPlayers((prev) =>
        prev.map((player) =>
          player.profileId === playerId
            ? {
                ...player,
                registered: false,
                source: undefined,
                profileId: undefined,
                thumbUrl: undefined,
              }
            : player,
        ),
      );
      await loadRegisteredPlayers();
    } catch (err) {
      console.error('Failed to delete player:', err);
    }
  }

  async function handleResetDatabase() {
    if (!confirm('Удалить все сохраненные лица из базы?')) {
      return;
    }
    try {
      await api.resetPlayers();
      setPlayers(createSlots(playerCount));
      setCurrentIndex(0);
      setScanState('idle');
      setScanProgress(0);
      setHint(DEFAULT_HINT);
      await loadRegisteredPlayers();
      setShowPlayerList(false);
    } catch (err) {
      console.error('Failed to reset players:', err);
    }
  }

  return (
    <div className="setup-shell">
      <div className="setup-container setup-container--with-stage">
        <GlassCard className="setup-stage-shell">
          <SetupStageHeader
            current="faces"
            title="Регистрация лиц игроков"
            subtitle="Выберите лица из базы или запишите новые профили для каждого места."
          />
        </GlassCard>

        <div className="setup-wizard">
          <div className="setup-grid setup-grid--face">
            <GlassCard className="face-reg__main">
              <div className="face-reg__current">
                <h2 className="face-reg__title">Место {currentPlayer.id}</h2>
                <span className={`status-tag ${currentPlayer.registered ? 'status-tag--success' : 'status-tag--warn'}`}>
                  {currentPlayer.registered ? 'Заполнено' : 'Ожидает'}
                </span>
              </div>

              <Input
                label="Имя игрока"
                value={currentPlayer.name}
                placeholder="Введите имя"
                onChange={(event) => handleNameChange(event.target.value)}
                disabled={scanState === 'scanning'}
                error={error}
              />

              <FaceIDScanner
                state={currentPlayer.registered ? 'success' : scanState}
                progress={scanProgress}
                videoUrl={videoRunning ? api.getVideoStreamUrl() : undefined}
              />

              <div className={`face-reg__hint ${scanState === 'error' ? 'is-error' : ''}`.trim()}>
                {scanState === 'scanning' ? hint : DEFAULT_HINT}
              </div>

              <div className="face-reg__actions">
                <Button
                  onClick={() => void handleStartScan()}
                  disabled={scanState === 'scanning' || !currentPlayer.name.trim()}
                  fullWidth
                >
                  Начать сканирование
                </Button>
                <Button
                  variant="secondary"
                  onClick={() => (scanState === 'scanning' ? void handleCancelScan() : void handleOpenDatabase())}
                  fullWidth
                >
                  {scanState === 'scanning' ? 'Отменить' : 'Выбрать из базы'}
                </Button>
              </div>
            </GlassCard>

            <GlassCard className="face-reg__slots">
              <div className="face-reg__slots-head">
                <div className="face-reg__slots-title">
                  <h2 className="face-reg__title">Игровые места</h2>
                  <span>{registeredCount} / {playerCount}</span>
                </div>
                <Button variant="secondary" size="sm" onClick={() => void handleOpenDatabase()}>
                  База лиц ({registeredPlayers.length})
                </Button>
              </div>
              <div className="face-reg__slot-list">
                {players.map((player, index) => (
                  <div
                    key={player.id}
                    className={`face-reg__slot-item ${index === currentIndex ? 'is-active' : ''} ${player.registered ? 'is-filled' : ''}`.trim()}
                    role="button"
                    tabIndex={scanState === 'scanning' ? -1 : 0}
                    onClick={() => {
                      if (scanState !== 'scanning') {
                        setCurrentIndex(index);
                      }
                    }}
                    onKeyDown={(event) => {
                      if (scanState === 'scanning') {
                        return;
                      }
                      if (event.key === 'Enter' || event.key === ' ') {
                        event.preventDefault();
                        setCurrentIndex(index);
                      }
                    }}
                  >
                    <div className="face-reg__slot-thumb">
                      {player.thumbUrl ? (
                        <img src={player.thumbUrl} alt={player.name || `Игрок ${player.id}`} />
                      ) : (
                        <span>{player.id}</span>
                      )}
                    </div>

                    <div className="face-reg__slot-meta">
                      <strong>{player.name || `Игрок ${player.id}`}</strong>
                      <span>
                        {player.registered
                          ? player.source === 'database'
                            ? 'Назначен из базы'
                            : 'Новый профиль'
                          : 'Не зарегистрирован'}
                      </span>
                    </div>

                    {player.registered && scanState !== 'scanning' && (
                      <Button
                        variant="secondary"
                        size="sm"
                        onClick={(event) => {
                          event.stopPropagation();
                          clearSlot(index);
                        }}
                      >
                        Сброс
                      </Button>
                    )}
                  </div>
                ))}
              </div>
            </GlassCard>
          </div>

          <div className="setup-actions">
            <Button
              variant="secondary"
              size="lg"
              onClick={() => {
                void stopVideoStream();
                navigate('/setup/players');
              }}
            >
              Назад
            </Button>
            <Button
              size="lg"
              disabled={!allRegistered}
              onClick={() => {
                void stopVideoStream();
                navigate('/setup/voice', {
                  state: {
                    playerCount,
                    players: players.map((player) => ({
                      id: player.id,
                      name: player.name?.trim() || `Игрок ${player.id}`,
                    })),
                  },
                });
              }}
            >
              {allRegistered ? 'Продолжить' : 'Завершите регистрацию всех игроков'}
            </Button>
          </div>
        </div>
      </div>

      {showPlayerList && (
        <div className="face-db" onClick={() => setShowPlayerList(false)}>
          <GlassCard className="face-db__panel" onClick={(event) => event.stopPropagation()}>
            <div className="face-db__header">
              <h2 className="feature-card__title">База лиц ({registeredPlayers.length})</h2>
              <div className="face-db__header-actions">
                <Button
                  variant="secondary"
                  onClick={() => void handleResetDatabase()}
                  className="face-db__danger-action"
                >
                  Очистить базу
                </Button>
                <Button variant="secondary" onClick={() => setShowPlayerList(false)}>
                  Закрыть
                </Button>
              </div>
            </div>

            <Input
              placeholder="Поиск по имени"
              value={searchQuery}
              onChange={(event) => setSearchQuery(event.target.value)}
            />

            <div className="face-db__grid">
              {filteredRegisteredPlayers.map((player) => {
                const displayName = player.name || `Игрок ${player.id}`;
                const thumbUrl = api.getPlayerThumbUrl(player);
                const alreadyUsed = usedProfileIds.has(player.id);
                const isCurrentSlot = currentPlayer.profileId === player.id;

                return (
                  <GlassCard key={player.id} className="face-db__card">
                    <div className="face-db__thumb">
                      {thumbUrl ? <img src={thumbUrl} alt={displayName} /> : <span>Нет превью</span>}
                    </div>
                    <div className="face-db__meta">
                      <strong>{displayName}</strong>
                      <span>ID профиля: {player.id}</span>
                    </div>
                    <Button
                      size="sm"
                      disabled={alreadyUsed && !isCurrentSlot}
                      onClick={() => assignFromDatabase(player)}
                      fullWidth
                    >
                      {alreadyUsed && !isCurrentSlot ? 'Уже назначен' : 'Выбрать'}
                    </Button>
                    <Button
                      size="sm"
                      variant="secondary"
                      onClick={() => void handleDeletePlayer(player.id)}
                      className="face-db__danger-action"
                      fullWidth
                    >
                      Удалить
                    </Button>
                  </GlassCard>
                );
              })}
            </div>

            {filteredRegisteredPlayers.length === 0 && (
              <div className="face-db__empty">В базе пока нет лиц под текущий фильтр.</div>
            )}
          </GlassCard>
        </div>
      )}
    </div>
  );
}
