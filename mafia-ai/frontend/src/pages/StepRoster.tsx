import React, { useEffect, useRef, useState } from "react";
import { GlassButton } from "@/components/GlassButton";
import FaceEnrollRing from "@/components/FaceEnrollRing";
import {
  enrollStart,
  enrollStep,
  enrollFinish,
  enrollCancel,
  listPlayers,
  deletePlayer,
} from "@/api";

type Player = { id:number; name:string; thumb:string; rev?:number };


export function StepRoster(props: { value: 8 | 10; onChange: (n: 8 | 10) => void; players: 8 | 10 }) {
  // список зарегистрированных на сервере
  const [list, setList] = useState<Player[]>([]);
  const [busy, setBusy] = useState(false);

  // состояние энролла
  const [isRec, setIsRec] = useState(false);
  const [progress, setProgress] = useState(0); // 0..1
  const [status, setStatus] = useState<"idle" | "recording" | "done" | "error">("idle");
  const [hint, setHint] = useState<string>("");

  // имя и целевой объём кадров
  const [newName, setNewName] = useState("");
  const [target, setTarget] = useState<number>(8);

  const timerRef = useRef<number | undefined>(undefined);
  const targetRef = useRef<number>(8);

  // загрузка списка игроков
  const load = async () => {
    try {
      const r = await listPlayers();
      setList(Array.isArray(r?.players) ? r.players : []);
    } catch {
      setList([]);
    }
  };
  useEffect(() => {
    load();
  }, []);

  // подчистка при размонтировании
  useEffect(() => {
    return () => {
      if (timerRef.current) window.clearInterval(timerRef.current);
      timerRef.current = undefined;
      if (isRec) {
        // не бросаем ошибку, просто best-effort
        enrollCancel().catch(() => undefined);
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // карта причин в подсказки
  const reasonToHint = (reason?: string) => {
    switch (reason) {
      case "no_face":
        return "Лицо не найдено — подойдите ближе и смотрите в камеру";
      case "low_quality":
        return "Слишком размыто/темно — подойдите ближе, добавьте света";
      case "not_diverse":
        return "Нужен новый ракурс — плавно поверните голову";
      case "bad_crop":
        return "Неудачный кадр — попробуйте ещё раз";
      default:
        return "";
    }
  };

  const start = async () => {
    if (isRec) return;
    setBusy(true);
    setHint("");
    try {
      targetRef.current = target;
      const r = await enrollStart(newName || undefined, targetRef.current);
      if (!r?.ok) throw new Error(r?.error || "start failed");

      setIsRec(true);
      setStatus("recording");
      setProgress(0);

      // быстрый цикл ~200 мс
      const tick = async () => {
        try {
          const rs = await enrollStep();
          if (rs?.ok) {
            const cnt = Number(rs.count ?? 0);
            const tgt = Number(rs.target ?? targetRef.current);
            setProgress(Math.min(1, cnt / Math.max(1, tgt)));
            // подсказки
            if (rs.reason) setHint(reasonToHint(rs.reason));
            else setHint("");

            if (cnt >= tgt) {
              const fin = await enrollFinish(newName || undefined);
              if (fin?.ok) {
                setStatus("done");
                setIsRec(false);
                if (timerRef.current) window.clearInterval(timerRef.current);
                timerRef.current = undefined;
                await load();
                setNewName(""); // сбросить имя
                // плавный авто-сброс кольца
                setTimeout(() => {
                  setStatus("idle");
                  setProgress(0);
                  setHint("");
                }, 900);
                return;
              } else {
                setStatus("error");
              }
            }
          }
        } catch {
          // держим цикл живым
        }
      };

      tick();
      timerRef.current = window.setInterval(tick, 190);
    } catch (e) {
      console.error(e);
      setStatus("error");
    } finally {
      setBusy(false);
    }
  };

  const cancel = async () => {
    if (timerRef.current) window.clearInterval(timerRef.current);
    timerRef.current = undefined;
    setIsRec(false);
    setProgress(0);
    setStatus("idle");
    setHint("");
    try {
      await enrollCancel();
    } catch {
      /* noop */
    }
  };

  const doDelete = async (id: number) => {
    if (!confirm("Удалить игрока?")) return;
    try {
      const r = await deletePlayer(id);
      if (r?.ok) await load();
    } catch {
      /* noop */
    }
  };

  return (
    <div className="section roster">
      <h2 className="steptitle">Состав</h2>
      <p className="lead" style={{ opacity: 0.9 }}>
        Добавьте игроков. Смотрите в камеру и плавно поверните голову — набор займёт 1–2 секунды.
      </p>

      <div className="roster-grid">
        {/* Левая колонка — видео + кольцо прогресса + панель действий */}
        <div className="roster-left">
          <div className="stream-wrap">
            <img
              className="video-stream"
              src="http://127.0.0.1:8000/video/mjpeg"
              alt="video"
              draggable={false}
            />
            <div className="stream-overlay">
              <FaceEnrollRing progress={progress} state={status} />
            </div>
          </div>

          <div className="roster-actions">
            <input
              className="glass-input"
              placeholder="Имя игрока (необязательно)"
              value={newName}
              onChange={(e) => setNewName(e.target.value)}
              disabled={isRec}
              maxLength={40}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !isRec) start();
              }}
            />

            <select
              className="glass-input"
              value={target}
              onChange={(e) => setTarget(Number(e.target.value))}
              disabled={isRec}
              title="Количество кадров для набора"
              style={{ width: 140 }}
            >
              <option value={6}>6 кадров</option>
              <option value={8}>8 кадров</option>
              <option value={10}>10 кадров</option>
              <option value={12}>12 кадров</option>
            </select>

            {!isRec ? (
              <GlassButton onClick={start} disabled={busy}>
                Добавить игрока
              </GlassButton>
            ) : (
              <GlassButton onClick={cancel} variant="ghost">
                Отмена
              </GlassButton>
            )}
          </div>

          {/* динамическая подсказка по качеству/ракурсу */}
          {hint && <div className="muted" style={{ marginTop: 8 }}>{hint}</div>}
        </div>

        {/* Правая колонка — список игроков */}
        <div className="roster-right">
          <div className="players-grid">
            {list.map((p) => (
              <div className="player-card" key={`${p.id}-${p.rev ?? 0}`}>
                <img
                  src={`http://127.0.0.1:8000/static/${p.thumb}?v=${p.rev ?? 0}`}
                  alt=""
                />
                <div className="pc-row">
                  <div className="pc-name">#{p.id} {p.name || "Без имени"}</div>
                  <button className="pc-del" onClick={() => doDelete(p.id)} title="Удалить">×</button>
                </div>
              </div>
            ))}
            {list.length === 0 && <div className="muted">Пока пусто — добавьте первого игрока.</div>}
          </div>
        </div>
      </div>
    </div>
  );
}
