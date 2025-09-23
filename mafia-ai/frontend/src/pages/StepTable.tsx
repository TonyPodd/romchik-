// src/pages/StepTable.tsx
import React, { useEffect, useMemo, useRef, useState } from "react";
import { GlassButton } from "@/components/GlassButton";
import {
  startVideo,
  getTableStatus,
  autoDetectTable,
  setTableROI,
  clearTableROI,
  beginTableCalibration,
  endTableCalibration,
} from "@/api";

type Pt = { x: number; y: number }; // нормализованные [0..1]

export function StepTable() {
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);

  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);

  const [poly, setPoly] = useState<Pt[] | null>(null);
  const [drawing, setDrawing] = useState(false);
  const [draft, setDraft] = useState<Pt[]>([]);

  // размеры видео-контейнера (точное совпадение с SVG)
  const [dims, setDims] = useState<{ w: number; h: number }>({ w: 0, h: 0 });

  // включаем видео и режим калибровки
  useEffect(() => {
    let mounted = true;
    (async () => {
      try { await startVideo({ fps: 30 }); } catch {}
      try { await beginTableCalibration(); } catch {}
      try {
        const st = await getTableStatus();
        if (mounted && Array.isArray(st?.poly_norm) && st.poly_norm.length >= 3) {
          setPoly(st.poly_norm.map(([x, y]: [number, number]) => ({ x, y })));
        }
      } catch {}
    })();
    return () => {
      mounted = false;
      endTableCalibration().catch(() => {});
    };
  }, []);

  // следим за размерами контейнера
  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      const r = entries[0]?.contentRect;
      if (r) setDims({ w: Math.max(1, r.width), h: Math.max(1, r.height) });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // норм → пикс
  const toPx = (p: Pt) => ({ x: p.x * dims.w, y: p.y * dims.h });

  // пикс → норм
  const toNorm = (clientX: number, clientY: number) => {
    const r = wrapRef.current?.getBoundingClientRect();
    if (!r) return { x: 0, y: 0 };
    const x = Math.min(1, Math.max(0, (clientX - r.left) / r.width));
    const y = Math.min(1, Math.max(0, (clientY - r.top) / r.height));
    return { x, y };
  };

  // автодетект
  const onAuto = async () => {
    setBusy(true); setMsg("Автопоиск стола…");
    try {
      const r = await autoDetectTable();
      if (r?.ok && Array.isArray(r.poly_norm) && r.poly_norm.length >= 3) {
        setPoly(r.poly_norm.map(([x, y]: [number, number]) => ({ x, y })));
        setDrawing(false); setDraft([]);
        setMsg("Стол найден ✓");
      } else {
        setMsg("Не удалось автоматически — обведите вручную");
        setDrawing(true); setDraft([]);
      }
    } catch {
      setMsg("Ошибка автопоиска — попробуйте вручную");
      setDrawing(true); setDraft([]);
    } finally {
      setBusy(false);
      setTimeout(() => setMsg(null), 1500);
    }
  };

  // ручной режим
  const onManual = () => {
    setMsg("Кликните точки по периметру. Двойной клик — завершить. ПКМ — шаг назад.");
    setDrawing(true); setDraft([]);
    setTimeout(() => setMsg(null), 2500);
  };

  // сброс
  const onClear = async () => {
    setBusy(true);
    try {
      await clearTableROI();
      setPoly(null); setDraft([]); setDrawing(false);
      setMsg("Контур удалён");
    } catch {
      setMsg("Не удалось очистить");
    } finally {
      setBusy(false);
      setTimeout(() => setMsg(null), 1200);
    }
  };

  // сохранить вручную
  const onSave = async () => {
    if (!draft || draft.length < 3) return;
    setBusy(true); setMsg("Сохраняем контур…");
    try {
      const body = draft.map(p => [Number(p.x.toFixed(5)), Number(p.y.toFixed(5))]) as [number, number][];
      const r = await setTableROI(body);
      if (r?.ok && Array.isArray(r.poly_norm) && r.poly_norm.length >= 3) {
        setPoly(r.poly_norm.map(([x, y]: [number, number]) => ({ x, y })));
        setDrawing(false); setDraft([]);
        setMsg("Контур сохранён ✓");
      } else {
        setMsg("Не получилось сохранить контур");
      }
    } catch {
      setMsg("Ошибка сохранения");
    } finally {
      setBusy(false);
      setTimeout(() => setMsg(null), 1400);
    }
  };

  // сбор точек
  const onOverlayClick = (e: React.MouseEvent) => {
    if (!drawing) return;
    const p = toNorm(e.clientX, e.clientY);
    setDraft(d => [...d, p]);
  };
  const onOverlayDouble = (e: React.MouseEvent) => {
    if (!drawing) return;
    e.preventDefault();
    if (draft.length >= 3) {
      setMsg("Контур готов — нажмите «Сохранить»");
      setTimeout(() => setMsg(null), 1200);
    }
  };
  const onOverlayContext = (e: React.MouseEvent) => {
    if (!drawing) return;
    e.preventDefault();
    setDraft(d => d.slice(0, -1));
  };

  // пути для SVG
  const polyPath = useMemo(() => {
    if (!poly || dims.w === 0 || dims.h === 0) return "";
    const pts = poly.map(toPx);
    return pts.map((p, i) => (i === 0 ? `M ${p.x},${p.y}` : `L ${p.x},${p.y}`)).join(" ") + " Z";
  }, [poly, dims]);

  const draftPath = useMemo(() => {
    if (!drawing || draft.length < 1 || dims.w === 0 || dims.h === 0) return "";
    const pts = draft.map(toPx);
    return pts.map((p, i) => (i === 0 ? `M ${p.x},${p.y}` : `L ${p.x},${p.y}`)).join(" ");
  }, [drawing, draft, dims]);

  return (
    <div className="section">
      <h2 className="steptitle">Калибровка стола</h2>
      <p className="muted">Во время калибровки лица и жесты отключены. Используйте автопоиск или обведите вручную.</p>

      <div className="table-step" style={{ display: "grid", gap: 12, justifyItems: "center" }}>
        {/* ВИДЕО-БЛОК: широкий, центрированный, строго 16:9 */}
        <div
          ref={wrapRef}
          className="table-video"
          style={{
            position: "relative",
            width: "min(1400px, 96vw)",
            aspectRatio: "16 / 9",
            borderRadius: 18,
            overflow: "hidden",
            border: "1px dashed rgba(255,255,255,.25)",
            background: "rgba(255,255,255,.04)",
          }}
        >
          {/* сам поток — растягиваем ровно по контейнеру (16:9 → нет обрезаний) */}
          <img
            ref={imgRef}
            className="video-stream"
            src="http://127.0.0.1:8000/video/mjpeg"
            alt="video"
            draggable={false}
            style={{
              position: "absolute",
              inset: 0,
              width: "100%",
              height: "100%",
              objectFit: "fill", // контейнер уже 16:9 → без кропа
              userSelect: "none",
              pointerEvents: "none",
              display: "block",
            }}
          />

          {/* SVG-оверлей для полигона/черновика */}
          <svg
            className="overlay"
            width="100%"
            height="100%"
            style={{ position: "absolute", inset: 0 }}
            onClick={onOverlayClick}
            onDoubleClick={onOverlayDouble}
            onContextMenu={onOverlayContext}
          >
            {poly && poly.length >= 3 && (
              <>
                <path d={polyPath} fill="rgba(80,190,255,0.15)" stroke="rgb(255,210,70)" strokeWidth={2} />
                {poly.map((p, i) => {
                  const q = toPx(p);
                  return <circle key={i} cx={q.x} cy={q.y} r={5} fill="white" />;
                })}
              </>
            )}
            {drawing && draft.length > 0 && (
              <>
                <path d={draftPath} fill="none" stroke="rgba(255,255,255,0.9)" strokeWidth={2} strokeDasharray="6 6" />
                {draft.map((p, i) => {
                  const q = toPx(p);
                  return <circle key={i} cx={q.x} cy={q.y} r={4} fill="#fff" opacity={0.9} />;
                })}
              </>
            )}
          </svg>
        </div>

        {/* ПАНЕЛЬ КНОПОК — ВНЕ ОВЕРЛЕЯ, ПОД ВИДЕО */}
        <div style={{ display: "flex", gap: 10, flexWrap: "wrap", justifyContent: "center" }}>
          <GlassButton onClick={onAuto} disabled={busy}>Автопоиск</GlassButton>
          <GlassButton onClick={onManual} disabled={busy}>Обвести вручную</GlassButton>
          <GlassButton onClick={onClear} variant="ghost" disabled={busy || (!poly && draft.length === 0)}>Сбросить</GlassButton>
          {drawing && draft.length >= 3 && (
            <GlassButton onClick={onSave} disabled={busy}>Сохранить</GlassButton>
          )}
        </div>

        {/* СООБЩЕНИЕ — ПОД ВИДЕО, НЕ ПЕРЕКРЫВАЕТ НИЧЕГО */}
        {msg && (
          <div
            style={{
              fontWeight: 800,
              padding: "6px 10px",
              borderRadius: 10,
              background: "rgba(0,0,0,.35)",
              border: "1px solid rgba(255,255,255,.16)",
              backdropFilter: "blur(6px)",
            }}
          >
            {msg}
          </div>
        )}
      </div>
    </div>
  );
}
