// src/api.ts
export async function startVideo(params?: { camera_index?: number; fps?: number; table_y_ratio?: number }) {
  const qs = new URLSearchParams();
  if (params?.camera_index != null) qs.set("camera_index", String(params.camera_index));
  if (params?.fps != null) qs.set("fps", String(params.fps));
  if (params?.table_y_ratio != null) qs.set("table_y_ratio", String(params.table_y_ratio));
  const r = await fetch("http://127.0.0.1:8000/video/start" + (qs.toString() ? `?${qs.toString()}` : ""), { method: "POST" });
  return r.json();
}

export async function getTableStatus() {
  const r = await fetch("http://127.0.0.1:8000/table/status");
  return r.json();
}

export async function autoDetectTable() {
  const r = await fetch("http://127.0.0.1:8000/table/autodetect", { method: "POST" });
  return r.json();
}

export async function setTableROI(poly: [number, number][]) {
  const r = await fetch("http://127.0.0.1:8000/table/set_roi", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ poly }),
  });
  return r.json();
}

export async function clearTableROI() {
  const r = await fetch("http://127.0.0.1:8000/table/clear", { method: "POST" });
  return r.json();
}

/** --- НОВОЕ: включить режим калибровки (видеопоток без лиц/жестов) --- */
export async function beginTableCalibration() {
  try {
    const r = await fetch("http://127.0.0.1:8000/table/begin", { method: "POST" });
    return r.ok ? r.json() : { ok: false };
  } catch {
    return { ok: false };
  }
}

/** --- НОВОЕ: выйти из режима калибровки (вернуть полный рендер) --- */
export async function endTableCalibration() {
  try {
    const r = await fetch("http://127.0.0.1:8000/table/end", { method: "POST" });
    return r.ok ? r.json() : { ok: false };
  } catch {
    return { ok: false };
  }
}
