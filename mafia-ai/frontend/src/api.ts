// src/api.ts
const BASE = "http://127.0.0.1:8000";

async function jfetch<T = any>(url: string, init?: RequestInit): Promise<T> {
  const r = await fetch(url, init);
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data?.error || `HTTP ${r.status}`);
  return data;
}

export async function startVideo(params?: { camera_index?: number; fps?: number; table_y_ratio?: number }) {
  const qs = new URLSearchParams();
  if (params?.camera_index != null) qs.set("camera_index", String(params.camera_index));
  if (params?.fps != null) qs.set("fps", String(params.fps));
  if (params?.table_y_ratio != null) qs.set("table_y_ratio", String(params.table_y_ratio));
  return jfetch(`${BASE}/video/start${qs.toString() ? `?${qs.toString()}` : ""}`, { method: "POST" });
}

export async function getTableStatus() {
  return jfetch(`${BASE}/table/status`);
}
export async function autoDetectTable() {
  return jfetch(`${BASE}/table/autodetect`, { method: "POST" });
}
export async function setTableROI(poly: [number, number][]) {
  return jfetch(`${BASE}/table/set_roi`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ poly }),
  });
}
export async function clearTableROI() {
  return jfetch(`${BASE}/table/clear`, { method: "POST" });
}
export async function tableBegin() {
  return jfetch(`${BASE}/table/begin`, { method: "POST" });
}
export async function tableEnd() {
  return jfetch(`${BASE}/table/end`, { method: "POST" });
}

export async function listPlayers() {
  return jfetch(`${BASE}/players/list`);
}
export async function setPlayerName(id: number, name: string) {
  return jfetch(`${BASE}/players/name`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ id, name }),
  });
}
export async function deletePlayer(id: number) {
  return jfetch(`${BASE}/players/delete`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ id }),
  });
}


export async function enrollStart(name?: string, target = 8) {
  const r = await fetch(`${BASE}/players/enroll/start`, {
    method: "POST",
    headers: { "Content-Type":"application/json" },
    body: JSON.stringify({ name, target }),
  });
  return r.json();
}
export async function enrollStep() {
  const r = await fetch(`${BASE}/players/enroll/step`, { method:"POST" });
  return r.json();
}
export async function enrollFinish(name?: string) {
  const r = await fetch(`${BASE}/players/enroll/finish`, {
    method: "POST",
    headers: { "Content-Type":"application/json" },
    body: JSON.stringify({ name }),
  });
  return r.json();
}
export async function enrollCancel() {
  const r = await fetch(`${BASE}/players/enroll/cancel`, { method:"POST" });
  return r.json();
}

// Алиасы для совместимости со старым кодом
export const beginTableCalibration = tableBegin;
export const endTableCalibration   = tableEnd;
