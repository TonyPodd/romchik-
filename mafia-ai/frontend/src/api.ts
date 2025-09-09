// src/api.ts
// Единая точка запросов к backend FastAPI.
// Базовый URL можно переопределить переменной окружения VITE_API_URL.

export type Health = { ok: boolean; clients: number; video_running: boolean };
export type Player = { id: number; name: string; thumb: string; embedding?: number[] };

// Если в .env задать VITE_API_URL, будет взят он, иначе — localhost
const API_BASE: string =
  (import.meta as any)?.env?.VITE_API_URL ?? 'http://127.0.0.1:8000';

// Универсальный helper с таймаутом и авто-JSON
async function request<T = any>(
  path: string,
  init: RequestInit = {},
  timeoutMs = 10000
): Promise<T> {
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), timeoutMs);
  const headers = new Headers(init.headers || {});
  if (init.body && !headers.has('Content-Type')) {
    headers.set('Content-Type', 'application/json');
  }
  try {
    const res = await fetch(`${API_BASE}${path}`, {
      ...init,
      headers,
      signal: ctrl.signal,
    });
    const ct = res.headers.get('content-type') || '';
    const data = ct.includes('application/json') ? await res.json() : await res.text();
    if (!res.ok) throw new Error((data && (data as any).error) || `HTTP ${res.status}`);
    return data as T;
  } finally {
    clearTimeout(t);
  }
}

/* -------------------- Видео / жесты / стол -------------------- */

export async function getHealth(): Promise<Health> {
  return request<Health>('/health');
}

export async function startVideo(params?: {
  camera_index?: number;
  fps?: number;
  table_y_ratio?: number;
}) {
  const qs = new URLSearchParams();
  if (params?.camera_index != null) qs.set('camera_index', String(params.camera_index));
  if (params?.fps != null) qs.set('fps', String(params.fps));
  if (params?.table_y_ratio != null) qs.set('table_y_ratio', String(params.table_y_ratio));
  const q = qs.toString();
  return request(`/video/start${q ? `?${q}` : ''}`, { method: 'POST' });
}

export async function stopVideo() {
  return request('/video/stop', { method: 'POST' });
}

export async function getTableStatus(): Promise<{ poly_norm: [number, number][] | null }> {
  return request('/table/status');
}

export async function setTableROI(poly: [number, number][]) {
  return request('/table/set_roi', {
    method: 'POST',
    body: JSON.stringify({ poly }),
  });
}

export async function clearTableROI() {
  return request('/table/clear', { method: 'POST' });
}

export async function autoDetectTable() {
  return request('/table/autodetect', { method: 'POST' });
}

/* -------------------- Игроки / авторизация лиц -------------------- */

export async function listPlayers(): Promise<{ players: Player[] }> {
  return request('/players/list');
}

export async function resetPlayers() {
  return request('/players/reset', { method: 'POST' });
}

export async function enrollPlayer(name?: string) {
  return request('/players/enroll', {
    method: 'POST',
    body: JSON.stringify({ name }),
  });
}

export async function setPlayerName(id: number, name: string) {
  return request('/players/name', {
    method: 'POST',
    body: JSON.stringify({ id, name }),
  });
}

export async function deletePlayer(id: number) {
  return request('/players/delete', {
    method: 'POST',
    body: JSON.stringify({ id }),
  });
}
