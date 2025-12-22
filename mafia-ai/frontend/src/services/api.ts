const API_BASE = 'http://127.0.0.1:8000';
const FETCH_TIMEOUT = 30000; // 30 seconds (увеличено для CompreFace)

// Helper function with timeout
async function fetchWithTimeout(url: string, options: RequestInit = {}, timeout = FETCH_TIMEOUT): Promise<Response> {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeout);

  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal,
    });
    clearTimeout(id);

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    return response;
  } catch (error) {
    clearTimeout(id);
    if (error instanceof Error && error.name === 'AbortError') {
      throw new Error(`Request timeout after ${timeout}ms`);
    }
    throw error;
  }
}

export interface EnrollStartResponse {
  ok: boolean;
  session?: {
    id: number;
    name: string;
    target: number;
    count: number;
  };
  error?: string;
}

export interface EnrollStatusResponse {
  ok: boolean;
  name?: string;
  count?: number;
  target?: number;
  progress?: number;
  hint?: string;
  error?: string;
}

export interface EnrollSnapResponse {
  ok: boolean;
  added?: boolean;
  count?: number;
  target?: number;
  reason?: string;
  error?: string;
}

export interface EnrollFinishResponse {
  ok: boolean;
  player?: any;
  error?: string;
}

// Video API
export async function startVideo(): Promise<{ ok: boolean }> {
  console.log('[API] Calling POST /video/start...');
  const res = await fetchWithTimeout(`${API_BASE}/video/start`, { method: 'POST' });
  const data = await res.json();
  console.log('[API] /video/start response:', data);
  return data;
}

export async function stopVideo(): Promise<{ ok: boolean }> {
  const res = await fetchWithTimeout(`${API_BASE}/video/stop`, { method: 'POST' });
  return res.json();
}

export async function getVideoStatus(): Promise<{ running: boolean }> {
  const res = await fetchWithTimeout(`${API_BASE}/video/status`);
  return res.json();
}

export function getVideoStreamUrl(): string {
  return `${API_BASE}/video/mjpeg`;
}

// Player enrollment API
export async function enrollStart(name: string, target: number = 12): Promise<EnrollStartResponse> {
  const res = await fetchWithTimeout(`${API_BASE}/players/enroll/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, target }),
  });
  return res.json();
}

export async function enrollStatus(): Promise<EnrollStatusResponse> {
  const res = await fetchWithTimeout(`${API_BASE}/players/enroll/status`);
  return res.json();
}

export async function enrollSnap(): Promise<EnrollSnapResponse> {
  const res = await fetchWithTimeout(`${API_BASE}/players/enroll/snap`, {
    method: 'POST',
  });
  return res.json();
}

export async function enrollFinish(name?: string): Promise<EnrollFinishResponse> {
  const res = await fetchWithTimeout(`${API_BASE}/players/enroll/finish`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name }),
  });
  return res.json();
}

export async function enrollCancel(): Promise<{ ok: boolean }> {
  const res = await fetchWithTimeout(`${API_BASE}/players/enroll/cancel`, {
    method: 'POST',
  });
  return res.json();
}

// Player management API
export interface Player {
  id: number;
  name: string;
  embedding: number[];
  thumb?: string;
  rev?: number;
}

export async function listPlayers(): Promise<{ players: Player[] }> {
  const res = await fetchWithTimeout(`${API_BASE}/players/list`);
  return res.json();
}

export async function deletePlayer(id: number): Promise<{ ok: boolean }> {
  const res = await fetchWithTimeout(`${API_BASE}/players/delete`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ id }),
  });
  return res.json();
}

export async function resetPlayers(): Promise<{ ok: boolean }> {
  const res = await fetchWithTimeout(`${API_BASE}/players/reset`, {
    method: 'POST',
  });
  return res.json();
}

// Voice Enrollment

export interface VoiceEnrollResponse {
  ok: boolean;
  voice_embedding?: number[];
  embedding_dim?: number;
  voice_path?: string;
  error?: string;
  message?: string;
}

export async function enrollVoice(playerId: number, audioBlob: Blob): Promise<VoiceEnrollResponse> {
  const formData = new FormData();
  formData.append('player_id', playerId.toString());
  formData.append('audio_file', audioBlob, 'voice.wav');

  const res = await fetchWithTimeout(
    `${API_BASE}/players/enroll/voice`,
    {
      method: 'POST',
      body: formData,
    },
    60000 // 60 seconds timeout for audio processing
  );
  return res.json();
}

export async function getPlayerVoice(playerId: number): Promise<{
  ok: boolean;
  has_voice?: boolean;
  voice_path?: string;
  embedding_dim?: number;
  error?: string;
}> {
  const res = await fetchWithTimeout(`${API_BASE}/players/${playerId}/voice`);
  return res.json();
}
