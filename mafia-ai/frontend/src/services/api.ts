const CURRENT_HOST =
  typeof window !== 'undefined' && window.location.hostname
    ? window.location.hostname
    : '';

const LOCAL_HOSTS = new Set(['', 'localhost', '127.0.0.1', '0.0.0.0']);
const isLocalDevHost = LOCAL_HOSTS.has(CURRENT_HOST);
const envApiBase = (import.meta.env.VITE_API_BASE || '').trim();
const API_BASE = envApiBase || (isLocalDevHost ? '/api' : `http://${CURRENT_HOST}:8000`);
const FETCH_TIMEOUT = 10000; // 10 seconds
const ENROLL_FINISH_TIMEOUT = 60000; // CompreFace enrollment can take longer
const VOICE_REGISTER_TIMEOUT = 120000; // First voice registration can be slow on cold start
const SPEECH_RECOGNIZE_TIMEOUT = 120000; // ASR warmup on CPU can be slow

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
  details?: any;
}

// Video API
export async function startVideo(): Promise<{ ok: boolean; [key: string]: any }> {
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

export async function getVideoStatus(): Promise<{ running: boolean; gestures_enabled?: boolean }> {
  const res = await fetchWithTimeout(`${API_BASE}/video/status`);
  return res.json();
}

export async function setVideoGestures(enabled: boolean): Promise<{ ok: boolean; gestures_enabled?: boolean; error?: string }> {
  const res = await fetchWithTimeout(`${API_BASE}/video/gestures`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ enabled }),
  });
  return res.json();
}

export async function setVideoFaceMatch(enabled: boolean): Promise<{ ok: boolean; face_match_enabled?: boolean; error?: string }> {
  const res = await fetchWithTimeout(`${API_BASE}/video/face-match`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ enabled }),
  });
  return res.json();
}

export function getVideoStreamUrl(): string {
  return `${API_BASE}/video/mjpeg`;
}

export type TablePoint = [number, number];

export interface TableStatusResponse {
  poly_norm: TablePoint[] | null;
}

export interface TableUpdateResponse {
  ok: boolean;
  poly_norm?: TablePoint[] | null;
  error?: string;
}

export async function getTableStatus(): Promise<TableStatusResponse> {
  const res = await fetchWithTimeout(`${API_BASE}/table/status`);
  return res.json();
}

export async function tableAutoDetect(): Promise<TableUpdateResponse> {
  const res = await fetchWithTimeout(`${API_BASE}/table/autodetect`, { method: 'POST' }, 30000);
  return res.json();
}

export async function tableSetRoi(poly: TablePoint[]): Promise<TableUpdateResponse> {
  const res = await fetchWithTimeout(`${API_BASE}/table/set_roi`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ poly }),
  });
  return res.json();
}

export async function tableClearRoi(): Promise<TableUpdateResponse> {
  const res = await fetchWithTimeout(`${API_BASE}/table/clear`, { method: 'POST' });
  return res.json();
}

export async function tableBeginCalibration(): Promise<{ ok: boolean; error?: string }> {
  const res = await fetchWithTimeout(`${API_BASE}/table/begin`, { method: 'POST' });
  return res.json();
}

export async function tableEndCalibration(): Promise<{ ok: boolean; error?: string }> {
  const res = await fetchWithTimeout(`${API_BASE}/table/end`, { method: 'POST' });
  return res.json();
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
  }, ENROLL_FINISH_TIMEOUT);
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

export function getPlayerThumbUrl(player: Pick<Player, 'thumb' | 'rev'>): string | undefined {
  if (!player.thumb) {
    return undefined;
  }
  const version = typeof player.rev === 'number' ? `?v=${player.rev}` : '';
  return `${API_BASE}/static/${player.thumb}${version}`;
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

// Voice API
export interface VoiceProfile {
  player_id: number;
  player_name: string;
  samples_count: number;
  created_at: number;
}

export interface VoiceRegisterResponse {
  ok: boolean;
  samples_registered?: number;
  error?: string;
}

export interface VoiceIdentifyResponse {
  ok: boolean;
  player_id?: number | null;
  player_name?: string | null;
  confidence?: number;
  error?: string;
}

export interface VoiceTestMatch {
  player_id: number;
  player_name: string;
  score: number;
}

export interface VoiceTestIdentifyResponse {
  ok: boolean;
  correct?: boolean;
  expected_player_id?: number;
  expected_player_name?: string | null;
  predicted_player_id?: number | null;
  predicted_player_name?: string | null;
  confidence?: number;
  top_matches?: VoiceTestMatch[];
  error?: string;
}

export async function voiceRegister(
  playerId: number,
  playerName: string,
  audioSamples: number[][],
  sampleRate: number = 16000,
): Promise<VoiceRegisterResponse> {
  const res = await fetchWithTimeout(
    `${API_BASE}/voice/register`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        player_id: playerId,
        player_name: playerName,
        audio_samples: audioSamples,
        sample_rate: sampleRate,
      }),
    },
    VOICE_REGISTER_TIMEOUT,
  );
  return res.json();
}

export async function voiceIdentify(
  audio: number[],
  sampleRate: number = 16000,
): Promise<VoiceIdentifyResponse> {
  const res = await fetchWithTimeout(
    `${API_BASE}/voice/identify`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        audio,
        sample_rate: sampleRate,
      }),
    },
    20000,
  );
  return res.json();
}

export async function voiceTestIdentify(
  expectedPlayerId: number,
  audio: number[],
  sampleRate: number = 16000,
): Promise<VoiceTestIdentifyResponse> {
  try {
    const res = await fetchWithTimeout(
      `${API_BASE}/voice/test/identify`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          expected_player_id: expectedPlayerId,
          audio,
          sample_rate: sampleRate,
        }),
      },
      20000,
    );
    return res.json();
  } catch (error: any) {
    // Совместимость со старым backend, где тестовый роут еще не добавлен.
    const message = String(error?.message || '');
    if (message.includes('HTTP 404')) {
      const fallback = await voiceIdentify(audio, sampleRate);
      const predictedId = fallback.player_id ?? null;
      return {
        ok: Boolean(fallback.ok),
        correct: predictedId === expectedPlayerId,
        expected_player_id: expectedPlayerId,
        expected_player_name: null,
        predicted_player_id: predictedId,
        predicted_player_name: fallback.player_name ?? null,
        confidence: Number(fallback.confidence || 0),
        top_matches: [],
        error: fallback.error,
      };
    }
    throw error;
  }
}

export async function voiceListProfiles(): Promise<{ ok: boolean; profiles: VoiceProfile[]; error?: string }> {
  const res = await fetchWithTimeout(`${API_BASE}/voice/profiles`);
  return res.json();
}

export async function voiceDeleteProfile(playerId: number): Promise<{ ok: boolean; error?: string }> {
  const res = await fetchWithTimeout(`${API_BASE}/voice/profile/delete`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ player_id: playerId }),
  });
  return res.json();
}

export async function voiceClearProfiles(): Promise<{ ok: boolean; error?: string }> {
  const res = await fetchWithTimeout(`${API_BASE}/voice/clear`, {
    method: 'POST',
  });
  return res.json();
}

export interface SpeechLogEntry {
  id: number;
  timestamp: number;
  speaker_id?: number | null;
  speaker_name?: string | null;
  speaker_label: string;
  confidence: number;
  text: string;
  line: string;
}

export interface SpeechRecognizeResponse {
  ok: boolean;
  speaker_id?: number | null;
  speaker_name?: string | null;
  speaker_label?: string;
  confidence?: number;
  text?: string;
  line?: string;
  asr_error?: string | null;
  entry?: SpeechLogEntry | null;
  skipped?: boolean;
  reason?: string;
  error?: string;
}

export async function speechRecognizeChunk(
  audio: number[],
  sampleRate: number = 16000,
  addToLogs: boolean = true,
): Promise<SpeechRecognizeResponse> {
  const res = await fetchWithTimeout(
    `${API_BASE}/voice/logs/recognize`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        audio,
        sample_rate: sampleRate,
        add_to_logs: addToLogs,
      }),
    },
    SPEECH_RECOGNIZE_TIMEOUT,
  );
  return res.json();
}

export async function speechLogsList(limit: number = 200): Promise<{ ok: boolean; logs: SpeechLogEntry[]; error?: string }> {
  const res = await fetchWithTimeout(`${API_BASE}/voice/logs?limit=${encodeURIComponent(String(limit))}`);
  return res.json();
}

export async function speechLogsClear(): Promise<{ ok: boolean; error?: string }> {
  const res = await fetchWithTimeout(`${API_BASE}/voice/logs/clear`, {
    method: 'POST',
  });
  return res.json();
}
