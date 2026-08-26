// Client for the NeuroProfile FastAPI serving layer (serve.py).

// API base resolves at runtime: a URL you paste into the dashboard (saved in localStorage)
// wins, else the build-time env var, else localhost. This means no .env.local / rebuild needed —
// paste your Colab tunnel URL in the header and it connects.
const DEFAULT_API = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

export function getApiBase(): string {
  if (typeof window !== "undefined") {
    try {
      const saved = localStorage.getItem("np_api");
      if (saved) return saved;
    } catch { /* ignore */ }
  }
  return DEFAULT_API;
}

export function setApiBase(url: string) {
  try {
    localStorage.setItem("np_api", url.trim().replace(/\/$/, ""));
  } catch { /* ignore */ }
}

export interface VideoSummary {
  video_id: string;
  title: string;
  duration: number | null;
  system_names: string[] | null;
  system_tiers: string[] | null;
  system_profile: number[] | null;
}

export interface Timeline {
  times: number[];
  system_ts: number[][]; // (T, S)
  valid: boolean[];
}

export interface Moment {
  t: number;
  system: string;
  tier?: string;
  value: number;
}

export interface VideoFull extends VideoSummary {
  system_derived?: boolean[];
  moments?: Moment[];
  timeline_path?: string;
  transcript?: string;
  video_url?: string;
  timeline: Timeline;
}

export interface Neighbor extends VideoSummary {
  score: number;
}

async function j<T>(path: string): Promise<T> {
  const res = await fetch(`${getApiBase()}${path}`);
  if (!res.ok) throw new Error(`${path} -> ${res.status}`);
  return res.json() as Promise<T>;
}

// URL for the original mp4 of a corpus clip (served by serve.py --videos-dir).
// If no file exists for the id, serve.py returns 404 and the player falls back to a clock.
export const videoSrc = (id: string) => `${getApiBase()}/video/${encodeURIComponent(id)}`;

export const getVideos = () => j<VideoSummary[]>("/api/videos");

// Delete a clip from the DB + its timeline + video file (backend does all three).
export async function deleteVideo(id: string): Promise<void> {
  const r = await fetch(`${getApiBase()}/api/videos/${encodeURIComponent(id)}`, { method: "DELETE" });
  if (!r.ok) throw new Error(`delete failed (${r.status})`);
}
export const getVideo = (id: string) => j<VideoFull>(`/api/videos/${encodeURIComponent(id)}`);
export const getSimilar = (id: string) =>
  j<Neighbor[]>(`/api/videos/${encodeURIComponent(id)}/similar`);

// --- live analyze (only works when API_BASE points at analyze_server.py on Colab) ---
export interface JobStatus {
  status: "queued" | "running" | "done" | "error" | "canceled";
  stage?: string;
  error?: string;
  video_id?: string;
  log?: string[];
}

export async function analyze(file: File): Promise<{ job_id: string; video_id: string }> {
  const fd = new FormData();
  fd.append("file", file);
  const r = await fetch(`${getApiBase()}/analyze`, { method: "POST", body: fd });
  if (r.status === 404) throw new Error("this API has no /analyze — point at the Colab live backend");
  if (!r.ok) throw new Error(`analyze failed (${r.status})`);
  return r.json();
}

export async function getJob(id: string): Promise<JobStatus> {
  const r = await fetch(`${getApiBase()}/jobs/${id}`);
  if (!r.ok) throw new Error(`job ${r.status}`);
  return r.json();
}

export async function cancelJob(id: string): Promise<void> {
  try {
    await fetch(`${getApiBase()}/jobs/${id}/cancel`, { method: "POST" });
  } catch { /* best-effort; the poll loop will still stop on canceled status */ }
}