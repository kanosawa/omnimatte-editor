import type { SegmentRequest, SessionResponse } from "../types";

const API_BASE: string = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

export async function uploadVideo(file: File): Promise<SessionResponse> {
  const fd = new FormData();
  fd.append("video", file);
  const res = await fetch(`${API_BASE}/session`, { method: "POST", body: fd });
  if (!res.ok) {
    const detail = await safeReadDetail(res);
    throw new Error(`session failed: ${res.status} ${detail}`);
  }
  return (await res.json()) as SessionResponse;
}

export async function segment(req: SegmentRequest): Promise<Blob> {
  const res = await fetch(`${API_BASE}/segment`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) {
    const detail = await safeReadDetail(res);
    throw new Error(`segment failed: ${res.status} ${detail}`);
  }
  return await res.blob();
}

async function safeReadDetail(res: Response): Promise<string> {
  try {
    const data = await res.clone().json();
    if (data && typeof data === "object" && "detail" in data) return String((data as { detail: unknown }).detail);
  } catch {
    // ignore
  }
  try {
    return await res.text();
  } catch {
    return "";
  }
}
