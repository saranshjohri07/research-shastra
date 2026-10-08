/**
 * Centralised API client for the ResearchShastra FastAPI backend.
 *
 * All fetch calls live here. Components never import fetch directly.
 * The base URL is read from NEXT_PUBLIC_API_URL (set in .env.local).
 */

import type {
  HealthResponse,
  Paper,
  UploadPaperResponse,
  ResearchQueryRequest,
  ResearchQueryResponse,
  ApiError,
} from "@/types/api";

// ── Base URL ─────────────────────────────────────────────────

const API_BASE =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") ??
  "http://localhost:8000";

// ── Helpers ───────────────────────────────────────────────────

/**
 * Extracts a human-readable error message from any HTTP error response.
 * Handles both FastAPI's standard 422 detail array and plain string details.
 */
async function extractErrorMessage(res: Response): Promise<string> {
  try {
    const body: ApiError = await res.json();
    if (typeof body.detail === "string") return body.detail;
    if (Array.isArray(body.detail)) {
      return body.detail.map((e) => e.msg).join("; ");
    }
  } catch {
    // fall through to status text
  }
  return `Request failed: ${res.status} ${res.statusText}`;
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "GET",
    headers: { Accept: "application/json" },
  });
  if (!res.ok) {
    const message = await extractErrorMessage(res);
    throw new Error(message);
  }
  return res.json() as Promise<T>;
}

async function postJson<TBody, TResponse>(
  path: string,
  body: TBody,
): Promise<TResponse> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "application/json",
    },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const message = await extractErrorMessage(res);
    throw new Error(message);
  }
  return res.json() as Promise<TResponse>;
}

async function postForm<TResponse>(
  path: string,
  formData: FormData,
): Promise<TResponse> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    // Do NOT set Content-Type — the browser sets the correct multipart boundary.
    headers: { Accept: "application/json" },
    body: formData,
  });
  if (!res.ok) {
    const message = await extractErrorMessage(res);
    throw new Error(message);
  }
  return res.json() as Promise<TResponse>;
}

// ── Public API functions ──────────────────────────────────────

/** GET /health — liveness check */
export async function checkHealth(): Promise<HealthResponse> {
  return get<HealthResponse>("/health");
}

/** GET /papers — list all indexed papers, newest first */
export async function listPapers(): Promise<Paper[]> {
  return get<Paper[]>("/papers/");
}

/**
 * POST /papers/upload — upload a PDF file.
 * @param file A File object from a file input or drop event.
 */
export async function uploadPaper(file: File): Promise<UploadPaperResponse> {
  const formData = new FormData();
  formData.append("file", file);
  return postForm<UploadPaperResponse>("/papers/upload", formData);
}

/**
 * POST /research/query — ask a question against an indexed paper.
 */
export async function queryResearch(
  request: ResearchQueryRequest,
): Promise<ResearchQueryResponse> {
  return postJson<ResearchQueryRequest, ResearchQueryResponse>(
    "/research/query",
    request,
  );
}
