// ─────────────────────────────────────────────────────────────
// Types derived directly from the F3 backend Pydantic schemas.
// Keep these in sync with:
//   backend/app/schemas/papers.py
//   backend/app/schemas/research.py
// ─────────────────────────────────────────────────────────────

// ── GET /papers ──────────────────────────────────────────────

export interface Paper {
  id: string;
  title: string;
  filename: string;
  page_count: number | null;
  created_at: string; // ISO 8601 UTC string from the backend
}

// ── POST /papers/upload ───────────────────────────────────────

export interface UploadPaperResponse {
  id: string;
  title: string;
  filename: string;
  page_count: number | null;
  chunk_count: number;
  indexing_status: string; // always "complete" on success
  file_path: string;
}

// ── POST /research/query ──────────────────────────────────────

export interface ResearchQueryRequest {
  paper_id: string;
  question: string;
  top_k?: number; // 1–10, default 5
}

export interface EvidenceItem {
  evidence_id: string;   // "E1", "E2", …
  chunk_id: string;
  paper_id: string;
  chunk_index: number;
  page_start: number | null;
  page_end: number | null;
  section: string | null;
  title: string;
  filename: string;
  similarity: number;    // cosine similarity 0–1
  text: string;          // raw passage text
}

export interface CitationItem {
  evidence_id: string;   // matches evidence[].evidence_id
  chunk_id: string;
  paper_id: string;
  title: string;
  filename: string;
  page_start: number | null;
  page_end: number | null;
  section: string | null;
}

export interface ResearchQueryResponse {
  question: string;
  answerable: boolean;
  answer: string | null;
  abstention_reason: string | null;
  citations: CitationItem[];
  evidence: EvidenceItem[];
}

// ── GET /health ───────────────────────────────────────────────

export interface HealthResponse {
  status: string; // "ok"
}

// ── Shared API error shape ────────────────────────────────────

export interface ApiError {
  detail: string | { msg: string; type: string }[];
}
