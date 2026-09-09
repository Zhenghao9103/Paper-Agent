export type DocumentStatus =
  | "uploaded"
  | "parsing"
  | "parsed"
  | "indexed"
  | "analyzed"
  | "failed";

export interface DocumentRecord {
  id: number;
  title: string;
  authors: string | null;
  year: number | null;
  file_type: "pdf";
  file_path: string;
  status: DocumentStatus;
  source_type: "uploaded" | "arxiv";
  abstract: string | null;
  created_at: string;
  updated_at: string;
}

export interface DocumentElement {
  id: number;
  document_id: number;
  element_uid: string;
  page_number: number | null;
  element_type: "figure" | "equation" | "table" | string;
  label: string | null;
  caption: string | null;
  image_path: string | null;
  raw_text: string | null;
  structured_data: Record<string, unknown> | unknown[] | null;
  parse_status: string;
  warning_codes?: string[];
  metadata?: Record<string, unknown>;
}

export interface DocumentChunkDetail {
  chunk_id: number;
  chunk_uid: string;
  chunk_type: string;
  metadata: Record<string, unknown>;
}

export interface DocumentChunk {
  id: number;
  document_id: number;
  page_number: number;
  chunk_index: number;
  content: string;
  detail: DocumentChunkDetail | null;
}
