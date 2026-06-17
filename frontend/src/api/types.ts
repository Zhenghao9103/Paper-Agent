export type DocumentStatus = "uploaded" | "parsing" | "indexed" | "analyzed" | "failed";

export interface DocumentRecord {
  id: number;
  title: string;
  authors: string | null;
  year: number | null;
  file_type: "pdf" | "pptx";
  file_path: string;
  status: DocumentStatus;
  source_type: "uploaded" | "arxiv";
  abstract: string | null;
  created_at: string;
  updated_at: string;
}
