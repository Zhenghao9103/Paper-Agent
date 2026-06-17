import { request } from "./client";
import type { DocumentRecord } from "./types";

export function listDocuments(): Promise<DocumentRecord[]> {
  return request<DocumentRecord[]>("/api/documents");
}

export function uploadDocument(file: File): Promise<DocumentRecord> {
  const formData = new FormData();
  formData.append("file", file);
  return request<DocumentRecord>("/api/documents/upload", {
    method: "POST",
    body: formData
  });
}
