import { request } from "./client";
import type { DocumentChunk, DocumentElement, DocumentRecord } from "./types";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

export interface UploadProgressCallbacks {
  onProgress?: (percentage: number) => void;
  onTransferComplete?: () => void;
}

export function listDocuments(): Promise<DocumentRecord[]> {
  return request<DocumentRecord[]>("/api/documents");
}

export function getDocument(documentId: number): Promise<DocumentRecord> {
  return request<DocumentRecord>(`/api/documents/${documentId}`);
}

export async function waitForDocument(
  documentId: number,
  onStatus?: (document: DocumentRecord) => void,
  intervalMs = 1000,
  maxAttempts = 1800
): Promise<DocumentRecord> {
  for (let attempt = 0; attempt < maxAttempts; attempt += 1) {
    const document = await getDocument(documentId);
    onStatus?.(document);
    if (["indexed", "analyzed", "failed"].includes(document.status)) return document;
    await new Promise((resolve) => window.setTimeout(resolve, intervalMs));
  }
  throw new Error("文档处理超时，请稍后刷新查看状态。");
}

export function uploadDocument(
  file: File,
  callbacks: UploadProgressCallbacks = {}
): Promise<DocumentRecord> {
  const formData = new FormData();
  formData.append("file", file);
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API_BASE_URL}/api/documents/upload-async`);
    xhr.responseType = "json";
    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable && event.total > 0) {
        callbacks.onProgress?.(Math.min(100, Math.round((event.loaded / event.total) * 100)));
      }
    };
    xhr.upload.onload = () => callbacks.onTransferComplete?.();
    xhr.onerror = () => reject(new Error("上传连接中断，请重试。"));
    xhr.onload = () => {
      const payload = xhr.response ?? (() => {
        try { return JSON.parse(xhr.responseText); } catch { return null; }
      })();
      if (xhr.status >= 200 && xhr.status < 300 && payload) {
        resolve(payload as DocumentRecord);
        return;
      }
      const detail = payload && typeof payload === "object" && "detail" in payload
        ? String(payload.detail)
        : xhr.responseText || `上传失败（${xhr.status}）`;
      reject(new Error(detail));
    };
    xhr.send(formData);
  });
}

export function reparseDocument(documentId: number): Promise<DocumentRecord> {
  return request<DocumentRecord>(`/api/documents/${documentId}/reparse`, {
    method: "POST"
  });
}

export function listDocumentElements(documentId: number): Promise<DocumentElement[]> {
  return request<DocumentElement[]>(`/api/documents/${documentId}/elements`);
}

export function listDocumentChunks(documentId: number): Promise<DocumentChunk[]> {
  return request<DocumentChunk[]>(`/api/documents/${documentId}/chunks`);
}

export function documentElementImageUrl(documentId: number, elementId: number): string {
  return `/api/documents/${documentId}/elements/${elementId}/image`;
}

export function enrichDocumentFigure(
  documentId: number,
  elementId: number
): Promise<DocumentElement> {
  return request<DocumentElement>(
    `/api/documents/${documentId}/elements/${elementId}/enrich`,
    { method: "POST" }
  );
}
