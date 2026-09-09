import { FormEvent, useEffect, useState } from "react";

import {
  documentElementImageUrl,
  enrichDocumentFigure,
  listDocumentChunks,
  listDocumentElements,
  listDocuments,
  reparseDocument,
  uploadDocument,
  waitForDocument
} from "../api/documents";
import type { DocumentChunk, DocumentElement, DocumentRecord } from "../api/types";

function objectData(element: DocumentElement): Record<string, unknown> {
  return element.structured_data && !Array.isArray(element.structured_data)
    ? element.structured_data
    : {};
}

function stringList(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === "string") : [];
}

function ElementEvidence({
  element,
  onEnrich
}: {
  element: DocumentElement;
  onEnrich: (element: DocumentElement) => void;
}) {
  const data = objectData(element);
  const columns = stringList(data.columns);
  const rows = Array.isArray(data.rows)
    ? data.rows.filter((row): row is Record<string, unknown> => !!row && typeof row === "object")
    : [];
  const warnings = element.warning_codes ?? stringList(data.warning_codes);
  return (
    <article className={`evidence-strip evidence-${element.element_type}`}>
      <div className="evidence-visual">
        {element.image_path ? (
          <img
            alt={`${element.label ?? element.element_type} crop`}
            src={documentElementImageUrl(element.document_id, element.id)}
          />
        ) : (
          <span>无图像</span>
        )}
      </div>
      <div className="evidence-body">
        <header className="evidence-heading">
          <div>
            <span className="evidence-kind">{element.element_type}</span>
            <h4>{element.label ?? element.caption ?? element.element_uid}</h4>
          </div>
          <span className={`parse-mark parse-${element.parse_status}`}>{element.parse_status}</span>
        </header>
        {element.element_type === "figure" ? (
          <>
            {typeof data.summary === "string" && data.summary ? <p>{data.summary}</p> : <p>暂无结构化描述</p>}
            {stringList(data.components).length ? (
              <dl><dt>组件</dt><dd>{stringList(data.components).join(" · ")}</dd></dl>
            ) : null}
            {stringList(data.relationships).length ? (
              <dl><dt>关系</dt><dd>{stringList(data.relationships).join("；")}</dd></dl>
            ) : null}
            {!data.summary ? <button type="button" onClick={() => onEnrich(element)}>生成结构化描述</button> : null}
          </>
        ) : null}
        {element.element_type === "equation" ? (
          <pre className="formula-source">{String(data.latex || data.raw_expression || element.raw_text || "未识别公式")}</pre>
        ) : null}
        {element.element_type === "table" && columns.length ? (
          <div className="table-scroll"><table><thead><tr>{columns.map((column) => <th key={column}>{column}</th>)}</tr></thead><tbody>{rows.map((row, index) => <tr key={index}>{columns.map((column) => <td key={column}>{String(row[column] ?? "")}</td>)}</tr>)}</tbody></table></div>
        ) : null}
        {warnings.length ? <p className="evidence-warning">{warnings.join(" · ")}</p> : null}
      </div>
    </article>
  );
}

export function KnowledgeBasePage() {
  const [documents, setDocuments] = useState<DocumentRecord[]>([]);
  const [file, setFile] = useState<File | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadPhase, setUploadPhase] = useState<"idle" | "uploading" | "processing" | "parsed" | "complete" | "error">("idle");
  const [uploadProgress, setUploadProgress] = useState(0);
  const [reparsingDocumentId, setReparsingDocumentId] = useState<number | null>(null);
  const [selectedDocumentId, setSelectedDocumentId] = useState<number | null>(null);
  const [elements, setElements] = useState<DocumentElement[]>([]);
  const [chunks, setChunks] = useState<DocumentChunk[]>([]);
  const [isInspecting, setIsInspecting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function refreshDocuments() {
    setIsLoading(true);
    try {
      setDocuments(await listDocuments());
      setError(null);
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : "文档加载失败。");
    } finally {
      setIsLoading(false);
    }
  }

  async function handleUpload(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = event.currentTarget;
    if (!file) return setError("请先选择 PDF 文件。");
    setIsUploading(true);
    setUploadPhase("uploading");
    setUploadProgress(0);
    try {
      const uploadedDocument = await uploadDocument(file, {
        onProgress: setUploadProgress,
        onTransferComplete: () => {
          setUploadProgress(100);
          setUploadPhase("processing");
        }
      });
      const completedDocument = await waitForDocument(uploadedDocument.id, (document) => {
        if (document.status === "parsed") {
          setUploadPhase("parsed");
          void refreshDocuments();
        }
      });
      if (completedDocument.status === "failed") {
        throw new Error("PDF 已上传，但解析失败，请检查文档后重试。");
      }
      setFile(null);
      form.reset();
      await refreshDocuments();
      setUploadPhase("complete");
    } catch (uploadError) {
      setUploadPhase("error");
      setError(uploadError instanceof Error ? uploadError.message : "上传失败。");
    } finally {
      setIsUploading(false);
    }
  }

  async function handleEnrichFigure(element: DocumentElement) {
    setError(null);
    try {
      const enriched = await enrichDocumentFigure(element.document_id, element.id);
      setElements((current) => current.map((item) => item.id === enriched.id ? enriched : item));
    } catch (enrichError) {
      setError(enrichError instanceof Error ? enrichError.message : "图片描述生成失败。");
    }
  }

  async function handleReparse(documentId: number) {
    setReparsingDocumentId(documentId);
    setError(null);
    try {
      const document = await reparseDocument(documentId);
      if (document.status === "failed") throw new Error("重新解析失败，请检查 PDF 后重试。");
      await refreshDocuments();
      if (selectedDocumentId === documentId) await handleInspect(documentId, true);
    } catch (reparseError) {
      setError(reparseError instanceof Error ? reparseError.message : "重新解析失败。");
    } finally {
      setReparsingDocumentId(null);
    }
  }

  async function handleInspect(documentId: number, force = false) {
    if (!force && selectedDocumentId === documentId) {
      setSelectedDocumentId(null);
      return;
    }
    setIsInspecting(true);
    setError(null);
    try {
      const [nextElements, nextChunks] = await Promise.all([
        listDocumentElements(documentId),
        listDocumentChunks(documentId)
      ]);
      setElements(nextElements);
      setChunks(nextChunks);
      setSelectedDocumentId(documentId);
    } catch (inspectError) {
      setError(inspectError instanceof Error ? inspectError.message : "解析内容加载失败。");
    } finally {
      setIsInspecting(false);
    }
  }

  useEffect(() => { void refreshDocuments(); }, []);

  return (
    <div className="app-shell">
      <aside className="sidebar"><div className="brand">PaperMind</div><nav><a className="nav-active">Knowledge Base</a><span>Research Chat</span><span>arXiv Search</span><span>Memory</span></nav></aside>
      <main className="workspace">
        <header><h1>Knowledge Base</h1><p>上传学术 PDF，核对公式、图表与用于检索的 chunk。</p></header>
        <section className="panel"><h2>Upload PDF</h2><form className="upload-form" onSubmit={handleUpload}><input aria-label="Document file" accept=".pdf" disabled={isUploading} type="file" onChange={(event) => { setFile(event.target.files?.[0] ?? null); setUploadPhase("idle"); }} /><button disabled={isUploading} type="submit">{isUploading ? (uploadPhase === "processing" || uploadPhase === "parsed" ? "处理中..." : "上传中...") : "上传"}</button></form>{uploadPhase !== "idle" ? <div className={`upload-progress upload-progress-${uploadPhase}`} aria-live="polite"><div className="upload-progress-row"><span>{uploadPhase === "uploading" ? `正在上传 PDF：${uploadProgress}%` : uploadPhase === "processing" ? "正在提取正文和表格…" : uploadPhase === "parsed" ? "正文与表格 chunk 已可用，正在增强公式并建立向量索引…" : uploadPhase === "error" ? "上传或解析失败" : "上传、解析和索引已完成"}</span>{uploadPhase === "uploading" ? <strong>{uploadProgress}%</strong> : null}</div><progress aria-label="PDF 上传与解析进度" max={uploadPhase === "processing" || uploadPhase === "parsed" ? undefined : 100} value={uploadPhase === "processing" || uploadPhase === "parsed" ? undefined : uploadPhase === "complete" ? 100 : uploadPhase === "error" ? 0 : uploadProgress} /></div> : null}{error ? <p className="error">{error}</p> : null}</section>
        <section className="panel"><h2>Documents</h2>{isLoading ? <div className="empty-state">正在加载...</div> : null}{!isLoading && !documents.length ? <div className="empty-state">暂无文档。</div> : null}<div className="document-list">{documents.map((document) => <div className="document-entry" key={document.id}><article className="document-card"><div><h3>{document.title}</h3><p>{document.file_type.toUpperCase()} · {new Date(document.created_at).toLocaleDateString()}</p></div><div className="document-actions"><span className={`status status-${document.status}`}>{document.status}</span><button className="inspect-button" disabled={isInspecting} onClick={() => void handleInspect(document.id)} type="button">{selectedDocumentId === document.id ? "收起解析内容" : "查看解析内容"}</button>{document.status !== "parsing" ? <button className="reparse-button" disabled={reparsingDocumentId === document.id} onClick={() => void handleReparse(document.id)} type="button">{reparsingDocumentId === document.id ? "解析中..." : "重新解析"}</button> : null}</div></article>{selectedDocumentId === document.id ? <section className="evidence-drawer" aria-label="解析内容"><div className="evidence-summary"><span>{elements.length} 个结构元素</span><span>{chunks.length} 个 chunk</span></div>{elements.length ? elements.map((element) => <ElementEvidence element={element} key={element.id} onEnrich={(item) => void handleEnrichFigure(item)} />) : <div className="empty-state">没有检测到公式、图片或表格；正文 chunk 共 {chunks.length} 个。</div>}</section> : null}</div>)}</div></section>
      </main>
    </div>
  );
}
