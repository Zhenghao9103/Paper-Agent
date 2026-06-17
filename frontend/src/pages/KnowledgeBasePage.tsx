import { FormEvent, useEffect, useState } from "react";

import { listDocuments, uploadDocument } from "../api/documents";
import type { DocumentRecord } from "../api/types";

export function KnowledgeBasePage() {
  const [documents, setDocuments] = useState<DocumentRecord[]>([]);
  const [file, setFile] = useState<File | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function refreshDocuments() {
    setIsLoading(true);
    try {
      setDocuments(await listDocuments());
      setError(null);
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : "Failed to load documents.");
    } finally {
      setIsLoading(false);
    }
  }

  async function handleUpload(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!file) {
      setError("Choose a PDF or PPTX file first.");
      return;
    }
    setIsUploading(true);
    try {
      await uploadDocument(file);
      setFile(null);
      event.currentTarget.reset();
      await refreshDocuments();
    } catch (uploadError) {
      setError(uploadError instanceof Error ? uploadError.message : "Upload failed.");
    } finally {
      setIsUploading(false);
    }
  }

  useEffect(() => {
    void refreshDocuments();
  }, []);

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">PaperMind</div>
        <nav>
          <a className="nav-active">Knowledge Base</a>
          <span>Research Chat</span>
          <span>arXiv Search</span>
          <span>Memory</span>
        </nav>
      </aside>
      <main className="workspace">
        <header>
          <h1>Knowledge Base</h1>
          <p>Upload papers and slides to build your local research knowledge base.</p>
        </header>

        <section className="panel">
          <h2>Upload PDF or PPTX</h2>
          <form className="upload-form" onSubmit={handleUpload}>
            <input
              aria-label="Document file"
              accept=".pdf,.pptx"
              type="file"
              onChange={(event) => setFile(event.target.files?.[0] ?? null)}
            />
            <button disabled={isUploading} type="submit">
              {isUploading ? "Uploading..." : "Upload"}
            </button>
          </form>
          {error ? <p className="error">{error}</p> : null}
        </section>

        <section className="panel">
          <h2>Documents</h2>
          {isLoading ? <div className="empty-state">Loading documents...</div> : null}
          {!isLoading && documents.length === 0 ? (
            <div className="empty-state">No documents yet.</div>
          ) : null}
          <div className="document-list">
            {documents.map((document) => (
              <article className="document-card" key={document.id}>
                <div>
                  <h3>{document.title}</h3>
                  <p>
                    {document.file_type.toUpperCase()} -{" "}
                    {new Date(document.created_at).toLocaleDateString()}
                  </p>
                </div>
                <span className={`status status-${document.status}`}>{document.status}</span>
              </article>
            ))}
          </div>
        </section>
      </main>
    </div>
  );
}
