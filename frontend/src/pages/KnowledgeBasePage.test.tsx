import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { DocumentRecord } from "../api/types";
import { KnowledgeBasePage } from "./KnowledgeBasePage";

const api = vi.hoisted(() => ({
  listDocumentChunks: vi.fn(),
  listDocumentElements: vi.fn(),
  listDocuments: vi.fn(),
  waitForDocument: vi.fn(),
  enrichDocumentFigure: vi.fn(),
  reparseDocument: vi.fn(),
  uploadDocument: vi.fn(),
  documentElementImageUrl: vi.fn((documentId, elementId) => `/image/${documentId}/${elementId}`)
}));

vi.mock("../api/documents", () => api);

const failedDocument: DocumentRecord = {
  id: 7,
  title: "Retry Paper",
  authors: null,
  year: null,
  file_type: "pdf",
  file_path: "retry.pdf",
  status: "failed",
  source_type: "uploaded",
  abstract: null,
  created_at: "2026-08-13T00:00:00Z",
  updated_at: "2026-08-13T00:00:00Z"
};

describe("KnowledgeBasePage document reparsing", () => {
  beforeEach(() => {
    for (const mock of Object.values(api)) {
      if ("mockReset" in mock) mock.mockReset();
    }
    api.documentElementImageUrl.mockImplementation(
      (documentId, elementId) => `/image/${documentId}/${elementId}`
    );
    api.waitForDocument.mockImplementation(async (_id, onStatus) => {
      const document = { ...failedDocument, status: "indexed" as const };
      onStatus?.(document);
      return document;
    });
  });

  it("reparses a failed document and refreshes the list", async () => {
    const indexedDocument: DocumentRecord = { ...failedDocument, status: "indexed" };
    api.listDocuments
      .mockResolvedValueOnce([failedDocument])
      .mockResolvedValueOnce([indexedDocument]);
    api.reparseDocument.mockResolvedValue(indexedDocument);
    render(<KnowledgeBasePage />);

    await userEvent.click(await screen.findByRole("button", { name: "重新解析" }));

    await waitFor(() => expect(api.reparseDocument).toHaveBeenCalledWith(failedDocument.id));
    expect(await screen.findByText("indexed")).toBeTruthy();
  });

  it("keeps reparse available for an indexed document", async () => {
    api.listDocuments.mockResolvedValue([{ ...failedDocument, status: "indexed" }]);
    render(<KnowledgeBasePage />);

    await screen.findByText("Retry Paper");
    expect(screen.getByRole("button", { name: "重新解析" })).toBeTruthy();
  });

  it("inspects figure, equation and table evidence", async () => {
    api.listDocuments.mockResolvedValue([{ ...failedDocument, status: "indexed" }]);
    api.listDocumentElements.mockResolvedValue([
      {
        id: 11,
        document_id: 7,
        element_uid: "figure-1",
        page_number: 2,
        element_type: "figure",
        label: "Figure 1",
        caption: "Pipeline",
        image_path: "figures/pipeline.png",
        raw_text: "An encoder pipeline.",
        structured_data: {
          figure_type: "architecture",
          components: ["Input", "Encoder"],
          relationships: ["Input enters Encoder"],
          summary: "An encoder pipeline."
        },
        parse_status: "success"
      },
      {
        id: 12,
        document_id: 7,
        element_uid: "equation-1",
        page_number: 3,
        element_type: "equation",
        label: "Eq. 16",
        caption: null,
        image_path: "equations/eq16.png",
        raw_text: "sum expression",
        structured_data: { latex: "\\sum_{k=1} P_{ik}" },
        parse_status: "success"
      },
      {
        id: 13,
        document_id: 7,
        element_uid: "table-1",
        page_number: 4,
        element_type: "table",
        label: "Table 1",
        caption: "Results",
        image_path: "tables/results.png",
        raw_text: "STG 0.77",
        structured_data: {
          columns: ["Method", "ACC"],
          rows: [{ Method: "STG", ACC: "0.77" }]
        },
        parse_status: "partial"
      }
    ]);
    api.listDocumentChunks.mockResolvedValue([]);
    render(<KnowledgeBasePage />);

    await userEvent.click(await screen.findByRole("button", { name: "查看解析内容" }));

    expect(await screen.findByText("An encoder pipeline.")).toBeTruthy();
    expect(screen.getByText("Input enters Encoder")).toBeTruthy();
    expect(screen.getByText("\\sum_{k=1} P_{ik}")).toBeTruthy();
    expect(screen.getByRole("columnheader", { name: "Method" })).toBeTruthy();
    expect(screen.getByRole("cell", { name: "STG" })).toBeTruthy();
  });

  it("shows transfer percentage and then the parsing phase", async () => {
    api.listDocuments.mockResolvedValue([]);
    let finishUpload: ((document: DocumentRecord) => void) | undefined;
    api.uploadDocument.mockImplementation((_file, callbacks) => {
      callbacks.onProgress(37);
      callbacks.onTransferComplete();
      return new Promise<DocumentRecord>((resolve) => { finishUpload = resolve; });
    });
    render(<KnowledgeBasePage />);
    const file = new File(["pdf"], "paper.pdf", { type: "application/pdf" });

    await userEvent.upload(screen.getByLabelText("Document file"), file);
    await userEvent.click(screen.getByRole("button", { name: "上传" }));

    expect(await screen.findByText("正在提取正文和表格…")).toBeTruthy();
    expect(screen.getByRole("progressbar")).toBeTruthy();
    finishUpload?.({ ...failedDocument, status: "indexed" });
    await waitFor(() => expect(api.waitForDocument).toHaveBeenCalledWith(7, expect.any(Function)));
    await waitFor(() => expect(api.listDocuments).toHaveBeenCalledTimes(2));
  });

  it("reports that chunks are available while final indexing continues", async () => {
    api.listDocuments.mockResolvedValue([]);
    api.uploadDocument.mockImplementation((_file, callbacks) => {
      callbacks.onTransferComplete();
      return Promise.resolve({ ...failedDocument, status: "parsing" });
    });
    api.waitForDocument.mockImplementation(async (_id, onStatus) => {
      onStatus({ ...failedDocument, status: "parsed" });
      return new Promise(() => undefined);
    });
    render(<KnowledgeBasePage />);

    await userEvent.upload(
      screen.getByLabelText("Document file"),
      new File(["pdf"], "paper.pdf", { type: "application/pdf" })
    );
    await userEvent.click(screen.getByRole("button", { name: "上传" }));

    expect(await screen.findByText("正文与表格 chunk 已可用，正在增强公式并建立向量索引…")).toBeTruthy();
  });

  it("stops the parsing indicator when upload or parsing fails", async () => {
    api.listDocuments.mockResolvedValue([]);
    api.uploadDocument.mockImplementation((_file, callbacks) => {
      callbacks.onTransferComplete();
      return Promise.resolve({ ...failedDocument, status: "parsing" });
    });
    api.waitForDocument.mockResolvedValue(failedDocument);
    render(<KnowledgeBasePage />);
    const file = new File(["pdf"], "broken.pdf", { type: "application/pdf" });

    await userEvent.upload(screen.getByLabelText("Document file"), file);
    await userEvent.click(screen.getByRole("button", { name: "上传" }));

    expect(await screen.findByText("上传或解析失败")).toBeTruthy();
    expect(screen.getByText("PDF 已上传，但解析失败，请检查文档后重试。")).toBeTruthy();
    expect(screen.queryByText("正在解析并建立索引…")).toBeNull();
  });

});
