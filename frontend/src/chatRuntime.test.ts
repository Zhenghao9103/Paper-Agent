import { describe, expect, it, vi } from "vitest";

import {
  consumeSse,
  renderAnswer,
  stripInternalEvidenceMarkers,
} from "./chatRuntime";

declare global {
  var PaperMindChat: {
    consumeSse: typeof consumeSse;
    renderAnswer: typeof renderAnswer;
    stripInternalEvidenceMarkers: typeof stripInternalEvidenceMarkers;
  };
}


function responseFromChunks(chunks: string[]): Response {
  const encoder = new TextEncoder();
  return new Response(
    new ReadableStream({
      start(controller) {
        for (const chunk of chunks) controller.enqueue(encoder.encode(chunk));
        controller.close();
      },
    }),
    { status: 200 },
  );
}


describe("stripInternalEvidenceMarkers", () => {
  it("exposes the runtime API to the classic browser page", () => {
    expect(globalThis.PaperMindChat.renderAnswer).toBe(renderAnswer);
    expect(globalThis.PaperMindChat.consumeSse).toBe(consumeSse);
  });

  it("removes evidence ids and citation-matched chunk ids only", () => {
    expect(
      stripInternalEvidenceMarkers(
        "结论 [E2, E4] [1330, 1333] [1, 2]",
        [1330, 1333],
      ),
    ).toBe("结论 [1, 2]");
    expect(stripInternalEvidenceMarkers("区间 [1330, 2000]", [1330, 1333])).toBe(
      "区间 [1330, 2000]",
    );
  });

  it("does not alter markdown whitespace when no marker is removed", () => {
    const markdown = "    indented code\nline with hard break  \nnext";
    expect(stripInternalEvidenceMarkers(markdown, [1330])).toBe(markdown);
  });
});


describe("renderAnswer", () => {
  it("renders markdown and all supported math delimiters", () => {
    const target = document.createElement("div");
    renderAnswer(target, "**重点** $x^2$ $$y$$ \\(z\\) \\[w\\]", []);
    expect(target.querySelector("strong")?.textContent).toBe("重点");
    expect(target.querySelectorAll(".katex").length).toBe(4);
  });

  it("sanitizes model html and hardens links", () => {
    const target = document.createElement("div");
    renderAnswer(
      target,
      '<img src="https://tracker.invalid/pixel" onerror="alert(1)">' +
        '<span style="background:url(https://tracker.invalid/a)">unsafe</span> ' +
        '<svg><image href="https://tracker.invalid/svg"></image></svg> ' +
        '<table background="https://tracker.invalid/bg"><tr><td>cell</td></tr></table> ' +
        '[link](https://example.com)',
      [],
    );
    expect(target.innerHTML).not.toContain("onerror");
    expect(target.querySelector("img")).toBeNull();
    expect(target.querySelector("svg")).toBeNull();
    expect(target.querySelector("[style]")).toBeNull();
    expect(target.querySelector("[background]")).toBeNull();
    expect(target.querySelector("a")?.getAttribute("target")).toBe("_blank");
    expect(target.querySelector("a")?.getAttribute("rel")).toBe(
      "noopener noreferrer",
    );
  });

  it("keeps invalid latex source visible", () => {
    const target = document.createElement("div");
    renderAnswer(target, "before $\\notacommand{$ after", []);
    expect(target.textContent).toContain("before");
    expect(target.textContent).toContain("notacommand");
  });

  it("falls back to display-safe plain text when rendering throws", () => {
    let fallback = "";
    const target = {
      set innerHTML(_value: string) {
        throw new Error("renderer unavailable");
      },
      set textContent(value: string) {
        fallback = value;
      },
    } as unknown as HTMLElement;

    renderAnswer(target, "**raw** [E2]", []);

    expect(fallback).toBe("**raw**");
  });
});


describe("consumeSse", () => {
  it("parses events split across arbitrary byte chunks", async () => {
    const onEvent = vi.fn();
    const response = responseFromChunks([
      'event: status\ndata: {"type":"status","co',
      'de":"routing","message":"routing"}\n\nevent: fin',
      'al\ndata: {"type":"final_answer","answer":"ok"}\n\n',
    ]);

    await consumeSse(response, onEvent);

    expect(onEvent).toHaveBeenCalledTimes(2);
    expect(onEvent.mock.calls[1][0]).toEqual({
      type: "final_answer",
      answer: "ok",
    });
  });

  it("forwards repeated status in order and final exactly once", async () => {
    const onEvent = vi.fn();
    const response = responseFromChunks([
      'event: status\ndata: {"type":"status","code":"context"}\n\n',
      'event: status\ndata: {"type":"status","code":"routing"}\n\n',
      'event: status\ndata: {"type":"status","code":"retrieval"}\n\n',
      'event: heartbeat\ndata: {"type":"heartbeat","elapsed_seconds":5}\n\n',
      'event: final\ndata: {"type":"final_answer","answer":"ok"}\n\n',
    ]);

    await consumeSse(response, onEvent);

    expect(onEvent.mock.calls.map(([event]) => event.type)).toEqual([
      "status",
      "status",
      "status",
      "heartbeat",
      "final_answer",
    ]);
    expect(
      onEvent.mock.calls.filter(([event]) => event.type === "final_answer"),
    ).toHaveLength(1);
  });

  it("skips malformed non-final frames", async () => {
    const onEvent = vi.fn();
    const response = responseFromChunks([
      "event: status\ndata: not-json\n\n",
      'event: final\ndata: {"type":"final_answer","answer":"ok"}\n\n',
    ]);

    await consumeSse(response, onEvent);

    expect(onEvent).toHaveBeenCalledTimes(1);
  });

  it("rejects a stream that ends without a final answer", async () => {
    const response = responseFromChunks([
      'event: status\ndata: {"type":"status","code":"routing"}\n\n',
    ]);

    await expect(consumeSse(response, () => undefined)).rejects.toThrow(
      "Stream ended before final answer",
    );
  });
});
