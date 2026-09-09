import DOMPurify from "dompurify";
import renderMathInElement from "katex/contrib/auto-render";
import { marked } from "marked";


export type ChatStreamEvent = {
  type: string;
  [key: string]: unknown;
};

const EVIDENCE_MARKER = /\[\s*E\d+(?:\s*,\s*E\d+)*\s*\]/gi;
const NUMERIC_MARKER = /\[\s*\d+(?:\s*,\s*\d+)*\s*\]/g;
const MATH_EXPRESSION =
  /\$\$[\s\S]+?\$\$|\\\[[\s\S]+?\\\]|\\\([\s\S]+?\\\)|\$(?!\$)[^\n$]+?\$/g;
const REMOVED_MARKER = "\u0000PAPERMIND_REMOVED\u0000";


function normalizeRemovedMarkers(value: string): string {
  return value.replace(
    /(?:[ \t]*\u0000PAPERMIND_REMOVED\u0000[ \t]*)+/g,
    (match, offset: number, source: string) => {
      const previous = offset > 0 ? source[offset - 1] : "";
      const next = source[offset + match.length] ?? "";
      if (!previous || previous === "\n" || !next || next === "\n") return "";
      if (/[,.;:!?，。；：！？)\]}]/.test(next)) return "";
      return " ";
    },
  );
}


export function stripInternalEvidenceMarkers(
  answer: string,
  citationChunkIds: number[],
): string {
  const allowedChunkIds = new Set(
    citationChunkIds.filter((value) => Number.isInteger(value)).map(Number),
  );
  const withoutEvidenceIds = String(answer).replace(EVIDENCE_MARKER, REMOVED_MARKER);
  const withoutChunkIds = withoutEvidenceIds.replace(NUMERIC_MARKER, (marker) => {
    if (allowedChunkIds.size === 0) return marker;
    const ids = marker
      .slice(1, -1)
      .split(",")
      .map((value) => Number(value.trim()));
    return ids.length > 0 && ids.every((value) => allowedChunkIds.has(value))
      ? REMOVED_MARKER
      : marker;
  });
  return normalizeRemovedMarkers(withoutChunkIds);
}


function protectMath(source: string): { markdown: string; expressions: string[] } {
  const expressions: string[] = [];
  const markdown = source.replace(MATH_EXPRESSION, (expression) => {
    const index = expressions.push(expression) - 1;
    return `PAPERMINDMATHEXPRESSION${index}TOKEN`;
  });
  return { markdown, expressions };
}


function restoreMath(target: HTMLElement, expressions: string[]): void {
  if (expressions.length === 0) return;
  const token = /PAPERMINDMATHEXPRESSION(\d+)TOKEN/g;
  const walker = document.createTreeWalker(target, NodeFilter.SHOW_TEXT);
  const nodes: Text[] = [];
  while (walker.nextNode()) nodes.push(walker.currentNode as Text);
  for (const node of nodes) {
    node.textContent = (node.textContent ?? "").replace(token, (match, rawIndex) => {
      return expressions[Number(rawIndex)] ?? match;
    });
  }
}


export function renderAnswer(
  target: HTMLElement,
  answer: string,
  citationChunkIds: number[],
): void {
  const display = stripInternalEvidenceMarkers(answer, citationChunkIds);
  try {
    const protectedMath = protectMath(display);
    const parsed = marked.parse(protectedMath.markdown, { async: false });
    target.innerHTML = DOMPurify.sanitize(String(parsed), {
      USE_PROFILES: { html: true },
      FORBID_TAGS: ["img", "picture", "source", "video", "audio", "style"],
      FORBID_ATTR: ["style", "src", "srcset", "background"],
    });
    restoreMath(target, protectedMath.expressions);
    for (const link of target.querySelectorAll("a")) {
      link.setAttribute("target", "_blank");
      link.setAttribute("rel", "noopener noreferrer");
    }
    renderMathInElement(target, {
      delimiters: [
        { left: "$$", right: "$$", display: true },
        { left: "\\[", right: "\\]", display: true },
        { left: "\\(", right: "\\)", display: false },
        { left: "$", right: "$", display: false },
      ],
      throwOnError: false,
      ignoredTags: ["script", "noscript", "style", "textarea", "pre", "code"],
    });
  } catch {
    target.textContent = display;
  }
}


function parseFrame(frame: string): { eventName: string; data: string } | null {
  let eventName = "message";
  const dataLines: string[] = [];
  for (const line of frame.split("\n")) {
    if (line.startsWith("event:")) eventName = line.slice(6).trim();
    if (line.startsWith("data:")) dataLines.push(line.slice(5).trimStart());
  }
  return dataLines.length > 0 ? { eventName, data: dataLines.join("\n") } : null;
}


export async function consumeSse(
  response: Response,
  onEvent: (event: ChatStreamEvent) => void,
): Promise<void> {
  if (!response.body) throw new Error("Streaming response body is unavailable");
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let terminalSeen = false;

  const consumeFrame = (rawFrame: string): void => {
    const parsed = parseFrame(rawFrame);
    if (!parsed) return;
    try {
      const event = JSON.parse(parsed.data) as ChatStreamEvent;
      onEvent(event);
      if (parsed.eventName === "final" || parsed.eventName === "error") {
        terminalSeen = true;
      }
    } catch (error) {
      if (parsed.eventName === "final") throw error;
    }
  };

  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value, { stream: !done });
    buffer = buffer.replace(/\r\n/g, "\n");
    let boundary = buffer.indexOf("\n\n");
    while (boundary >= 0) {
      consumeFrame(buffer.slice(0, boundary));
      buffer = buffer.slice(boundary + 2);
      boundary = buffer.indexOf("\n\n");
    }
    if (done) break;
  }
  if (buffer.trim()) consumeFrame(buffer);
  if (!terminalSeen) throw new Error("Stream ended before final answer");
}


globalThis.PaperMindChat = {
  consumeSse,
  renderAnswer,
  stripInternalEvidenceMarkers,
};

declare global {
  var PaperMindChat: {
    consumeSse: typeof consumeSse;
    renderAnswer: typeof renderAnswer;
    stripInternalEvidenceMarkers: typeof stripInternalEvidenceMarkers;
  };
}
