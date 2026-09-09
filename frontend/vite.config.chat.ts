import { cpSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { defineConfig } from "vite";

const outputDirectory = resolve(__dirname, "../backend/app/static/chat");

export default defineConfig({
  plugins: [
    {
      name: "copy-chat-runtime-assets",
      closeBundle() {
        mkdirSync(resolve(outputDirectory, "fonts"), { recursive: true });
        cpSync(
          resolve(__dirname, "node_modules/katex/dist/katex.min.css"),
          resolve(outputDirectory, "chat-runtime.css"),
        );
        cpSync(
          resolve(__dirname, "node_modules/katex/dist/fonts"),
          resolve(outputDirectory, "fonts"),
          { recursive: true },
        );
        const licenses = [
          ["marked 15.0.12", "node_modules/marked/LICENSE.md"],
          ["DOMPurify 3.4.15", "node_modules/dompurify/LICENSE"],
          ["KaTeX 0.16.22", "node_modules/katex/LICENSE"],
        ]
          .map(
            ([name, path]) =>
              `${name}\n${"=".repeat(name.length)}\n\n${readFileSync(resolve(__dirname, path), "utf8").trim()}\n`,
          )
          .join("\n");
        writeFileSync(
          resolve(outputDirectory, "THIRD_PARTY_LICENSES.txt"),
          licenses,
          "utf8",
        );
      },
    },
  ],
  build: {
    outDir: outputDirectory,
    emptyOutDir: true,
    lib: {
      entry: "src/chatRuntime.ts",
      name: "PaperMindChat",
      formats: ["iife"],
      fileName: () => "chat-runtime.js",
      cssFileName: "chat-runtime",
    },
  },
});
