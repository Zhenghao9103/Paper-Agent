from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["web"])


@router.get("/", response_class=HTMLResponse)
def index() -> str:
    return """
<!doctype html>
<html lang="zh-CN">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>PaperMind Agent</title>
    <style>
      :root {
        --bg: #eef1f4;
        --paper: #fbfaf6;
        --surface: #fffdf8;
        --surface-soft: #f5f1e8;
        --surface-strong: #e6edf0;
        --line: #d5d0c4;
        --line-soft: #ebe5d8;
        --text: #17212b;
        --muted: #657386;
        --brand: #123a4a;
        --brand-strong: #0b2532;
        --brand-soft: #dcecee;
        --accent: #b88735;
        --accent-soft: #f1dfb8;
        --ok: #24745a;
        --warn: #8b651b;
        --danger: #9a3b43;
        --shadow: 0 22px 55px rgba(18, 36, 50, 0.12);
        color: var(--text);
        background: var(--bg);
        font-family: "Inter", "IBM Plex Sans", ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      }

      * { box-sizing: border-box; }
      body {
        background:
          linear-gradient(rgba(18, 58, 74, 0.045) 1px, transparent 1px),
          linear-gradient(90deg, rgba(18, 58, 74, 0.035) 1px, transparent 1px),
          var(--bg);
        background-size: 32px 32px, 32px 32px, auto;
        margin: 0;
      }
      button, input, textarea { font: inherit; }
      button {
        align-items: center;
        border: 0;
        border-radius: 5px;
        background: var(--brand);
        color: #fff;
        cursor: pointer;
        display: inline-flex;
        font-weight: 650;
        justify-content: center;
        min-height: 36px;
        padding: 9px 14px;
        white-space: nowrap;
      }
      button:hover { background: var(--brand-strong); }
      button:focus-visible, input:focus-visible, textarea:focus-visible {
        outline: 2px solid var(--accent);
        outline-offset: 2px;
      }
      button:disabled { background: #99aeb6; cursor: not-allowed; }
      button.secondary {
        background: #edf2f3;
        border: 1px solid #d7e1e4;
        color: #22323d;
      }
      button.secondary:hover { background: #e1ebee; }
      button.danger { background: var(--danger); color: #fff; }
      button.danger:hover { background: #842b2b; }
      input, textarea {
        background: #fff;
        border: 1px solid var(--line);
        border-radius: 5px;
        color: var(--text);
        outline: none;
        padding: 10px 11px;
      }
      input:focus, textarea:focus { border-color: var(--brand); box-shadow: 0 0 0 3px rgba(18, 58, 74, 0.12); }
      textarea { min-height: 132px; resize: vertical; width: 100%; }
      a { color: var(--brand); text-decoration: none; }
      a:hover { text-decoration: underline; }

      .shell {
        display: grid;
        grid-template-columns: 252px minmax(0, 1fr);
        min-height: 100vh;
      }
      .sidebar {
        background: var(--brand-strong);
        border-right: 0;
        color: #e7f0f2;
        display: grid;
        grid-template-rows: auto 1fr;
        padding: 24px 18px;
        position: relative;
      }
      .sidebar::after {
        background: linear-gradient(180deg, var(--accent), rgba(184, 135, 53, 0.08));
        content: "";
        height: 100%;
        position: absolute;
        right: 0;
        top: 0;
        width: 3px;
      }
      .brand {
        border-bottom: 1px solid rgba(255, 255, 255, 0.12);
        display: grid;
        gap: 8px;
        margin-bottom: 20px;
        padding: 0 4px 20px;
      }
      .brand strong {
        color: #fff8e8;
        font-family: Georgia, "Times New Roman", serif;
        font-size: 24px;
        font-weight: 700;
        letter-spacing: 0;
      }
      .brand span { color: #a9bbc4; font-size: 12px; line-height: 1.55; }
      .nav { display: grid; gap: 8px; align-content: start; }
      .nav button {
        background: transparent;
        border: 1px solid transparent;
        color: #cbdae0;
        display: grid;
        font-weight: 600;
        grid-template-columns: 34px minmax(0, 1fr);
        justify-content: start;
        min-height: 46px;
        padding: 10px 12px;
        position: relative;
        text-align: left;
        width: 100%;
      }
      .nav button:hover {
        background: rgba(255, 255, 255, 0.07);
        border-color: rgba(255, 255, 255, 0.1);
      }
      .nav button.active {
        background: #fff8e8;
        color: var(--brand-strong);
        box-shadow: 0 12px 24px rgba(0, 0, 0, 0.18);
      }
      .nav-icon {
        color: var(--accent);
        font-family: "IBM Plex Mono", Consolas, monospace;
        font-weight: 800;
      }
      .workspace {
        display: grid;
        gap: 20px;
        max-width: 1320px;
        padding: 30px 34px 44px;
        width: 100%;
      }
      .topbar {
        align-items: center;
        background: rgba(251, 250, 246, 0.84);
        border: 1px solid rgba(213, 208, 196, 0.72);
        border-radius: 8px;
        box-shadow: 0 16px 36px rgba(18, 36, 50, 0.08);
        display: grid;
        gap: 16px;
        grid-template-columns: minmax(0, 1fr) auto;
        padding: 18px 20px;
      }
      .top-actions {
        align-items: end;
        display: grid;
        gap: 8px;
        justify-items: end;
      }
      .model-status {
        color: var(--muted);
        font-size: 12px;
        line-height: 1.45;
        max-width: 360px;
        text-align: right;
      }
      h1 {
        font-family: Georgia, "Times New Roman", serif;
        font-size: 34px;
        letter-spacing: 0;
        margin: 0;
      }
      h2 { font-size: 17px; margin: 0 0 14px; }
      h3 { font-size: 15px; margin: 0 0 7px; }
      p { color: var(--muted); line-height: 1.55; margin: 6px 0 0; }
      .view { display: none; }
      .view.active { display: grid; gap: 18px; }
      .grid-2 {
        display: grid;
        gap: 16px;
        grid-template-columns: minmax(0, 1fr) minmax(360px, 0.86fr);
      }
      .panel {
        background: var(--surface);
        border: 1px solid rgba(213, 208, 196, 0.86);
        border-radius: 7px;
        box-shadow: var(--shadow);
        padding: 20px;
        position: relative;
      }
      .panel::before {
        background: var(--accent);
        content: "";
        height: 3px;
        left: 20px;
        position: absolute;
        top: -1px;
        width: 52px;
      }
      .panel-header {
        align-items: start;
        display: flex;
        gap: 12px;
        justify-content: space-between;
        margin-bottom: 14px;
      }
      .panel-header h2 { margin: 0; }
      .panel-header p { font-size: 13px; margin-top: 4px; }
      .stack { display: grid; gap: 12px; }
      .row { align-items: center; display: flex; gap: 10px; }
      .row input { flex: 1; min-width: 0; }
      .toolbar { display: flex; flex-wrap: wrap; gap: 8px; }
      .upload-box {
        background:
          repeating-linear-gradient(135deg, rgba(184, 135, 53, 0.08) 0 1px, transparent 1px 12px),
          var(--surface-soft);
        border: 1px dashed #b7aa8f;
        border-radius: 7px;
        display: grid;
        gap: 12px;
        padding: 20px;
      }
      .document-list, .result-list { display: grid; gap: 10px; }
      .pagination {
        align-items: center;
        border-top: 1px solid var(--line-soft);
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        justify-content: space-between;
        margin-top: 14px;
        padding-top: 14px;
      }
      .page-status {
        color: var(--muted);
        font-size: 13px;
      }
      .document-card, .item {
        background: #fff;
        border: 1px solid var(--line-soft);
        border-radius: 7px;
        padding: 14px;
      }
      .document-card {
        align-items: center;
        display: grid;
        gap: 12px;
        grid-template-columns: minmax(0, 1fr) auto;
      }
      .document-card:hover {
        border-color: #d0b980;
        box-shadow: 0 10px 26px rgba(18, 36, 50, 0.08);
      }
      .document-title {
        align-items: center;
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
      }
      .document-title h3 { margin: 0; }
      .document-actions { display: flex; flex-wrap: wrap; gap: 8px; justify-content: flex-end; }
      .meta, .muted { color: var(--muted); font-size: 13px; }
      .badge, .status {
        border-radius: 999px;
        display: inline-flex;
        font-size: 12px;
        font-weight: 750;
        line-height: 1;
        padding: 5px 9px;
        text-transform: uppercase;
      }
      .badge { background: #eef3f4; color: #415266; }
      .status { background: #fff4d7; color: var(--warn); }
      .status-indexed, .status-analyzed { background: #ddf5e8; color: var(--ok); }
      .status-failed { background: #ffe1e1; color: var(--danger); }
      .empty-state {
        border: 1px dashed #c8d3df;
        border-radius: 7px;
        color: var(--muted);
        padding: 22px;
        text-align: center;
      }
      .error { color: var(--danger); margin: 8px 0 0; }
      pre {
        color: #314254;
        font-family: inherit;
        line-height: 1.55;
        margin: 8px 0 0;
        max-height: 360px;
        overflow: auto;
        white-space: pre-wrap;
      }
      .answer, .paper-result {
        background: var(--surface-soft);
        border: 1px solid var(--line-soft);
        border-radius: 7px;
        margin-top: 12px;
        padding: 14px;
        white-space: pre-wrap;
      }
      .citation { color: var(--muted); font-size: 13px; margin-top: 8px; }
      .trace {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        margin-top: 12px;
      }
      .trace span {
        background: #e7f1f5;
        border-radius: 999px;
        color: var(--brand-strong);
        font-size: 12px;
        padding: 5px 8px;
      }
      .chat-shell {
        background:
          linear-gradient(90deg, rgba(184, 135, 53, 0.06) 1px, transparent 1px),
          #f7f5ef;
        background-size: 44px 44px;
        border: 1px solid #ded5c4;
        border-radius: 7px;
        display: grid;
        grid-template-rows: minmax(360px, 1fr) auto;
        min-height: 560px;
        overflow: hidden;
      }
      .chat-messages {
        align-content: start;
        display: grid;
        gap: 14px;
        max-height: 58vh;
        overflow: auto;
        padding: 18px;
      }
      .chat-empty {
        align-self: center;
        color: var(--muted);
        line-height: 1.7;
        margin: 0 auto;
        max-width: 620px;
        text-align: center;
      }
      .message {
        display: grid;
        gap: 6px;
        max-width: min(760px, 82%);
      }
      .message.user { justify-self: end; }
      .message.agent { justify-self: start; }
      .message-role {
        color: var(--muted);
        font-size: 12px;
        font-weight: 700;
      }
      .message.user .message-role { text-align: right; }
      .bubble {
        border: 1px solid var(--line-soft);
        border-radius: 7px;
        line-height: 1.62;
        padding: 12px 14px;
        white-space: pre-wrap;
      }
      .message.user .bubble {
        background: var(--brand);
        border-color: var(--brand);
        color: #fff;
      }
      .message.agent .bubble {
        background: #fffdf8;
        border-color: #ded7c8;
        color: var(--text);
        box-shadow: 0 8px 22px rgba(18, 36, 50, 0.07);
      }
      .message.pending .bubble { color: var(--muted); }
      .chat-composer {
        background: #fffdf8;
        border-top: 1px solid #ded5c4;
        display: grid;
        gap: 10px;
        grid-template-columns: minmax(0, 1fr) auto;
        padding: 12px;
      }
      .chat-composer textarea {
        min-height: 54px;
        max-height: 150px;
      }
      .analysis-grid {
        display: grid;
        gap: 12px;
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }
      .analysis-grid .item:first-child { grid-column: 1 / -1; }
      .kpi-grid {
        display: grid;
        gap: 10px;
        grid-template-columns: repeat(4, minmax(0, 1fr));
      }
      .kpi {
        background: #fffdf8;
        border: 1px solid var(--line-soft);
        border-radius: 7px;
        padding: 13px;
        position: relative;
      }
      .kpi::after {
        background: var(--accent-soft);
        bottom: 0;
        content: "";
        height: 3px;
        left: 0;
        position: absolute;
        width: 100%;
      }
      .kpi strong {
        display: block;
        font-family: Georgia, "Times New Roman", serif;
        font-size: 28px;
      }
      .kpi span { color: var(--muted); font-size: 12px; }

      @media (max-width: 980px) {
        .shell { grid-template-columns: 1fr; }
        .sidebar {
          border-bottom: 1px solid var(--line);
          border-right: 0;
          grid-template-rows: auto auto;
        }
        .nav { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        .grid-2, .analysis-grid, .kpi-grid { grid-template-columns: 1fr; }
        .topbar, .row { align-items: stretch; grid-template-columns: 1fr; flex-direction: column; }
        .chat-shell { min-height: 520px; }
        .chat-messages { max-height: 62vh; padding: 14px; }
        .chat-composer { grid-template-columns: 1fr; }
        .message { max-width: 94%; }
        .document-card { grid-template-columns: 1fr; }
        .document-actions { justify-content: flex-start; }
      }
    </style>
  </head>
  <body class="research-workbench">
    <div class="shell">
      <aside class="sidebar">
        <div>
          <div class="brand">
            <strong>PaperMind</strong>
            <span>个人论文知识库与 RAG 研究助手</span>
          </div>
          <nav class="nav">
            <button class="active" data-view-button="kb" type="button"><span class="nav-icon">01</span><span>知识库</span></button>
            <button data-view-button="chat" type="button"><span class="nav-icon">02</span><span>研究问答</span></button>
            <button data-view-button="arxiv" type="button"><span class="nav-icon">03</span><span>arXiv 检索</span></button>
          </nav>
        </div>
      </aside>

      <main class="workspace">
        <header class="topbar">
          <div>
            <h1 id="view-title">知识库</h1>
            <p id="view-subtitle">上传论文或 PPT，解析页面内容，并生成结构化中文分析。</p>
          </div>
          <div class="top-actions">
            <div class="toolbar">
              <button class="secondary" onclick="refreshAll()" type="button">刷新数据</button>
              <button class="secondary" id="llm-test-button" onclick="testModel()" type="button">测试模型</button>
            </div>
            <div id="model-status" class="model-status">正在检查模型配置...</div>
          </div>
        </header>

        <section class="view active" data-view="kb">
          <section class="panel">
            <div class="kpi-grid">
              <div class="kpi"><strong id="kpi-documents">0</strong><span>文档</span></div>
              <div class="kpi"><strong id="kpi-indexed">0</strong><span>已入库</span></div>
              <div class="kpi"><strong id="kpi-analyzed">0</strong><span>已分析</span></div>
            </div>
          </section>

          <div class="grid-2">
            <section class="panel">
              <div class="panel-header">
                <div>
                  <h2>上传资料</h2>
                  <p>支持 PDF 论文与 PPTX 汇报材料，上传后会自动解析并切块。</p>
                </div>
              </div>
              <form class="upload-box" id="upload-form">
                <input id="file-input" accept=".pdf,.pptx" type="file" />
                <button id="upload-button" type="submit">上传并解析</button>
              </form>
              <p class="error" id="error"></p>
            </section>
          </div>

          <section class="panel">
            <div class="panel-header">
              <div>
                <h2>文档列表</h2>
                <p>点击“分析”生成中文论文笔记；解析文本会在后台用于 RAG 检索。</p>
              </div>
            </div>
            <div class="toolbar">
              <button class="danger" id="clear-data-button" onclick="clearAllData()" type="button">清空全部数据</button>
            </div>
            <div id="documents" class="document-list"></div>
            <div class="pagination" id="document-pagination">
              <button class="secondary" id="document-prev-button" onclick="changeDocumentPage(-1)" type="button">上一页</button>
              <span class="page-status" id="document-page-status">第 1 / 1 页</span>
              <button class="secondary" id="document-next-button" onclick="changeDocumentPage(1)" type="button">下一页</button>
            </div>
          </section>

          <section class="panel">
            <div class="panel-header">
              <div>
                <h2>论文分析</h2>
                <p>摘要、创新点、方法、实验、局限性和图表说明会在这里展示。</p>
              </div>
            </div>
            <div id="analysis" class="analysis-grid">
              <div class="empty-state">点击文档上的“分析”生成中文笔记。</div>
            </div>
          </section>
        </section>

        <section class="view" data-view="chat">
          <section class="panel">
            <div class="panel-header">
              <div>
                <h2>研究问答</h2>
                <p>基于本地知识库回答问题，并显示引用来源与 LangGraph 执行轨迹。</p>
              </div>
            </div>
            <div class="chat-shell">
              <div class="chat-messages" id="chat-messages">
                <p class="chat-empty">可以像聊天一样连续追问论文内容，例如：这篇论文的创新点是什么？方法和已有方法相比有什么区别？实验结论可靠吗？</p>
              </div>
              <form class="chat-composer" id="chat-form">
                <textarea id="question-input" placeholder="输入你的问题，按 Ctrl + Enter 发送"></textarea>
                <button id="ask-button" type="submit">发送</button>
              </form>
            </div>
            <div id="answer" hidden></div>
          </section>
        </section>

        <section class="view" data-view="arxiv">
          <section class="panel">
            <div class="panel-header">
              <div>
                <h2>arXiv 智能检索</h2>
                <p>按关键词搜索相关论文，用于补充本地知识库之外的研究背景。</p>
              </div>
            </div>
            <form class="row" id="arxiv-form">
              <input id="arxiv-input" placeholder="例如：multimodal RAG agent" />
              <button id="arxiv-button" type="submit">检索论文</button>
            </form>
            <div id="arxiv-results" class="result-list"></div>
          </section>
        </section>

      </main>
    </div>

    <script>
      const viewCopy = {
        kb: ["知识库", "上传论文或 PPT，解析页面内容，并生成结构化中文分析。"],
        chat: ["研究问答", "围绕本地论文知识库进行带引用的 RAG 问答。"],
        arxiv: ["arXiv 检索", "检索外部相关论文，补充研究背景与选题线索。"]
      };

      const list = document.getElementById("documents");
      const analysis = document.getElementById("analysis");
      const error = document.getElementById("error");
      const answer = document.getElementById("answer");
      const chatMessages = document.getElementById("chat-messages");
      const questionInput = document.getElementById("question-input");
      const chatForm = document.getElementById("chat-form");
      const askButton = document.getElementById("ask-button");
      const arxivForm = document.getElementById("arxiv-form");
      const arxivInput = document.getElementById("arxiv-input");
      const arxivButton = document.getElementById("arxiv-button");
      const arxivResults = document.getElementById("arxiv-results");
      const form = document.getElementById("upload-form");
      const input = document.getElementById("file-input");
      const button = document.getElementById("upload-button");
      const clearDataButton = document.getElementById("clear-data-button");
      const documentPagination = document.getElementById("document-pagination");
      const documentPageStatus = document.getElementById("document-page-status");
      const documentPrevButton = document.getElementById("document-prev-button");
      const documentNextButton = document.getElementById("document-next-button");
      const documentsPerPage = 5;
      let currentSessionId = null;
      let cachedDocuments = [];
      let currentDocumentPage = 1;
      const modelStatus = document.getElementById("model-status");
      const llmTestButton = document.getElementById("llm-test-button");

      document.querySelectorAll("[data-view-button]").forEach((button) => {
        button.addEventListener("click", () => showView(button.dataset.viewButton));
      });

      function showView(name) {
        document.querySelectorAll("[data-view-button]").forEach((button) => {
          button.classList.toggle("active", button.dataset.viewButton === name);
        });
        document.querySelectorAll("[data-view]").forEach((view) => {
          view.classList.toggle("active", view.dataset.view === name);
        });
        document.getElementById("view-title").textContent = viewCopy[name][0];
        document.getElementById("view-subtitle").textContent = viewCopy[name][1];
      }

      async function refreshAll() {
        await loadDocuments();
        await loadModelStatus();
      }

      async function clearAllData() {
        const confirmed = window.confirm("确定清空所有文档、问答历史、记忆和向量索引吗？此操作不可恢复。");
        if (!confirmed) return;

        clearDataButton.disabled = true;
        clearDataButton.textContent = "清空中...";
        error.textContent = "";
        try {
          const response = await fetch("/api/documents/clear", { method: "POST" });
          if (!response.ok) {
            const message = await readErrorMessage(response);
            throw new Error(message || "清空数据失败");
          }
          cachedDocuments = [];
          currentDocumentPage = 1;
          updateStats();
          renderDocumentPage();
          analysis.innerHTML = '<div class="empty-state">暂无分析结果。</div>';
          answer.innerHTML = "";
          resetChatMessages();
          currentSessionId = null;
          input.value = "";
          error.textContent = "已清空数据库、问答历史、记忆和向量索引。";
        } catch (clearError) {
          error.textContent = clearError.message || "清空数据失败";
        } finally {
          clearDataButton.disabled = false;
          clearDataButton.textContent = "清空全部数据";
        }
      }

      async function loadModelStatus() {
        try {
          const response = await fetch("/api/health/llm");
          const status = await response.json();
          modelStatus.textContent = status.key_configured
            ? `模型已配置：${status.model} · ${status.base_url}`
            : "未读取到 OPENAI_API_KEY，当前会使用本地规则兜底分析。";
        } catch (statusError) {
          modelStatus.textContent = "模型配置检查失败。";
        }
      }

      async function testModel() {
        llmTestButton.disabled = true;
        llmTestButton.textContent = "测试中...";
        modelStatus.textContent = "正在请求模型...";
        try {
          const response = await fetch("/api/health/llm?ping=true");
          const result = await response.json();
          modelStatus.textContent = result.ok
            ? `模型调用成功：${result.model} 返回 ${result.reply || "OK"}`
            : `模型调用失败：${result.error || "未知错误"}`;
        } catch (modelError) {
          modelStatus.textContent = `模型调用失败：${modelError.message || "请求异常"}`;
        } finally {
          llmTestButton.disabled = false;
          llmTestButton.textContent = "测试模型";
        }
      }

      async function loadDocuments() {
        error.textContent = "";
        const response = await fetch("/api/documents");
        cachedDocuments = await response.json();
        currentDocumentPage = 1;
        updateStats();
        renderDocumentPage();
      }

      function renderDocumentPage() {
        if (!cachedDocuments.length) {
          list.innerHTML = '<div class="empty-state">还没有文档，先上传一篇论文或 PPT。</div>';
          documentPagination.hidden = true;
          return;
        }
        documentPagination.hidden = cachedDocuments.length <= documentsPerPage;
        const totalPages = Math.max(1, Math.ceil(cachedDocuments.length / documentsPerPage));
        currentDocumentPage = Math.min(Math.max(currentDocumentPage, 1), totalPages);
        const start = (currentDocumentPage - 1) * documentsPerPage;
        const pageDocuments = cachedDocuments.slice(start, start + documentsPerPage);
        list.innerHTML = pageDocuments.map((document) => `
          <article class="document-card">
            <div>
              <div class="document-title">
                <h3>${escapeHtml(document.title)}</h3>
                <span class="badge">${document.file_type.toUpperCase()}</span>
              </div>
              <div class="meta">上传时间：${new Date(document.created_at).toLocaleDateString()}</div>
            </div>
            <div class="document-actions">
              <button onclick="analyzeDocument(${document.id})" type="button">分析</button>
              <span class="status status-${document.status}">${statusText(document.status)}</span>
            </div>
          </article>
        `).join("");
        documentPageStatus.textContent = `第 ${currentDocumentPage} / ${totalPages} 页 · 共 ${cachedDocuments.length} 个文档`;
        documentPrevButton.disabled = currentDocumentPage <= 1;
        documentNextButton.disabled = currentDocumentPage >= totalPages;
      }

      function changeDocumentPage(delta) {
        currentDocumentPage += delta;
        renderDocumentPage();
      }

      function updateStats() {
        document.getElementById("kpi-documents").textContent = cachedDocuments.length;
        document.getElementById("kpi-indexed").textContent = cachedDocuments.filter((item) => item.status === "indexed" || item.status === "analyzed").length;
        document.getElementById("kpi-analyzed").textContent = cachedDocuments.filter((item) => item.status === "analyzed").length;
      }

      function statusText(status) {
        const map = { uploaded: "已上传", indexed: "已入库", analyzed: "已分析", failed: "失败", parsing: "解析中" };
        return map[status] || status;
      }

      async function analyzeDocument(documentId) {
        showView("kb");
        analysis.innerHTML = '<div class="empty-state">正在生成中文分析...</div>';
        const response = await fetch(`/api/documents/${documentId}/analyze`, { method: "POST" });
        if (!response.ok) {
          const errorPayload = await response.json().catch(() => ({}));
          analysis.innerHTML = `<div class="empty-state">${escapeHtml(errorPayload.detail || "分析失败，请检查文档是否成功解析。")}</div>`;
          await loadDocuments();
          return;
        }
        const result = await response.json();
        analysis.innerHTML = `
          ${analysisItem("中文摘要", result.summary_zh)}
          ${analysisItem("创新点", result.innovations)}
          ${analysisItem("方法分析", result.methodology)}
          ${analysisItem("实验结论", result.experiments)}
          ${analysisItem("局限性", result.limitations)}
          ${analysisItem("图表解读", result.chart_insights)}
        `;
        await loadDocuments();
      }

      function analysisItem(title, value) {
        return `<article class="item"><h3>${title}</h3><pre>${escapeHtml(value || "")}</pre></article>`;
      }

      form.addEventListener("submit", async (event) => {
        event.preventDefault();
        if (!input.files.length) {
          error.textContent = "请先选择 PDF 或 PPTX 文件。";
          return;
        }
        button.disabled = true;
        button.textContent = "上传中...";
        const data = new FormData();
        data.append("file", input.files[0]);
        try {
          const response = await fetch("/api/documents/upload", { method: "POST", body: data });
          if (!response.ok) throw new Error(await response.text());
          input.value = "";
          await loadDocuments();
        } catch (uploadError) {
          error.textContent = uploadError.message || "上传失败。";
        } finally {
          button.disabled = false;
          button.textContent = "上传并解析";
        }
      });

      function resetChatMessages() {
        chatMessages.innerHTML = '<p class="chat-empty">可以像聊天一样连续追问论文内容，例如：这篇论文的创新点是什么？方法和已有方法相比有什么区别？实验结论可靠吗？</p>';
      }

      function appendChatMessage(role, label, bodyHtml, options = {}) {
        chatMessages.querySelector(".chat-empty")?.remove();
        const message = document.createElement("article");
        message.className = `message ${role}${options.pending ? " pending" : ""}`;
        message.innerHTML = `
          <div class="message-role">${escapeHtml(label)}</div>
          <div class="bubble">${bodyHtml}</div>
        `;
        chatMessages.appendChild(message);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        return message;
      }

      function renderAgentAnswer(result) {
        return escapeHtml(result.answer);
      }

      chatForm.addEventListener("submit", async (event) => {
        event.preventDefault();
        const question = questionInput.value.trim();
        if (!question) return;
        appendChatMessage("user", "你", escapeHtml(question));
        questionInput.value = "";
        askButton.disabled = true;
        askButton.textContent = "思考中...";
        answer.innerHTML = "";
        const pendingMessage = appendChatMessage("agent", "Agent", "正在检索本地知识库并组织回答...", { pending: true });
        try {
          const response = await fetch("/api/chat/ask", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question })
          });
          if (!response.ok) {
            const message = await readErrorMessage(response);
            throw new Error(message || "问答失败，请查看后端日志。");
          }
          const result = await response.json();
          currentSessionId = result.session_id;
          const agentAnswer = renderAgentAnswer(result);
          pendingMessage.remove();
          appendChatMessage("agent", "Agent", agentAnswer);
          answer.innerHTML = `<div class="answer">${agentAnswer}</div>`;
        } catch (chatError) {
          pendingMessage.remove();
          const message = escapeHtml(chatError.message || "问答失败。");
          appendChatMessage("agent", "Agent", `<span class="error">${message}</span>`);
          answer.innerHTML = `<p class="error">${message}</p>`;
        } finally {
          askButton.disabled = false;
          askButton.textContent = "发送";
          questionInput.focus();
        }
      });

      questionInput.addEventListener("keydown", (event) => {
        if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
          event.preventDefault();
          chatForm.requestSubmit();
        }
      });

      arxivForm.addEventListener("submit", async (event) => {
        event.preventDefault();
        const query = arxivInput.value.trim();
        if (!query) return;
        arxivButton.disabled = true;
        arxivButton.textContent = "检索中...";
        arxivResults.innerHTML = "";
        try {
          const response = await fetch(`/api/arxiv/search?query=${encodeURIComponent(query)}&max_results=5`);
          const papers = await response.json();
          arxivResults.innerHTML = papers.map((paper) => `
            <article class="paper-result">
              <h3>${escapeHtml(paper.title)}</h3>
              <div class="citation">${escapeHtml(paper.authors.join(", "))} · ${paper.published}</div>
              <p>${escapeHtml(paper.summary.slice(0, 700))}</p>
              <a href="${paper.entry_url}" target="_blank">arXiv</a>
              ${paper.pdf_url ? ` · <a href="${paper.pdf_url}" target="_blank">PDF</a>` : ""}
            </article>
          `).join("") || '<div class="empty-state">没有找到相关论文。</div>';
        } catch (searchError) {
          arxivResults.innerHTML = `<p class="error">${escapeHtml(searchError.message || "检索失败。")}</p>`;
        } finally {
          arxivButton.disabled = false;
          arxivButton.textContent = "检索论文";
        }
      });

      function escapeHtml(value) {
        return String(value)
          .replaceAll("&", "&amp;")
          .replaceAll("<", "&lt;")
          .replaceAll(">", "&gt;")
          .replaceAll('"', "&quot;")
          .replaceAll("'", "&#039;");
      }

      async function readErrorMessage(response) {
        const text = await response.text();
        try {
          const payload = JSON.parse(text);
          return payload.detail || payload.message || text;
        } catch {
          return text;
        }
      }

      refreshAll();
    </script>
  </body>
</html>
"""
