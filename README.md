# PaperMind Agent

PaperMind Agent 是一个面向论文与学术资料的本地优先 RAG / Agent 项目。它可以上传 PDF 或 PPTX，解析正文并切分为知识片段，写入本地 SQLite 与 Chroma 向量库，然后通过研究问答、论文分析、arXiv 检索、记忆与轨迹记录等能力，搭建一个可离线演示、可扩展的个人论文知识库。

项目当前更偏向原型与作品集阶段：后端功能较完整，根路径提供一个无需 Node.js 的内置演示页面；`frontend/` 目录中也保留了 React/Vite 前端，方便后续继续打磨。

## 核心功能

- 本地上传 `.pdf` 与 `.pptx` 文件。
- 解析文档页面文本，并保存页面、分块、文档元数据。
- 使用 ChromaDB 建立本地向量索引。
- 基于本地检索结果进行带引用的研究问答。
- 使用 LangGraph 组织问答流程，并记录 trace 事件。
- 支持上下文压缩、短期会话摘要和长期记忆向量。
- 支持论文中文分析笔记，包括摘要、创新点、方法、实验、局限和图表洞察。
- 支持 arXiv 关键词检索。
- 支持 OpenAI 兼容接口；未配置 API Key 时，会使用本地证据生成降级结果。
- 提供 API 健康检查、LLM 配置检查、文档清空、记忆和 transcript 查询等接口。

## 技术栈

- 后端：FastAPI、SQLAlchemy、Pydantic、Uvicorn
- RAG / Agent：LangGraph、LangChain、ChromaDB、sentence-transformers、FlagEmbedding
- 文档解析：PyMuPDF、python-pptx、Pillow
- 数据存储：SQLite、`storage/` 本地文件目录、`chroma/` 本地向量库
- 前端：内置 HTML 演示页；另有 React 18 + Vite + TypeScript 前端目录
- 测试：pytest、pytest-asyncio、ruff

## 目录结构

```text
paper-agent/
|-- main.py                         # 一键启动入口，运行 FastAPI 并自动打开浏览器
|-- requirements.txt                # 项目根依赖，包含运行与开发依赖
|-- backend/
|   |-- app/
|   |   |-- api/routes/             # health、documents、chat、arxiv、memory API
|   |   |-- agents/                 # LangGraph 研究问答流程
|   |   |-- core/                   # 配置、路径与运行目录初始化
|   |   |-- db/                     # SQLAlchemy 会话与数据库初始化
|   |   |-- ingestion/              # PDF / PPTX 解析
|   |   |-- rag/                    # 分块、重排、向量库
|   |   |-- services/               # 文档入库、分析、问答、记忆、LLM 等业务逻辑
|   |   |-- schemas/                # API 请求与响应模型
|   |   |-- models/                 # 数据库模型
|   |   |-- web.py                  # 根路径内置演示页面
|   |   `-- main.py                 # FastAPI 应用实例
|   |-- tests/                      # 后端测试
|   `-- .env.example                # 后端环境变量示例
|-- frontend/                       # React/Vite 前端原型
|-- docs/                           # 设计文档与实施计划
|-- storage/                        # 上传文件、页面资源和导出文件
`-- chroma/                         # ChromaDB 本地向量库
```

## 环境要求

- Python 3.11 或更高版本

## 安装依赖

在项目根目录执行：

```powershell
python -m pip install -r requirements.txt
```

如果你已经有独立的虚拟环境，也可以直接使用该环境中的 Python。例如当前项目原 README 中使用的是：

```powershell
D:\envs_dirs\paper-agent\python.exe -m pip install -r requirements.txt
```

## 一键运行本地演示

在项目根目录执行：

```powershell
python main.py
```

或使用指定解释器：

```powershell
D:\envs_dirs\paper-agent\python.exe main.py
```

默认服务地址：

```text
http://localhost:8000
```

`main.py` 会启动 `backend.app.main:app`，监听 `127.0.0.1:8000`，并自动打开浏览器。如果不希望自动打开浏览器，可以设置：

```powershell
$env:PAPER_AGENT_OPEN_BROWSER="0"
python main.py
```

## 基本使用流程

1. 打开 `http://localhost:8000`。
2. 上传一篇 PDF 或 PPTX。
3. 等待文档状态变为 `indexed`。
4. 查看文档列表、页面解析内容和分块结果。
5. 在 Research Chat 中提问，系统会检索本地知识库并返回带引用的回答。
6. 点击分析功能，生成中文论文笔记。
7. 使用 arXiv 搜索查找相关论文。
8. 查看记忆与 transcript，了解问答过程和 Agent 轨迹。

## 可选 LLM 配置

项目可以在没有 API Key 的情况下运行。未配置远程模型时，问答和分析会尽量基于本地解析内容给出降级结果。

如需启用 OpenAI 兼容模型，在项目根目录或 `backend/` 目录创建 `.env`：

```dotenv
APP_NAME=PaperMind Agent API
APP_VERSION=0.1.0
DATABASE_URL=sqlite:///./data/papermind.db
STORAGE_DIR=../storage
CHROMA_DIR=../chroma
BACKEND_CORS_ORIGINS=http://localhost:5173

OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_CHAT_MODEL=gpt-4.1-mini
```

也可以配置其他 OpenAI 兼容服务，例如：

```dotenv
OPENAI_API_KEY=your_deepseek_api_key
OPENAI_BASE_URL=https://api.deepseek.com
OPENAI_CHAT_MODEL=deepseek-v4-flash
```

## API 概览

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/api/health` | 服务健康检查 |
| `GET` | `/api/health/llm` | 查看 LLM 配置状态 |
| `GET` | `/api/health/llm?ping=true` | 测试远程模型连通性 |
| `GET` | `/api/documents` | 获取文档列表 |
| `POST` | `/api/documents/upload` | 上传 PDF / PPTX |
| `POST` | `/api/documents/clear` | 清空本地运行数据 |
| `GET` | `/api/documents/{document_id}` | 获取单个文档 |
| `GET` | `/api/documents/{document_id}/pages` | 获取文档页面解析结果 |
| `GET` | `/api/documents/{document_id}/chunks` | 获取文档分块结果 |
| `POST` | `/api/documents/{document_id}/analyze` | 生成论文分析 |
| `GET` | `/api/documents/{document_id}/analysis` | 获取已有论文分析 |
| `POST` | `/api/chat/ask` | 普通研究问答 |
| `POST` | `/api/chat/stream` | SSE 形式的问答轨迹事件 |
| `GET` | `/api/arxiv/search` | arXiv 论文搜索 |
| `GET` | `/api/memories` | 获取长期记忆记录 |
| `GET` | `/api/transcripts/session/{session_id}` | 获取指定会话 transcript |

健康检查示例：

```powershell
Invoke-RestMethod http://localhost:8000/api/health
```

arXiv 搜索示例：

```powershell
Invoke-RestMethod "http://localhost:8000/api/arxiv/search?query=multimodal%20RAG&max_results=5"
```

## React 前端开发

`frontend/` 是独立的 React/Vite 原型，目前用于后续前端打磨；日常演示可以直接使用后端根路径的内置页面。

如果需要开发 React 前端：

```powershell
cd frontend
npm install
npm run dev
```

默认地址：

```text
http://localhost:5173
```

前端会请求后端 API，因此需要同时运行后端服务，并确保 `BACKEND_CORS_ORIGINS` 包含 `http://localhost:5173`。

## 测试

运行全部后端测试：

```powershell
python -m pytest backend\tests -q
```

使用指定解释器：

```powershell
D:\envs_dirs\paper-agent\python.exe -m pytest backend\tests -q
```

也可以运行单个测试文件，例如：

```powershell
python -m pytest backend\tests\test_web.py -q
```

React 前端测试：

```powershell
cd frontend
npm test
```

## 本地数据与缓存

运行后会自动创建或使用以下目录：

- `backend/data/papermind.db`：SQLite 数据库。
- `storage/uploads/`：上传的原始 PDF / PPTX。
- `storage/pages/`：解析页面相关文件。
- `storage/exports/`：预留导出目录。
- `chroma/`：ChromaDB 向量索引。
- `.hf-cache/`：Hugging Face / sentence-transformers 缓存目录。

如果要重置演示数据，可以通过页面中的清空按钮，或调用：

```powershell
Invoke-RestMethod -Method Post http://localhost:8000/api/documents/clear
```

## 当前状态与注意事项

- 项目处于本地原型阶段，适合论文知识库、RAG、Agent trace 和学术分析流程演示。
- 当前上传解析主要面向可提取文本的 PDF / PPTX；扫描版 PDF 需要先 OCR。
- 远程 LLM 是可选项，未配置时会使用本地降级逻辑，但回答质量会受限。
- 部分中文提示词和内置演示页文本仍有待进一步统一与润色。
- `frontend/` 与后端内置页面并存，当前一键演示优先使用 `http://localhost:8000`。

## 后续可改进方向

- 修复并统一内置页面中的中文文案。
- 完善 React 前端中的问答、分析、arXiv、记忆和 transcript 视图。
- 增加 OCR 与图表理解能力。
- 增加 Markdown / PDF / DOCX 分析报告导出。
- 为检索质量、引用质量和论文分析质量加入更多评测样例。
- 支持更多文献来源、引用格式和项目级知识库管理。
