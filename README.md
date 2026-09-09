# Paper-Agent v2

Paper-Agent 是一个面向学术 PDF 的本地优先 RAG / Agent 应用。上传 PDF 后，系统会自动完成 MinerU 解析、SQLite 持久化、BM25 建索引、BGE-M3 向量化和 Chroma 入库；问答阶段由随仓库发布的本地 Router 模型选择 `direct`、`simple_rag` 或 `agentic_rag`。

本发布版只包含运行代码和最终 Router 模型，不包含 `docs/`、`data/`、`reports/`、评估脚本、训练材料或预生成的 Chroma 数据库。

## 系统要求

- Windows x64
- Python 3.11（必须是 3.11）
- Git 与 Git LFS
- 至少 20 GB 可用空间
- 首次完整安装需要网络连接

## 完整安装

先安装并初始化 Git LFS，然后克隆仓库：

```powershell
git lfs install
git clone https://github.com/Zhenghao9103/Paper-Agent.git
cd Paper-Agent
.\setup.ps1
```

`setup.ps1` 默认执行完整安装：创建主环境和独立图表理解环境、安装依赖、拉取最终 Q4 Router、下载并校验 llama.cpp、BGE-M3、BGE-Reranker、MinerU 主模型、MinerU 图表模型及 tiktoken 缓存，并生成仅引用当前克隆目录的本地配置。

`.mineru/`、`.tools/`、`.hf-cache/` 和 `.cache/` 都由安装脚本生成，用户不需要手动下载或提交这些目录。所有外部资源的版本和校验信息记录在 `resources.lock.json`。

如只想查看代码而暂时不下载约 1.1 GB 的 Router 模型：

```powershell
$env:GIT_LFS_SKIP_SMUDGE="1"
git clone https://github.com/Zhenghao9103/Paper-Agent.git
```

之后运行 `git lfs pull` 再执行 `setup.ps1`。

## 配置模型服务

安装脚本会生成根目录 `.env`，并保留已存在的 API Key。至少配置用于回答和 Agentic RAG 的模型：

```dotenv
AGENT_API_KEY=your_api_key
AGENT_BASE_URL=https://your-provider.example/v1
AGENT_MODEL=your_model
```

本地 Router 无需远程 API Key。`JUDGE_*` 仅用于开发期质量评估，发布版正常运行不需要配置。

## 启动与使用

```powershell
.\.venv\Scripts\python.exe main.py
```

浏览器默认打开 `http://127.0.0.1:8000`。如果不希望自动打开：

```powershell
$env:PAPER_AGENT_OPEN_BROWSER="0"
.\.venv\Scripts\python.exe main.py
```

上传 PDF 后会自动生成当前文档的 SQLite、BM25 和 Chroma 数据；仓库不提供也不需要预置 Chroma。文档状态只有在向量数量与内容校验完成后才会变为 `indexed`。

## 主要能力

- MinerU PDF 结构化解析、公式识别与图表理解
- SQLite 文档存储与 BM25 全文检索
- BGE-M3 向量召回、ChromaDB 本地索引和 BGE-Reranker 精排
- 本地 GGUF Router 与 llama.cpp 托管
- `direct`、`simple_rag`、`agentic_rag` 三类问答路径
- 引用、会话记忆、上下文压缩和安全轨迹记录
- 内置 Web 界面及 React/Vite 前端源码

## 目录说明

```text
backend/             FastAPI 后端、解析、检索与 Agent 代码
frontend/            React/Vite 前端源码与已构建静态资源
models/router/       唯一随仓库发布的 Q4 Router（Git LFS）
scripts/             安装、安装后检查与发布审计脚本
resources.lock.json  外部资源版本锁
setup.ps1            Windows 完整安装入口
```

以下均为运行时生成并被 Git 忽略的目录：`.mineru/`、`.tools/`、`.hf-cache/`、`.cache/`、`chroma/`、`storage/`、`backend/data/`。

## 常见问题

- `Git LFS resource mismatch`：确认已安装 Git LFS，然后运行 `git lfs pull`。
- 可用空间不足：清理磁盘后重新运行 `setup.ps1`；已校验完成的资源会被复用。
- 外部资源下载中断：直接重跑 `setup.ps1`，临时文件不会替换已完成资源。
- API 无法生成回答：检查 `.env` 中 `AGENT_API_KEY`、`AGENT_BASE_URL` 与 `AGENT_MODEL`。

## 开发测试

```powershell
.\.venv\Scripts\python.exe -m pip install -r backend\requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest backend\tests -q
```

## License

[MIT](LICENSE)
