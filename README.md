# Paper-Agent v2.2

> **可选本地模型或 Jev 路由的 Evidence-driven Agentic RAG**

Paper-Agent 是一个面向学术 PDF 的本地优先研究助手。它把 PDF 结构化解析、混合检索、证据管理和答案校验组织成一条可控的研究链路，并使用本地小模型或 Jev 判断每个问题应该直接处理、执行一次 RAG，还是进入多轮 Agentic RAG。

| 核心亮点 | 作用 |
| --- | --- |
| **可选意图 Router** | 默认使用随仓库发布的 Qwen3-1.7B Q4 本地模型，也可选择 Jev API |
| **三路按需执行** | 在 `direct`、`simple_rag`、`agentic_rag` 之间选择，避免所有问题都启动复杂 Agent |
| **Evidence-driven Agent Loop** | 围绕 claim 拆解、证据准入、覆盖状态和答案校验执行有边界的多轮研究 |
| **Memory V2** | 候选提取、建议式 Judge、确定性 Harness 与可恢复的双存储同步 |
| **本地知识库** | MinerU、SQLite、BM25、BGE-M3、ChromaDB 与 BGE-Reranker 组成完整 PDF 入库和检索链路 |

![Paper-Agent 学术论文智能体整体框架](frontend/public/readme/paper-agent-architecture.png)

## 意图路由（本地小模型/jev）

Router 根据当前问题和有限会话上下文输出一个意图，不负责生成答案或规划检索：

- `direct`：文档状态等无需检索的问题直接在本地处理。
- `simple_rag`：执行一次查询规划、混合检索、重排和基于证据的回答。
- `agentic_rag`：复杂比较、多对象分析和多跳问题进入 Agent 研究循环。

本地模式使用 `Qwen3-1.7B-Router-SFT-V3-Q4_K_M.gguf`，约 1.1 GB，通过 Git LFS 管理；启动器会用 llama.cpp 在 `127.0.0.1:8089` 托管它，无需 Router API Key。Jev 模式通过配置的 API 完成同样的三分类，且不启动本地 Router。两种模式下，Router 超时、协议错误或输出无效时都会保守地降级到 `agentic_rag`。

## Evidence-driven Agentic RAG

传统 RAG 通常在一次检索后直接生成答案。Paper-Agent 的 `agentic_rag` 会先明确需要回答的 claims，再围绕尚未覆盖的证据缺口迭代检索；只有整理出受控的 Evidence Pack 后才允许生成最终答案。

![Evidence-driven Agentic RAG 工作流](frontend/public/readme/agentic-rag-workflow.png)

这条链路并不是把全部控制权交给 LLM：模型负责 claim 拆解、语义判断和答案生成，代码负责工具白名单、Schema 校验、Evidence Pool 状态更新、token 预算、引用合法性、超时与失败降级。默认最多执行 3 轮研究和 5 次工具调用，并受 180 秒总时限约束。

本发布版只包含运行代码和最终 Router 模型，不包含 `docs/`、`data/`、`reports/`、评估脚本、训练材料或预生成的 Chroma 数据库。

## Memory V2：长期记忆

![Paper-Agent Memory V2 整体架构](frontend/public/readme/memory-v2-architecture.png)

Memory 保存用户偏好、研究兴趣、跨会话研究上下文和尚未完成的事项，对应 `user_preference`、`research_interest`、`research_context`、`open_loop` 四种类型。当前会话的 SessionMemory 与近期消息提供短期上下文，长期记忆则跨会话复用；论文事实及引用仍来自检索证据。

### Write Memory

Checkpoint 提交后，Memory Extractor 对比 old/new SessionMemory 的四字段快照：目标与约束、当前决定、待解问题、下一步行动，输出 `memory_type/content/confidence` 候选。图中“本轮新增对话”用于生成会话状态，不会直接传给长期提取接口；`confirmed_findings`、Evidence、工具输出及完整回答也不进入该接口。

Memory Harness 先校验、规范化去重，再筛选同类型相关旧记忆。规则无法判断关系时，由同一个 Extractor 兼任 Judge，基于 Harness 提供的受限快照给出建议；数据库查询、目标复核及状态修改始终由 Harness 执行。相似度只用于筛选相关项，不直接证明重复或冲突。

| 操作 | 处理方式 |
| --- | --- |
| ADD | 独立事项创建新记录 |
| UPDATE | 同一事项被细化，保留 ID，使用候选内容 |
| MERGE | 信息互补，保留目标 ID，使用经校验的合并稿 |
| SUPERSEDE | 明确替代旧状态，旧记录 superseded，新记录 active |
| IGNORE | 重复、低 confidence、不确定、无效建议或过期目标，不修改记忆 |

“记住、以后、我的偏好是”等明确指令走同一个 Extractor/Harness，在本轮研究前尝试同步，让成功写入的偏好立即生效。普通有引用回答不再自动写入长期记忆；提取失败不撤销成功的 checkpoint。

### Read Memory

读取保持 `问题 embedding → Chroma 候选 → SQLite active 校验 → 排序 → Top-K → 上下文注入`。默认召回 20 条候选，过滤低于 0.20 的语义相似度，最终取 Top-3；兼容历史 `research_topic`，排除 pending/superseded 记录。

`score = 0.8 × semantic + 0.2 × recency`，其中 `recency = 2 ^ (-age_days / 90)`，优先使用 SQLite 的 `updated_at`，缺失时回退到 `created_at`。Importance 已删除；confidence 只用于写入门槛，pin 仅作标记，均不参与排序。图中 metadata 时间戳为概念展示，实际新近度以 SQLite 时间戳为准。

Memory 只作为 context，不能生成 Evidence ID 或 citation，也不作为论文事实依据。回答中的方法、实验数值和论文结论仍须由可验证的检索证据支撑。

### 持久化同步与升级

SQLite 是记忆真值来源，Chroma 用于向量检索。Harness 先在 SQLite 事务中登记持久化待同步操作（Durable Outbox），将受影响记录设为 `pending_sync` 并提交，再执行 Chroma 同步；所有向量操作成功后，才在 SQLite 事务中更新内容与最终状态。UPDATE/MERGE 重新 embedding，SUPERSEDE 同步旧、新记录。

失败或中断保留 pending，允许暂时不可召回；启动和后续写入时在单后端进程内串行、有限、幂等重放。IGNORE、重试和单纯状态切换不刷新记忆时间戳，只有实际内容更新才刷新 `updated_at`。

默认 confidence 门槛 0.8，相关旧记忆阈值 0.75、最多 5 条，每轮同步最多 20 个操作；阈值和评分权重集中配置于 Settings，示例见 `backend/.env.example`。

升级已有安装时，先停止后端，备份当前 SQLite 数据库和 Chroma 目录，再在项目根目录执行：

```powershell
.\.venv\Scripts\python.exe -m alembic -c backend/alembic.ini upgrade head
```

迁移删除旧 Importance 列、创建待同步操作表，并为历史记忆登记 REINDEX，以清除旧向量 metadata；保留内容、类型、pin、最终状态和原时间戳。历史记录在自身重建完成前暂时不可召回。`create_all` 只负责新库建表，不能替代旧库迁移。已有 `.env` 如显式设置了 `APP_VERSION`，请同步改为 `2.2.0`。

## 系统要求

- Windows x64
- Python 3.11（必须是 3.11）
- Git；选择本地 Router 时还需 Git LFS
- 至少 20 GB 可用空间
- 首次完整安装需要网络连接

## 完整安装

克隆时先跳过 Git LFS 自动下载，再选择 Router：

```powershell
$env:GIT_LFS_SKIP_SMUDGE="1"
git clone https://github.com/Zhenghao9103/Paper-Agent.git
Remove-Item Env:GIT_LFS_SKIP_SMUDGE
cd Paper-Agent
```

在项目根目录创建或编辑 `.env`，用 `ROUTER_PROVIDER` 选择 Router：

```dotenv
ROUTER_PROVIDER=local
```

可填 `local` 或 `jev`；没有 `.env` 或没有这一项时默认使用 `local`。安装时运行 `.\setup.ps1`，脚本会读取该配置。也可临时用 `.\setup.ps1 -RouterProvider jev` 覆盖 `.env`，并将所选模式写回 `.env`。

本地 Router 会下载约 1.1 GB 的 Q4 模型及 llama.cpp：

```powershell
git lfs install
.\setup.ps1
```

`setup.ps1` 会创建主环境和独立图表理解环境、安装依赖、下载并校验 BGE-M3、BGE-Reranker、MinerU 主模型、MinerU 图表模型及 tiktoken 缓存，并生成仅引用当前克隆目录的本地配置。选择本地 Router 时还会拉取最终 Q4 Router 和 llama.cpp。

`.mineru/`、`.hf-cache/` 和 `.cache/` 都由安装脚本生成；本地 Router 还会生成 `.tools/`。用户不需要手动下载或提交这些目录。所有外部资源的版本和校验信息记录在 `resources.lock.json`。

如选择 Jev，先将根目录 `.env` 中的配置改为：

```dotenv
ROUTER_PROVIDER=jev
JEV_API_KEY=your_api_key
JEV_BASE_URL=
JEV_MODEL=jev-1.13.0
```

然后运行安装脚本，无需下载本地 Q4 Router 和 llama.cpp：

```powershell
.\setup.ps1
```

Jev 模式仅省去 Router 模型与 llama.cpp，PDF 解析和检索所需模型仍照常安装。安装后改动 `.env` 的 `ROUTER_PROVIDER`，重启程序即可切换；若最初按 Jev 模式安装，切换回本地 Router 时还需运行 `.\setup.ps1` 下载本地 Router 资源。

## 配置模型服务

安装脚本会生成根目录 `.env`，并保留已存在的 API Key。至少配置用于回答和 Agentic RAG 的模型：

```dotenv
AGENT_API_KEY=your_api_key
AGENT_BASE_URL=https://your-provider.example/v1
AGENT_MODEL=your_model
```

本地 Router 无需远程 API Key。选择 Jev 时，在根目录 `.env` 填写 `JEV_API_KEY`。使用 TypeSafe 官方密钥时可将 `JEV_BASE_URL` 留空；使用 OpenCode Zen 等渠道时，还需填写对应的 `JEV_BASE_URL` 和 `JEV_MODEL`。`JUDGE_*` 仅用于开发期质量评估，发布版正常运行不需要配置。

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
- 可选本地 Qwen3-1.7B Q4 GGUF Router 或 Jev API
- `direct`、`simple_rag`、`agentic_rag` 三类按需问答路径
- Research Planner、Evidence Judge、Evidence Pool、Evidence Pack 与 Answer Verifier
- 工具预算、超时、Schema 校验、引用验证和安全轨迹记录
- 会话记忆与上下文压缩
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

## License

[MIT](LICENSE)
