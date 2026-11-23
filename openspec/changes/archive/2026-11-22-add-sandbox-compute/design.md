# Design

## Context

接入基线为 GitHub `main` 的 v2.2 合并提交 `e1d4c6b`。发布版从 `main.py` 启动，默认聊天页由 `backend/app/web.py` 提供，走现有 SSE 客户端。React/Vite 应用当前提供独立的知识库页面。

## Goals / Non-Goals

把已验证的 Runtime 接到发布版的显式 Compute 入口，沿用既有聊天与会话数据模型。应用侧接入范围限定在聊天请求和响应、Compute 服务、MCP 适配、会话关联产物及默认聊天页。

## Decisions

1. **复用独立 Runtime。** 迁移 Phase 6 的 Python 包、MCP 服务、Docker 镜像源码和产物管理，保留原有隔离及生命周期行为。把 Docker 逻辑移入后端可以少运行一个进程，但会耦合两个依赖环境，也需要重新验证已验收的执行边界。
2. **在聊天入口分派 Compute。** 两个 API 在进入 v2.2 Research 工作流前依据显式 `mode` 分派。沿用原 Compute 的结构化生成、可选论文取证和 MCP 调用。采用路由器自动判断会改变既有问题的执行语义，也需要重新评估路由行为。
3. **沿用流式协议与会话记录。** Research 沿用既有消息格式。Compute 按旧有约定在普通响应及 SSE 最终事件中附上执行结果，在默认聊天页按现有 SSE 消费逻辑展示。用既有 `TranscriptEvent` 保存 `sandbox_execution` 记录，沿用最近 20 条、最长 7 天的上下文限制。产物下载按会话记录和产物 ID 联合校验。这种做法依赖现有会话表，省去数据库迁移。
4. **按需安装。** 后端仅在启用时使用可选 MCP 客户端，Sandbox Runtime 使用单独的 Python 环境。标准 `setup.ps1` 继续按原流程工作，README 增加可复制的 Sandbox 安装、镜像构建和启用步骤。若把 Docker 纳入标准安装，配置环节更集中，但会增加所有 Research 用户的环境要求和资源开销。

## Risks / Trade-offs

- 发布目录审计当前对 GitHub `main` 通过。新增 Runtime 与 OpenSpec 目录需要为可分发源码和正式 Spec 设置窄范围规则。使用本地测试检出执行回归，延续当前发布审计对测试、日志、环境、数据和凭据的过滤。
- Compute 依赖已运行的 Linux Docker Engine、执行镜像和已配置的模型。先通过固定生成输入验证执行链路，再通过真实模型和默认网页验证生成及展示。两类结果分别记录。
- 容器运行在受信任的单机 Docker 环境。产物会话关联复用当前本地应用的访问模型。

## Migration Plan

现有安装无需迁移数据库。启用时安装 Runtime 和可选 MCP 依赖、构建镜像并设置 `SANDBOX_ENABLED=true`。若需停用，将其设回 `false` 后重启 Paper-Agent。
