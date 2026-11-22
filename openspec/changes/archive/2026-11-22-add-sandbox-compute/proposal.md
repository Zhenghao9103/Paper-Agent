# Proposal

## Why

开发工作区的 Sandbox Runtime 已有真实 Docker 和 MCP 的验收记录。GitHub v2.2 缺少这条执行链路，用户无法从发布版聊天页调用隔离环境来计算和画图。

## What Changes

- 将 Phase 6 Runtime 作为可独立安装的可选组件接入 v2.2。
- 在发布版聊天页及 `/api/chat/ask`、`/api/chat/stream` 接入显式 Compute，保留 Research 为默认。
- 展示实际计算结果，支持同会话复用结构化输入和下载对应产物。
- 在发布版 README 中记录独立安装、启用、验证和关闭步骤。

## Capabilities

### New Capabilities

- `sandbox-compute`：显式 Python 计算、Docker 隔离、会话内输入复用和产物访问。

### Modified Capabilities

无。

## Impact

涉及独立 Runtime、后端配置及聊天 API、发布版聊天页和安装说明。标准安装继续运行 Research。Docker、MCP 和执行镜像由需要 Compute 的用户另行准备。
