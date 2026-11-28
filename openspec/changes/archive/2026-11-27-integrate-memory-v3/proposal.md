# Paper-Agent 3.0 记忆升级

## Why

V2 能在后续会话复用用户偏好与研究上下文。3.0 进一步利用具体交互中的经验，让助手既能记住用户关心什么，也能学会如何完成同类任务。

## What Changes

- 有选择地保留形成或修订语义记忆、程序记忆所需的情景来源。
- 整合本地分支已有的反馈提炼、Skill 写入、RSI 检索和普通问答注入。
- 将项目版本统一为 3.0.0。本提案建议无显式配置时启用 RSI，尊重已有配置。

## Capabilities

### New Capabilities

- `episodic-grounding`：选取交互经历，为从中形成的语义记忆和程序记忆提供可追溯来源。

### Modified Capabilities

- `procedural-memory`：定义 3.0 的默认模式和升级兼容行为。

## Impact

涉及聊天与 checkpoint 写入、Memory Harness 的来源关联、版本配置、前端包元数据和 README。复用同一应用数据库及现有迁移。
