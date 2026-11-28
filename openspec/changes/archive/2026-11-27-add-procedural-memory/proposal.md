# Paper-Agent Memory V3

## Why

普通问答在 research.py 中调用 answer_general(question)，当前接口只接收问题。用户即使明确纠正过讲解方式，下一次会话也缺少把这条经验交给生成模型的路径。

## What Changes

从有反馈的历史事件中提炼可追溯的 Skill，以 ACE 风格增量维护，经 RSI 在线聚类和两阶段检索后送入普通问答。对照实验检验收益和成本。

## Capabilities

### New Capabilities

- `procedural-memory` 管理交互策略从提取到跨会话使用的生命周期。

### Modified Capabilities

无。

## Impact

涉及后端记忆模型、Alembic 迁移、写入恢复、程序性检索、普通问答生成及离线评测。范围外包括论文 RAG 算法、跨会话历史事件检索、Master Skill、晋升、拆分和固定库容量。
