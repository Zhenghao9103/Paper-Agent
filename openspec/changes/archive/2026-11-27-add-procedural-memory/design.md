# Memory V3 设计与审阅决定

## Context

核对本地 1ae327b 与 origin/main 的 backend 和 openspec，二者目录无已跟踪差异。schemas/memory.py 的四种类型服务于 V2。models/memory.py 的 MemorySyncOperation 已有 pending、attempts、last_error，Memory 本身没有 owner 字段。services/memory_harness.py 包含 apply_candidate 和 process_pending_writes。services/research.py 的普通问答分支直接调用 services/answering.py 的 answer_general(question)。

## Goals / Non-Goals

交付可从纠正中学习、跨会话使用、撤销后停止使用的用户级策略。策略可指导回答结构，不能承担事实证据或权限。语义记忆、会话摘要与历史事件继续提供原有能力。

## Decisions

1. 为 Skill 使用独立关系记录和派生索引。复用 V2 校验及待同步操作的模式；给旧 memories 增加大量可空策略字段会混淆两类检索。迁移只新增结构，旧数据无需合成 Skill。
2. 写入顺序为已提交事件 → 触发判断 → Reflector 提案 → Judge 匹配同范围现有 Skill → Harness 校验和事务 → 索引同步。来源事件含会话和事件标识。记录反馈与版本关联，重放按稳定事件键去重。后台失败允许重试。
3. 追加式索引采用带版本成员和逻辑失效。读取时按数据库当前版本复核；更新与删除后修正成员及中心，必要时从有效记录重建。平面检索更易验证，也是故障回退；RSI 减少候选比较，引入中心维护及召回损失。实验分别测量。
4. 扩展普通问答的可选程序性上下文参数。给模型的策略仅含适用条件、步骤和约束，来源和版本写入 trace。既有记忆文本继续按数据处理。预算按完整 Skill 分配；阈值、向量模型与预算随评测配置固化。
5. Paper-Agent 部署在本地，供一人使用。各会话共享本实例的 Skill 库，会话 ID 用于追溯来源。
6. 开发实验中的独立 ACE 组仅反思写入，用于测量新增调用和提取质量。平面与 RSI 组共享写入快照，使用按时间分离的学习与测试任务，冻结测试期间写入。任务达成按事先标注的可观察结果评分；命中按带标签任务的正确策略召回比例统计，延迟分别记录写入与查询。

## Risks / Trade-offs

误学一次性指令会使后续回答偏离意图。反思须区分仅本次要求和可复用纠正，无法确定时跳过。启用开关默认关闭，回退时保留数据库记录。结构验收与收益评估分开；实验如实报告结果，提升幅度尚无实测依据。
