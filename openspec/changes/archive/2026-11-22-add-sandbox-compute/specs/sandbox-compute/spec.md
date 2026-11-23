# Sandbox Compute

## Purpose

让 Paper-Agent v2 用户在聊天中主动执行论文相关的 Python 计算，查看真实结果和产物，同时保留既有 Research 工作流和标准安装方式。

## ADDED Requirements

### Requirement: 显式选择计算模式

聊天界面及普通和流式聊天 API SHALL 支持显式 Compute。省略模式时 SHALL 使用现有 Research 流程。

#### Scenario: 既有请求保持兼容
- **WHEN** 客户端省略模式或使用 Research，即使问题中包含执行代码的措辞
- **THEN** 系统使用既有 Research 工作流和响应约定，且不调用 Sandbox

#### Scenario: 显式发起计算
- **WHEN** 已启用 Sandbox，且用户选择 Compute 或 API 请求传入 `mode=compute`
- **THEN** 系统走独立的计算路径，按需读取用户选择的论文证据，每个请求最多执行一次

### Requirement: 可选安装与明确失败

Sandbox SHALL 默认关闭，标准安装 SHALL 在没有 Docker 和 MCP 的环境中继续支持 Research。

#### Scenario: 组件关闭或不可用
- **WHEN** Compute 被关闭，或启用后 MCP、Docker 或执行镜像不可用
- **THEN** Compute 以可区分的状态说明未启用或不可用，且不在宿主机执行生成代码

### Requirement: 计算结果与网页展示

计算响应 SHALL 返回实际执行状态、输出和产物元数据。网页 SHALL 展示实际结果，并为返回的产物提供会话内访问入口。

#### Scenario: 真实执行成功
- **WHEN** 普通 API 或流式 API 提交均值与绘图任务，且 Runtime 成功执行
- **THEN** 响应包含预期数值及实际生成的 PNG，聊天页的流式调用可以展示结果并访问 PNG

#### Scenario: 执行失败
- **WHEN** 用户代码失败或执行超时
- **THEN** 响应保留实际失败类别，不显示伪造的成功输出

### Requirement: 隔离与容器回收

计算 SHALL 运行在临时 Linux 容器内，维持 Phase 6 的非 root、禁网络、只读根文件系统、资源上限、禁特权和无宿主挂载等隔离条件。执行结果和清理结果 SHALL 可独立判定。

#### Scenario: 临时容器完成执行
- **WHEN** 任务成功、失败、超时或被 Runtime 取消
- **THEN** Runtime 尝试回收该容器，报告清理结果并保留原执行状态

#### Scenario: Runtime 异常停止后恢复
- **WHEN** Runtime 重启并发现归属已失效的残留容器
- **THEN** Runtime 回收其负责的残留容器，并保留其他实例正在使用的容器

### Requirement: 会话内复用和产物访问

系统 SHALL 在现有会话记录中保留限量、有限期的计算上下文，只允许当前会话复用输入及访问关联产物。计算代码、输出和产物 SHALL 保持在计算上下文内。

#### Scenario: 同会话复用结构化输入
- **WHEN** 用户在同一会话中要求用上一次成功计算的数据画不同的图
- **THEN** 新任务复用该数据，在新容器执行并返回新的执行 ID，计算记录不进入 Research 上下文或 Memory V2

#### Scenario: 验证产物会话归属
- **WHEN** 客户端请求某个产物 ID
- **THEN** 服务只向与该产物关联的会话提供文件，其他会话的请求失败
