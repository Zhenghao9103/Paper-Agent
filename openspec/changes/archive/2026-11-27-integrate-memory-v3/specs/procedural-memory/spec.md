# 程序记忆的 3.0 配置

## ADDED Requirements

### Requirement: 默认启用与升级兼容

系统 SHALL 在无显式配置时采用 RSI 模式，并优先采用部署者已有的有效模式配置。

#### Scenario: 新安装
- **WHEN** 以 3.0 默认配置完成安装
- **THEN** 程序记忆模式为 rsi，空库的首轮普通问答能够完成。

#### Scenario: 从 2.2 升级
- **WHEN** 带有已保存语义记忆的 2.2 数据库迁移到 3.0
- **THEN** 原有记忆仍可按原有规则召回；如显式设置了 off、write、flat 或 rsi，则使用该值。
