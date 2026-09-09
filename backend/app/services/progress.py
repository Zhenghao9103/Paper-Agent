from collections.abc import Callable
from typing import NotRequired, TypedDict


class ProgressEvent(TypedDict):
    code: str
    message: str
    round: NotRequired[int]


ProgressCallback = Callable[[ProgressEvent], None]

_MESSAGES = {
    "context": "正在读取会话上下文",
    "routing": "正在判断问题类型",
    "planning": "正在规划检索",
    "retrieval": "正在检索本地知识库",
    "agent_planning": "正在分解研究任务",
    "agent_retrieval": "正在补充证据",
    "evidence": "正在筛选和核验证据",
    "generation": "正在生成回答",
    "degraded": "部分检索通道不可用，正在使用可用证据继续",
}


def emit_progress(
    callback: ProgressCallback | None,
    code: str,
    *,
    round_number: int | None = None,
) -> None:
    if code not in _MESSAGES:
        raise ValueError(f"Unsupported progress code: {code}")
    if callback is None:
        return
    event: ProgressEvent = {"code": code, "message": _MESSAGES[code]}
    if round_number is not None:
        event["round"] = max(1, int(round_number))
    try:
        callback(event)
    except Exception:
        return
