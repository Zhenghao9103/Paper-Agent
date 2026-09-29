import json

from backend.app.services.router_contract import ROUTER_SYSTEM_PROMPT
from .contract import RouterCase

# The M0/phase04 training artifacts import the prompt from this module; the
# frozen production contract is the single source of truth so the served and
# trained prompts can never drift.
ROUTER_M0_SYSTEM_PROMPT = ROUTER_SYSTEM_PROMPT


def build_router_messages(case: RouterCase) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": ROUTER_M0_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(case.input.model_dump(), ensure_ascii=False),
        },
    ]
