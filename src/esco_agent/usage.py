"""Token use per turn: every LLM call adds its usage to the counter of the turn
that is running (a context variable, so concurrent sessions don't mix).

    counter = usage.start()       # before the turn
    reply = usage.count(await llm.ainvoke(...))
    ... counter holds calls, input/output tokens, input tokens served from cache
"""

from __future__ import annotations

from contextvars import ContextVar

_current: ContextVar[dict | None] = ContextVar("esco_usage", default=None)
KEYS = ("calls", "input_tokens", "output_tokens", "cached_input_tokens")


def start() -> dict:
    counter = dict.fromkeys(KEYS, 0)
    _current.set(counter)
    return counter


def count(reply):
    """Add one LLM reply's usage to the running counter; returns the reply."""
    counter = _current.get()
    if counter is not None:
        u = getattr(reply, "usage_metadata", None) or {}
        counter["calls"] += 1
        counter["input_tokens"] += u.get("input_tokens", 0) or 0
        counter["output_tokens"] += u.get("output_tokens", 0) or 0
        counter["cached_input_tokens"] += (u.get("input_token_details") or {}).get("cache_read", 0) or 0
    return reply


def add(total: dict | None, turn: dict) -> dict:
    total = dict(total or dict.fromkeys(KEYS, 0))
    for k in KEYS:
        total[k] = total.get(k, 0) + turn.get(k, 0)
    return total
