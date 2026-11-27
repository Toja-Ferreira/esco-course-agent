"""DeepSeek clients and calls with time limits.

Every client is built here (deepseek()) and every call goes through call():
  - a call whose reply arrives in one piece (queries, judge, drafting, the
    baseline) has a limit on the whole call: LLM_CALL_TIMEOUT_S (default 60 s);
  - the conversation reply, streamed to the page, has a limit on the gap between
    chunks instead (LLM_STREAM_GAP_S, default 30 s; the wait for the first chunk
    counts as a gap): the client's stream_chunk_timeout, which measures silence in
    content (DeepSeek's keep-alive lines do not reset it, unlike the HTTP read
    timeout, which is set to the same value as a second guard);
  - a timed-out call is retried once; if it times out again, LLMTimeout ends the
    turn (the web app tells the teacher and logs it). A streamed reply is retried
    only if none of it has reached the page yet; otherwise the turn ends at once
    and the text shown so far is kept, marked as interrupted (LLMTimeout.partial);
  - other transient errors (connection, rate limit, server errors) are retried
    twice with a short pause, as the client's own default did (it is turned off:
    its retries were silent, and with its 600 s default timeout one stalled call
    took ten minutes; 27 Nov laptop timing).
Defaults, from the call times on record (27 Nov): the judge phase, several calls
in parallel, took at most 5.0 s in 126 replayed first turns (median 2.5 s); whole
chat turns in the session logs at most 24 s (median 2.5 s). 60 s and 30 s are
well above both, so a slow but working call is not cut off.

Timeouts of the running turn are collected like token use (a context variable):
    log = llm_calls.start()   # before the turn: [{"t", "call", "attempt", "outcome"}]
"""

from __future__ import annotations

import asyncio
import os
from contextvars import ContextVar
from datetime import datetime, timezone

from esco_agent import usage

_current: ContextVar[list | None] = ContextVar("esco_llm_timeouts", default=None)
_shown: ContextVar[list | None] = ContextVar("esco_llm_shown", default=None)   # text sent to the page
MESSAGE = {"pt": "⚠ O modelo não respondeu a tempo. Tente novamente dentro de momentos.",
           "en": "⚠ The model did not answer in time. Please try again in a moment."}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name) or default)
    except ValueError:
        return default


def call_timeout_s() -> float:
    return _env_float("LLM_CALL_TIMEOUT_S", 60.0)


def stream_gap_s() -> float:
    return _env_float("LLM_STREAM_GAP_S", 30.0)


class LLMTimeout(Exception):
    """partial: the text of a streamed reply that had already reached the page
    when it stalled ("" if none)."""
    def __init__(self, call: str, partial: str = ""):
        super().__init__(f"the model did not answer in time ({call})")
        self.call, self.partial = call, partial


def start() -> list[dict]:
    log: list[dict] = []
    _current.set(log)
    _shown.set([])
    return log


def shown(text: str) -> None:
    """The web app calls this for each piece of a streamed reply it sends to the page."""
    sent = _shown.get()
    if sent is not None:
        sent.append(text)


def _note(call: str, attempt: int, outcome: str) -> None:
    log = _current.get()
    if log is not None:
        log.append({"t": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "call": call, "attempt": attempt, "outcome": outcome})


def deepseek(json_output: bool = False, streamed: bool = False, model: str | None = None,
             **extra):
    """A DeepSeek chat model with the time limits above. streamed: the reply is
    streamed (the conversation), so the limit is the gap between chunks. (In the
    web app the JSON calls stream too, under LangGraph; their gap limit is then the
    call limit, and call() limits the whole call.)"""
    import httpx
    from langchain_deepseek import ChatDeepSeek

    gap = stream_gap_s() if streamed else call_timeout_s()
    llm = ChatDeepSeek(model=model or os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
                       temperature=0, max_retries=0, stream_chunk_timeout=gap,
                       timeout=httpx.Timeout(connect=10.0, read=gap, write=30.0, pool=10.0),
                       **extra)
    return llm.bind(response_format={"type": "json_object"}) if json_output else llm


def _is_timeout(exc: BaseException) -> bool:
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return True
    try:
        import httpx
        import openai
        return isinstance(exc, (openai.APITimeoutError, httpx.TimeoutException))
    except ImportError:   # pragma: no cover
        return False


def _is_transient(exc: BaseException) -> bool:
    try:
        import openai
        return isinstance(exc, (openai.APIConnectionError, openai.RateLimitError,
                                openai.InternalServerError))
    except ImportError:   # pragma: no cover
        return False


async def call(llm, messages, name: str, streamed: bool = False):
    """llm.ainvoke(messages) with the time limits; counts token use (usage.count).
    streamed: the gap limit is the client's (deepseek(streamed=True)); otherwise the
    whole call is limited here too."""
    timeouts = errors = 0
    while True:
        sent = _shown.get()
        before = len(sent) if sent is not None else 0
        try:
            coro = llm.ainvoke(messages)
            reply = await (coro if streamed else asyncio.wait_for(coro, call_timeout_s()))
            return usage.count(reply)
        except Exception as exc:
            if _is_timeout(exc):
                timeouts += 1
                partial = "".join(sent[before:]) if streamed and sent is not None else ""
                if partial:      # the teacher has seen part of it: no retry, keep what was shown
                    _note(name, timeouts, "interrupted")
                    raise LLMTimeout(name, partial) from exc
                if timeouts > 1:
                    _note(name, timeouts, "gave up")
                    raise LLMTimeout(name) from exc
                _note(name, timeouts, "retried")
            elif _is_transient(exc) and errors < 2:
                errors += 1
                await asyncio.sleep(errors)
            else:
                raise


def summary(log: list[dict]) -> dict:
    """The session file's "llm_timeouts"."""
    return {"count": len(log),
            "gave_up": sum(e["outcome"] in ("gave up", "interrupted") for e in log),
            "interrupted": sum(e["outcome"] == "interrupted" for e in log), "calls": log}
