"""Time limits on DeepSeek calls, with stalled mock clients: one retry, then the
turn ends with a message and the session file records it. Two tests run the real
DeepSeek client against a local server that streams and then stalls (no network)."""

import asyncio
import json
import time

import pytest
from langchain_core.messages import AIMessage
from test_web import begin, saved

from esco_agent import llm_calls


@pytest.fixture(autouse=True)
def short_limits(monkeypatch):
    monkeypatch.setenv("LLM_CALL_TIMEOUT_S", "0.2")
    monkeypatch.setenv("LLM_STREAM_GAP_S", "0.5")


class Stalled:
    """ainvoke never answers, for the first `stalls` calls; then answers."""
    def __init__(self, stalls=99, reply="ok"):
        self.stalls, self.reply, self.calls = stalls, reply, 0

    async def ainvoke(self, messages):
        self.calls += 1
        if self.calls <= self.stalls:
            await asyncio.sleep(3600)
        return AIMessage(self.reply)


def test_a_stalled_call_is_retried_once_then_gives_up():
    log = llm_calls.start()
    llm = Stalled()
    t0 = time.monotonic()
    with pytest.raises(llm_calls.LLMTimeout) as err:
        asyncio.run(llm_calls.call(llm, [("user", "x")], "judge"))
    assert err.value.call == "judge" and llm.calls == 2
    assert time.monotonic() - t0 < 1.5                     # two limits of 0.2 s, not an hour
    assert [(e["call"], e["attempt"], e["outcome"]) for e in log] == [("judge", 1, "retried"),
                                                                       ("judge", 2, "gave up")]
    assert llm_calls.summary(log) == {"count": 2, "gave_up": 1, "interrupted": 0, "calls": log}


def test_a_stall_then_an_answer_succeeds_on_the_retry():
    log = llm_calls.start()
    reply = asyncio.run(llm_calls.call(Stalled(stalls=1, reply="fine"), [("user", "x")], "queries"))
    assert reply.content == "fine" and [e["outcome"] for e in log] == ["retried"]


def test_a_streamed_call_has_no_limit_on_the_total():
    """Streamed: only the client's gap limit applies; a long but flowing reply is fine."""
    class Slow:
        async def ainvoke(self, messages):
            await asyncio.sleep(0.6)                    # longer than LLM_CALL_TIMEOUT_S
            return AIMessage("done")
    log = llm_calls.start()
    assert asyncio.run(llm_calls.call(Slow(), [], "conversation", streamed=True)).content == "done"
    assert log == []


def test_transient_errors_are_retried_as_the_client_did():
    import httpx
    import openai

    class Flaky:
        calls = 0

        async def ainvoke(self, messages):
            Flaky.calls += 1
            if Flaky.calls == 1:
                raise openai.APIConnectionError(request=httpx.Request("POST", "http://x"))
            return AIMessage("ok")
    assert asyncio.run(llm_calls.call(Flaky(), [], "judge")).content == "ok" and Flaky.calls == 2


# --- the real client against a local server that streams, then stalls --------------

def _chunk(text):
    return "data: " + json.dumps({"id": "c", "object": "chat.completion.chunk", "created": 0,
                                  "model": "deepseek-chat",
                                  "choices": [{"index": 0, "delta": {"role": "assistant", "content": text},
                                               "finish_reason": None}]}) + "\n\n"


async def _serve(events):
    """A one-route HTTP server: answers every request with the SSE events from
    `events()` (an async generator of strings, or "SLEEP:<s>")."""
    async def handle(reader, writer):
        while (await reader.readline()) not in (b"\r\n", b""):
            pass                                            # headers; the body is ignored
        writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                     b"Transfer-Encoding: chunked\r\n\r\n")
        try:
            async for e in events():
                data = e.encode()
                writer.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
                await writer.drain()
            writer.write(b"0\r\n\r\n")
            await writer.drain()
        except (ConnectionError, asyncio.CancelledError):
            pass
        finally:
            writer.close()
    return await asyncio.start_server(handle, "127.0.0.1", 0)


def _run_against(events, monkeypatch):
    async def go():
        srv = await _serve(events)
        port = srv.sockets[0].getsockname()[1]
        monkeypatch.setenv("DEEPSEEK_API_BASE", f"http://127.0.0.1:{port}/v1")
        monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key-not-real")
        llm = llm_calls.deepseek(streamed=True, streaming=True)
        log = llm_calls.start()
        try:
            t0 = time.monotonic()
            reply = await llm_calls.call(llm, [("user", "hi")], "conversation", streamed=True)
            return reply, log, time.monotonic() - t0
        except llm_calls.LLMTimeout as exc:
            return exc, log, time.monotonic() - t0
        finally:
            srv.close()
    return asyncio.run(go())


def test_real_client_a_long_stream_with_short_gaps_is_not_cut(monkeypatch):
    async def flowing():
        for i in range(8):                     # 8 chunks, 0.2 s apart: 1.6 s in all, gap 0.5 s
            yield _chunk(f"w{i} ")
            await asyncio.sleep(0.2)
        yield "data: [DONE]\n\n"
    reply, log, took = _run_against(flowing, monkeypatch)
    assert isinstance(reply, AIMessage) and reply.content.startswith("w0 w1") and log == []
    assert took > 1.4                          # longer than the gap limit, still fine


def test_real_client_a_stream_that_stalls_between_chunks_times_out(monkeypatch):
    async def stalls():
        yield _chunk("Hello")
        for _ in range(100):                   # then only keep-alive lines: no content
            yield ": keep-alive\n\n"
            await asyncio.sleep(0.1)
    exc, log, took = _run_against(stalls, monkeypatch)
    assert isinstance(exc, llm_calls.LLMTimeout) and exc.call == "conversation"
    assert [e["outcome"] for e in log] == ["retried", "gave up"]
    assert took < 4                            # two gaps of 0.5 s, not the 10 s the server would go on


# --- the web turn ----------------------------------------------------------------

class StalledAgent:
    """The conversation model times out, as the client would report it."""
    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages):
        raise asyncio.TimeoutError()


def test_web_turn_ends_with_a_message_and_the_session_file_logs_it(tmp_path):
    """The first turn needs no conversation model; the first chat message then
    stalls twice: the teacher gets a short message in the session's language."""
    from fastapi.testclient import TestClient
    from langgraph.checkpoint.memory import InMemorySaver
    from test_assistant import FAKE_ESCO, Judge

    from esco_agent.assistant import build_assistant
    from esco_agent.web.app import Sessions, create_app
    sessions = Sessions(graph=build_assistant(FAKE_ESCO, StalledAgent(), Judge(), InMemorySaver()),
                        model="fake", results_dir=str(tmp_path))
    c = TestClient(create_app(sessions))
    sid, _ = begin(c, lang="pt")
    r = c.post("/api/message", json={"session": sid, "text": "que profissões?"})
    assert r.status_code == 200, r.text
    new = r.json()["new"]
    assert new[-1]["role"] == "assistant" and "não respondeu a tempo" in new[-1]["content"]
    log = saved(sessions, sid)["llm_timeouts"]
    assert (log["count"], log["gave_up"]) == (2, 1)
    assert {e["call"] for e in log["calls"]} == {"conversation"}
    r = c.post("/api/message", json={"session": sid, "text": "e agora?"})   # the session goes on
    assert r.status_code == 200


# --- a streamed reply that stalls after part of it reached the page ----------------

def test_streamed_reply_is_not_retried_once_text_has_reached_the_page():
    class CutOff:
        calls = 0

        async def ainvoke(self, messages):
            CutOff.calls += 1
            llm_calls.shown("Three occupations fit: ")   # streamed to the page
            raise asyncio.TimeoutError()
    log = llm_calls.start()
    with pytest.raises(llm_calls.LLMTimeout) as err:
        asyncio.run(llm_calls.call(CutOff(), [], "conversation", streamed=True))
    assert err.value.partial == "Three occupations fit: " and CutOff.calls == 1
    assert [(e["attempt"], e["outcome"]) for e in log] == [(1, "interrupted")]
    assert llm_calls.summary(log)["interrupted"] == 1


def test_streamed_reply_is_retried_if_nothing_had_reached_the_page():
    class Silent:
        calls = 0

        async def ainvoke(self, messages):
            Silent.calls += 1
            if Silent.calls == 1:
                raise asyncio.TimeoutError()             # stalled before its first chunk
            llm_calls.shown("ok")
            return AIMessage("ok")
    log = llm_calls.start()
    assert asyncio.run(llm_calls.call(Silent(), [], "conversation", streamed=True)).content == "ok"
    assert [e["outcome"] for e in log] == ["retried"]


class CutOffAgent:
    """The conversation model stalls after part of its reply was streamed."""
    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages):
        llm_calls.shown("Partial reply about ")
        raise asyncio.TimeoutError()


def _web(tmp_path, agent, judge=None):
    from fastapi.testclient import TestClient
    from langgraph.checkpoint.memory import InMemorySaver
    from test_assistant import FAKE_ESCO, Judge

    from esco_agent.assistant import build_assistant
    from esco_agent.web.app import Sessions, create_app
    sessions = Sessions(graph=build_assistant(FAKE_ESCO, agent, judge or Judge(), InMemorySaver()),
                        model="fake", results_dir=str(tmp_path))
    return TestClient(create_app(sessions)), sessions


def test_web_keeps_an_interrupted_reply_marked_and_does_not_retry(tmp_path):
    c, sessions = _web(tmp_path, CutOffAgent())
    sid, _ = begin(c, lang="en")
    r = c.post("/api/message", json={"session": sid, "text": "which occupations?"})
    assert r.status_code == 200, r.text
    new = [m for m in r.json()["new"] if m["role"] == "assistant"]
    assert new[0]["content"] == "Partial reply about " and new[0]["interrupted"] is True
    assert "did not answer in time" in new[1]["content"]
    log = saved(sessions, sid)["llm_timeouts"]
    assert (log["count"], log["interrupted"]) == (1, 1) and log["calls"][0]["call"] == "conversation"


# --- the first turn gives up: "Try again" -------------------------------------------

class StallingJudge:
    """The queries/judge model; stalls while `stall` is set."""
    stall = True

    def __init__(self):
        from test_assistant import Judge
        self.judge = Judge()

    async def ainvoke(self, messages):
        if StallingJudge.stall:
            await asyncio.sleep(3600)
        return await self.judge.ainvoke(messages)


def test_first_turn_gives_up_then_try_again_fails_again_then_works(tmp_path):
    from test_assistant import Agent
    StallingJudge.stall = True
    c, sessions = _web(tmp_path, Agent(AIMessage("De nada.")), StallingJudge())
    sid, done = begin(c, lang="pt")                   # the course text is complete: first turn runs
    s = sessions.sessions[sid]
    sections, expected = dict(s["sections"]), s["expected"]
    assert done["stage"] == "first_turn_failed"
    assert done["chips"] == [{"key": "retry_first_turn", "text": "Tentar novamente", "send": True}]
    assert "não respondeu a tempo" in done["new"][-1]["content"]

    # anything but the button is refused, and changes nothing
    r = c.post("/api/message", json={"session": sid, "text": "então?"})
    assert r.status_code == 400 and "Tentar novamente" in r.json()["detail"]

    # the button, while the model still stalls: a second failure, the button again
    r = c.post("/api/message", json={"session": sid, "text": "Tentar novamente",
                                     "from_button": "retry_first_turn"})
    assert r.status_code == 200, r.text
    assert r.json()["stage"] == "first_turn_failed" and r.json()["chips"][0]["key"] == "retry_first_turn"

    # the model answers again: the button reruns the first turn on the same text
    StallingJudge.stall = False
    r = c.post("/api/message", json={"session": sid, "text": "Tentar novamente",
                                     "from_button": "retry_first_turn"})
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["stage"] == "chat" and "computer programming" in out["new"][-1]["content"]
    assert s["sections"] == sections and s["expected"] == expected
    log = saved(sessions, sid)
    assert [e["event"] for e in log["first_turn"]] == ["gave up", "retry", "gave up", "retry", "ok"]
    assert log["first_turn"][0]["call"] == "queries"
    assert log["llm_timeouts"]["gave_up"] == 2
    # the conversation goes on from the successful first turn
    r = c.post("/api/message", json={"session": sid, "text": "obrigado"})
    assert r.status_code == 200
