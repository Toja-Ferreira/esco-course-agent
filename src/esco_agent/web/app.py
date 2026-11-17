"""The skill assistant's web page: a small API over the assistant graph, and one
static page (static/index.html) that calls it.

    python -m esco_agent.web            # http://127.0.0.1:8000

One process serves every visitor. It starts ONE ESCO MCP server (the models load
once) and builds ONE graph; each visitor gets their own conversation thread.
The session file under results/sessions/ is rewritten after every change.

Flow (see the spec's page design):
  1. POST /api/session    email (stored only as a salted hash), language and the
                          teacher's expected skills, which never reach the assistant
  2. POST /api/message    intake (fixed messages: course name, learning outcomes,
                          contents), then the first suggestions, then the conversation
     POST /api/skill      keep / remove a skill from the list panel
  3. POST /api/finish     the review data for the closing questions
     POST /api/answers    the closing questions
     GET  /api/download/{id}   the session file

Environment: DEEPSEEK_API_KEY, DEEPSEEK_MODEL, ESCO_DATA_DIR as for the command
line; HOST, PORT; ESCO_WEB_USER + ESCO_WEB_PASSWORD to require a login;
ESCO_SALT for the email hash (default: a random salt kept in results/.salt, so
the same email gives the same hash across restarts; keep that file).
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import secrets
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from esco_agent.assistant import SKILL_URI, _now, export
from esco_agent.suggest import course_text
from esco_agent.web import intake
from esco_agent.web.questionnaire import check_answers, questions

STATIC = Path(__file__).parent / "static"
MAX_MESSAGE_CHARS = 2000
MAX_TURNS = 40                  # conversation messages per session, after intake
ESCO_PAGE = "https://esco.ec.europa.eu/en/classification/skill?uri="


class Problem(Exception):
    """Something the visitor should be told; becomes an HTTP 400 with this text."""


# --- identity -----------------------------------------------------------------

def load_salt(results_dir: str) -> bytes:
    if os.environ.get("ESCO_SALT"):
        return os.environ["ESCO_SALT"].encode()
    path = Path(results_dir) / ".salt"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(secrets.token_hex(16))
    return path.read_text().strip().encode()


def hash_email(email: str, salt: bytes) -> str:
    return hmac.new(salt, email.strip().lower().encode(), hashlib.sha256).hexdigest()[:16]


# --- what the page shows ------------------------------------------------------

def panel(state: dict) -> dict:
    """The skill list and suggested text changes, as the page renders them."""
    skills = []
    for n, s in enumerate(state.get("known") or [], 1):
        skills.append({
            "number": n, "label": s["label"], "type": s["type"], "status": s["status"],
            "in_text": s["in_text"], "definition": s["description"],
            "evidence": [{"section": e["section"], "text": e["text"]} for e in s["evidence"][:3]],
            "origin": s["origin"],
            "url": ESCO_PAGE + quote(SKILL_URI + s["id"], safe=""),
        })
    changes = [{"number": n, "label": c["label"], "section": c["section"], "text": c["text"],
                "verified": c["verified"], "status": c["status"]}
               for n, c in enumerate(state.get("text_changes") or [], 1)]
    return {"skills": skills, "text_changes": changes}


# --- sessions -----------------------------------------------------------------

class Sessions:
    """Owns the graph and every visitor's session."""

    def __init__(self, graph=None, model: str = "", results_dir: str = "results"):
        self._graph, self.model = graph, model
        self.dir = Path(results_dir) / "sessions"
        self.salt = load_salt(results_dir)
        self.sessions: dict[str, dict] = {}
        self._graph_lock = asyncio.Lock()

    # The MCP server connection lives in a task of its own for the life of the process.
    async def graph(self):
        async with self._graph_lock:
            if self._graph is None:
                ready = asyncio.get_running_loop().create_future()
                self._holder = asyncio.create_task(self._hold(ready))
                self._graph, self.model = await ready
        return self._graph

    async def _hold(self, ready) -> None:
        from langchain_mcp_adapters.client import MultiServerMCPClient
        from langchain_mcp_adapters.tools import load_mcp_tools
        from langgraph.checkpoint.memory import InMemorySaver

        from esco_agent.assistant import build_assistant, deepseek_models, esco_from_mcp
        from esco_agent.mcp_client import server_config
        try:
            async with MultiServerMCPClient(server_config()).session("esco") as session:
                model, llm, judge = deepseek_models()
                graph = build_assistant(esco_from_mcp(await load_mcp_tools(session)),
                                        llm, judge, InMemorySaver())
                ready.set_result((graph, model))
                await asyncio.Event().wait()
        except Exception as exc:
            if not ready.done():
                ready.set_exception(exc)
            raise

    def get(self, sid: str) -> dict:
        s = self.sessions.get(sid)
        if s is None:
            raise Problem("Session not found; please start again. / "
                          "Sessão não encontrada; por favor recomece.")
        return s

    def _config(self, s: dict) -> dict:
        return {"configurable": {"thread_id": s["id"]}}

    async def _state(self, s: dict) -> dict:
        if not s["started"]:
            return {}
        return (await (await self.graph()).aget_state(self._config(s))).values

    def _say(self, s: dict, role: str, content: str, **extra) -> dict:
        msg = {"role": role, "content": content, "t": _now(), **extra}
        s["chat"].append(msg)
        return msg

    def save(self, s: dict, state: dict | None = None) -> None:
        state = state or {}
        messages = state.get("messages") or []
        served = (messages[-1].response_metadata.get("model_name")
                  if messages and hasattr(messages[-1], "response_metadata") else None)
        data = {
            "saved": _now(), "session": s["id"], "participant": s["participant"],
            "language": s["language"], "course_name": s["course_name"],
            "started": s["created"], "finished": s.get("finished"),
            "expected_skills_before": s["expected"],
            "model": served or self.model,
            "chat": s["chat"],            # what the teacher saw, with button use marked
            "answers": s.get("answers"),
            "agent": export(state) if state else None,   # list, events, full transcript
        }
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = s["path"].with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(s["path"])

    # --- 1. start ---

    def create(self, email: str, language: str, expected: str) -> dict:
        email, expected = (email or "").strip(), (expected or "").strip()
        if "@" not in email or len(email) > 200:
            raise Problem("Please enter your email. / Por favor indique o seu email.")
        if len([ln for ln in expected.splitlines() if ln.strip()]) < 1 or len(expected) < 5:
            raise Problem("Please write the skills you expect before starting. / "
                          "Escreva as competências que espera antes de começar.")
        language = language if language in intake.MESSAGES else "pt"
        sid = uuid.uuid4().hex
        participant = hash_email(email, self.salt)
        s = {"id": sid, "participant": participant, "language": language,
             "expected": expected[:5000], "stage": "name", "course_name": None,
             "sections": {}, "started": False, "turns": 0, "chat": [],
             "created": _now(), "lock": asyncio.Lock(),
             "path": self.dir / f"{datetime.now():%Y-%m-%d_%H%M%S}_{participant}_{sid[:6]}.json"}
        self.sessions[sid] = s
        self._say(s, "assistant", intake.MESSAGES[language]["hello"])
        self.save(s)
        return {"session": sid, "language": language, "stage": "name", "chat": s["chat"]}

    # --- 2. the chat ---

    async def message(self, sid: str, text: str, from_button: str | None = None) -> dict:
        result = None
        async for event in self.message_events(sid, text, from_button):
            if "done" in event:
                result = event["done"]
        return result

    async def message_events(self, sid: str, text: str, from_button: str | None = None):
        """Like message(), but yields {"progress": {...}} while the assistant works,
        then {"done": result}. The page shows the progress as it arrives."""
        s = self.get(sid)
        text = (text or "").strip()
        if not text:
            raise Problem("Write a message first. / Escreva primeiro uma mensagem.")
        if len(text) > (intake.MAX_SECTION_CHARS + 500 if s["stage"] != "chat"
                        else MAX_MESSAGE_CHARS):
            raise Problem("That message is too long. / Essa mensagem é demasiado longa.")
        if s["stage"] in ("finished",):
            raise Problem("This session is finished. / Esta sessão terminou.")
        async with s["lock"]:
            queue: asyncio.Queue = asyncio.Queue()
            s["progress"] = queue.put_nowait
            start = len(s["chat"])
            # The course text is long; the chat keeps it, marked, so the transcript is complete.
            self._say(s, "user", text, **({"from_button": from_button} if from_button else {}),
                      **({"intake": s["stage"]} if s["stage"] != "chat" else {}))
            if s["stage"] == "chat":
                if s["turns"] >= MAX_TURNS:
                    s["chat"].pop()
                    raise Problem("This session has reached its message limit; press Finish. / "
                                  "Esta sessão atingiu o limite de mensagens; carregue em Terminar.")
                s["turns"] += 1
                work = asyncio.create_task(self._converse(s, text))
            else:
                work = asyncio.create_task(self._intake(s, text))
            while not work.done():
                getter = asyncio.create_task(queue.get())
                await asyncio.wait({work, getter}, return_when=asyncio.FIRST_COMPLETED)
                if getter.done():
                    yield {"progress": getter.result()}
                else:
                    getter.cancel()
            while not queue.empty():
                yield {"progress": queue.get_nowait()}
            try:
                state = work.result()
            except Exception:
                self._say(s, "assistant", "⚠ " + ("Ocorreu um erro. Tente novamente."
                                                 if s["language"] == "pt" else
                                                 "Something went wrong. Please try again."))
                self.save(s, await self._state(s))
                raise
            self.save(s, state)
            yield {"done": {"stage": s["stage"], "new": s["chat"][start:],
                            "course_name": s["course_name"], **panel(state)}}

    async def _intake(self, s: dict, text: str) -> dict:
        r = intake.step(s["stage"], text, s["language"])
        if "name" in r:
            s["course_name"] = r["name"]
        if r.get("outcomes"):
            s["sections"]["Learning outcomes"] = r["outcomes"]
        if r.get("contents"):
            s["sections"]["Contents"] = r["contents"]
        self._say(s, "assistant", r["reply"])
        s["stage"] = r["stage"]
        if r["stage"] != "ready":
            return {}
        s["started"] = True
        state = await self._run(s, {"sections": s["sections"], "language": s["language"],
                                    "messages": [HumanMessage(course_text(s["sections"]))]})
        self._say(s, "assistant", state["messages"][-1].content)
        s["stage"] = "chat"
        return state

    async def _converse(self, s: dict, text: str) -> dict:
        graph = await self.graph()
        before = len((await graph.aget_state(self._config(s))).values["messages"])
        state = await self._run(s, {"messages": [HumanMessage(text)]})
        replies = [m.content for m in state["messages"][before + 1:]
                   if m.type == "ai" and m.content]
        self._say(s, "assistant", "\n\n".join(replies) or "…")
        return state

    async def _run(self, s: dict, graph_input: dict) -> dict:
        """Run the graph, passing its progress events to the session's listener."""
        graph = await self.graph()
        report = s.get("progress") or (lambda _: None)
        async for chunk in graph.astream(graph_input, self._config(s), stream_mode="custom"):
            report(chunk)
        return (await graph.aget_state(self._config(s))).values

    async def decide(self, sid: str, number: int, action: str) -> dict:
        """Keep or remove a skill from the list panel, without the LLM."""
        s = self.get(sid)
        if s["stage"] != "chat":
            raise Problem("The skill list isn't ready yet. / A lista ainda não está pronta.")
        if action not in ("accept", "reject"):
            raise Problem("Unknown action.")
        async with s["lock"]:
            graph = await self.graph()
            state = (await graph.aget_state(self._config(s))).values
            known = [dict(k) for k in state["known"]]
            if not 1 <= number <= len(known):
                raise Problem(f"No skill number {number}.")
            skill = known[number - 1]
            status = "accepted" if action == "accept" else "rejected"
            events = []
            if skill["status"] != status:
                events.append({"t": _now(), "type": status, "skill": skill["id"],
                               "was": skill["status"], "in_text": skill["in_text"],
                               "source": "button"})
                skill["status"] = status
            # as_node="first_turn": a node that leads to END, so the next chat
            # message starts a normal turn.
            await graph.aupdate_state(self._config(s), {"known": known, "events": events},
                                      as_node="first_turn")
            word = {("pt", "accepted"): "mantida", ("pt", "rejected"): "removida",
                    ("en", "accepted"): "kept", ("en", "rejected"): "removed"}[(s["language"], status)]
            self._say(s, "action", f"{number}. {skill['label']}: {word}", source="button")
            state = (await graph.aget_state(self._config(s))).values
            self.save(s, state)
            return {"stage": s["stage"], "new": s["chat"][-1:], **panel(state)}

    # --- 3. finish ---

    async def finish(self, sid: str) -> dict:
        s = self.get(sid)
        async with s["lock"]:
            state = await self._state(s)
            s["stage"] = "finished"
            s["finished"] = s.get("finished") or _now()
            self.save(s, state)
            accepted = [x for x in panel(state)["skills"] if x["status"] == "accepted"]
            return {"expected": s["expected"], "skills": accepted,
                    "chat": [m for m in s["chat"] if not m.get("intake")],
                    "course_name": s["course_name"], "questions": questions(s["language"])}

    async def answers(self, sid: str, raw: dict) -> dict:
        s = self.get(sid)
        if s["stage"] != "finished":
            raise Problem("Press Finish first. / Carregue primeiro em Terminar.")
        try:
            clean = check_answers(raw)
        except ValueError as e:
            raise Problem(f"Please answer every question ({e}). / "
                          "Por favor responda a todas as perguntas.") from None
        async with s["lock"]:
            s["answers"] = {**clean, "submitted": _now()}
            self.save(s, await self._state(s))
        return {"ok": True}


# --- the app ------------------------------------------------------------------

class StartIn(BaseModel):
    email: str
    language: str = "pt"
    expected: str


class MessageIn(BaseModel):
    session: str
    text: str
    from_button: str | None = None


class SkillIn(BaseModel):
    session: str
    number: int
    action: str


class SessionIn(BaseModel):
    session: str


class AnswersIn(BaseModel):
    session: str
    sus: list[int]
    statements: dict
    open: dict = {}


def create_app(sessions: Sessions | None = None) -> FastAPI:
    sessions = sessions or Sessions()
    app = FastAPI(title="ESCO skill assistant", docs_url=None, redoc_url=None)
    user, password = os.environ.get("ESCO_WEB_USER"), os.environ.get("ESCO_WEB_PASSWORD")
    basic = HTTPBasic(auto_error=bool(user and password))

    def login(cred: HTTPBasicCredentials | None = Depends(basic)):
        if not (user and password):
            return
        ok = cred and secrets.compare_digest(cred.username, user) and \
            secrets.compare_digest(cred.password, password)
        if not ok:
            raise HTTPException(401, headers={"WWW-Authenticate": "Basic"})

    async def run(coro):
        try:
            return await coro
        except Problem as e:
            raise HTTPException(400, str(e)) from None

    async def sync(fn, *a):
        try:
            return fn(*a)
        except Problem as e:
            raise HTTPException(400, str(e)) from None

    @app.get("/", response_class=HTMLResponse, dependencies=[Depends(login)])
    async def page():
        return (STATIC / "index.html").read_text(encoding="utf-8")

    @app.post("/api/session", dependencies=[Depends(login)])
    async def start(body: StartIn):
        return await sync(sessions.create, body.email, body.language, body.expected)

    @app.post("/api/message", dependencies=[Depends(login)])
    async def message(body: MessageIn):
        return await run(sessions.message(body.session, body.text, body.from_button))

    @app.post("/api/message/stream", dependencies=[Depends(login)])
    async def message_stream(body: MessageIn):
        """Newline-delimited JSON: {"progress": ...} lines, then {"done": ...} or {"error": ...}."""
        events = sessions.message_events(body.session, body.text, body.from_button)
        try:
            first = await events.__anext__()   # input problems surface here, as HTTP 400
        except Problem as e:
            raise HTTPException(400, str(e)) from None

        async def lines():
            yield json.dumps(first, ensure_ascii=False) + "\n"
            try:
                async for event in events:
                    yield json.dumps(event, ensure_ascii=False) + "\n"
            except Problem as e:
                yield json.dumps({"error": str(e)}, ensure_ascii=False) + "\n"
            except Exception:
                yield json.dumps({"error": "Something went wrong. / Ocorreu um erro."}) + "\n"

        return StreamingResponse(lines(), media_type="application/x-ndjson")

    @app.post("/api/skill", dependencies=[Depends(login)])
    async def skill(body: SkillIn):
        return await run(sessions.decide(body.session, body.number, body.action))

    @app.post("/api/finish", dependencies=[Depends(login)])
    async def finish(body: SessionIn):
        return await run(sessions.finish(body.session))

    @app.post("/api/answers", dependencies=[Depends(login)])
    async def answers(body: AnswersIn):
        return await run(sessions.answers(body.session, body.model_dump()))

    @app.get("/api/download/{sid}", dependencies=[Depends(login)])
    async def download(sid: str):
        try:
            s = sessions.get(sid)
        except Problem as e:
            raise HTTPException(404, str(e)) from None
        return FileResponse(s["path"], media_type="application/json",
                            filename=f"esco-skills-{(s['course_name'] or 'course')[:40]}.json")

    return app


def main() -> None:
    import uvicorn
    from dotenv import load_dotenv
    load_dotenv()
    uvicorn.run(create_app(), host=os.environ.get("HOST", "127.0.0.1"),
                port=int(os.environ.get("PORT", "8000")))


if __name__ == "__main__":
    main()
