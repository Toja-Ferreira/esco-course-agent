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
import logging
import os
import secrets
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
from starlette.middleware.sessions import SessionMiddleware

from esco_agent.assistant import SKILL_URI, _now, course_message, export
from esco_agent.grounding import check_reply
from esco_agent.suggest import course_text

log = logging.getLogger("esco_agent.web")
from esco_agent.web import auth, intake, pof_auth
from esco_agent.web.questionnaire import check_answers, questions

STATIC = Path(__file__).parent / "static"
MAX_MESSAGE_CHARS = 2000
MAX_TURNS = 40                  # conversation messages per session, after intake
DAILY_LIMIT = int(os.environ.get("ESCO_DAILY_LIMIT", "400"))   # model turns per day, all users
ESCO_PAGE = "https://esco.ec.europa.eu/en/classification/skill?uri="
OCCUPATION_PAGE = "https://esco.ec.europa.eu/en/classification/occupation?uri="
OCCUPATION_URI = "http://data.europa.eu/esco/occupation/"


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

def sections_after(state: dict) -> dict:
    """The sections that have accepted text changes, with the new text at the end
    (the same place the check put it). The course page itself is never changed."""
    out = {}
    for c in state.get("text_changes") or []:
        if c["status"] == "accepted":
            base = out.get(c["section"], (state.get("sections") or {}).get(c["section"], ""))
            out[c["section"]] = base.rstrip() + "\n" + c["text"]
    return out


def gap_now(gap: dict | None, known: list[dict]) -> dict | None:
    """An occupation's gap against the list as it is now (not as it was when checked)."""
    if not gap or not isinstance(gap.get("essential"), list):
        return gap
    number = {s["id"]: n for n, s in enumerate(known, 1)}
    on_list, in_text, missing = 0, [], []
    for sk in gap["essential"]:
        n = number.get(sk["id"])
        if n and known[n - 1]["status"] == "accepted":
            on_list += 1
        elif n and sk["in_text"]:
            in_text.append({"number": n, "label": sk["label"]})
        else:
            missing.append(sk["label"])
    return {"essential": len(gap["essential"]), "on_list": on_list,
            "in_text": in_text, "not_in_text": missing}


def panel(state: dict) -> dict:
    """The skill list and suggested text changes, as the page renders them."""
    skills = []
    for n, s in enumerate(state.get("known") or [], 1):
        skills.append({
            "number": n, "label": s["label"], "type": s["type"], "status": s["status"],
            "reuse": s.get("reuse", ""),
            "in_text": s["in_text"], "definition": s["description"],
            "evidence": [{"section": e["section"], "text": e["text"]} for e in s["evidence"][:3]],
            "origin": s["origin"],
            "url": ESCO_PAGE + quote(SKILL_URI + s["id"], safe=""),
        })
    skill_no = {s["id"]: n for n, s in enumerate(state.get("known") or [], 1)}
    changes = [{"number": n, "label": c["label"], "mode": c.get("mode", "show"),
                "skill_number": skill_no.get(c.get("skill_id")),
                "edited": bool(c.get("edited_by_teacher")),
                "add_to_course": c.get("add_to_course") or [], "section": c["section"],
                "section_text": (state.get("sections") or {}).get(c["section"], ""),
                "definition": c.get("definition", ""),
                "text": c["text"], "verified": c["verified"], "reason": c.get("reason", ""),
                "status": c["status"]}
               for n, c in enumerate(state.get("text_changes") or [], 1)]
    occupations = [{"rank": r, "label": o["label"], "isco_group": o.get("isco_group", ""),
                    "matched_on": [m["label"] for m in o.get("top_matching_skills", [])][:4],
                    "url": OCCUPATION_PAGE + quote(OCCUPATION_URI + o["id"], safe=""),
                    "gap": gap_now(o.get("gap"), state.get("known") or [])}
                   for r, o in enumerate(state.get("occupations") or [], 1)]
    return {"skills": skills, "text_changes": changes, "occupations": occupations}


# --- suggested next messages ----------------------------------------------------

CHIPS = {
    "pt": {"keep_all": "Manter todas as sugeridas",
           "why": "Porquê a número {n}?",
           "topic": "Há alguma coisa sobre {topic}?",
           "anything": "Há alguma coisa sobre ",
           "jobs": "Que ocupações estão relacionadas com esta lista?",
           "gap": "O que falta para {occupation}?",
           "text": "Sugere texto para cobrir «{skill}»",
           "accept_change": "Aceito a alteração {n}",
           "reword_change": "Quero reescrever a alteração {n}: "},
    "en": {"keep_all": "Keep all the suggested ones",
           "why": "Why number {n}?",
           "topic": "Is there anything on {topic}?",
           "anything": "Is there anything on ",
           "jobs": "Which occupations does this list relate to?",
           "gap": "What is missing for {occupation}?",
           "text": "Suggest text to cover \"{skill}\"",
           "accept_change": "I accept text change {n}",
           "reword_change": "I want to reword text change {n}: "},
}
MAX_CHIPS = 5


GENERIC_START = ("understand ", "implement ", "design ", "apply ", "analyse ", "analyze ",
                 "develop ", "use ", "learn ")


def suggestions(state: dict, language: str, asked: set[str] | None = None) -> list[dict]:
    """Next messages worth offering, chosen by rules from the conversation's state
    (no LLM call: instant, free, and the same for every teacher in the same state).
    asked: the teacher's earlier messages, lower-cased, so nothing is offered twice.
    "send": True sends on click; otherwise the text is put in the box to finish."""
    c = CHIPS[language if language in CHIPS else "en"]
    asked = asked or set()
    known = state.get("known") or []
    if not known and not state.get("queries"):
        return []
    out = []

    def add(key, text, send=True):
        if (len(out) < MAX_CHIPS and text.strip().lower() not in asked
                and all(x["text"] != text for x in out)):
            out.append({"key": key, "text": text, "send": send})

    # a text change waiting for a decision comes first (a failed one can't be accepted as is)
    for n, ch in enumerate(state.get("text_changes") or [], 1):
        if ch["status"] in ("proposed", "failed"):
            if ch["status"] == "proposed":
                add("accept_change", c["accept_change"].format(n=n))
            add("reword_change", c["reword_change"].format(n=n), send=False)
            break
    pending = [n for n, k in enumerate(known, 1) if k["status"] == "pending"]
    accepted = [k for k in known if k["status"] == "accepted"]
    for n in pending:                          # the first undecided one not asked about yet
        if c["why"].format(n=n).lower() not in asked:
            add("why", c["why"].format(n=n))
            break
    if len(pending) > 1:
        add("keep_all", c["keep_all"])
    occupations = state.get("occupations") or []
    if accepted and not occupations:
        add("jobs", c["jobs"])
    if occupations:
        add("gap", c["gap"].format(occupation=occupations[0]["label"]))
    # a skill an occupation needs that the text doesn't show: offer a text change
    changed = {ch["skill_id"] for ch in state.get("text_changes") or []}
    for skill in (state.get("offered") or {}).values():
        if skill["origin"].startswith("required by") and skill["id"] not in changed:
            add("text", c["text"].format(skill=skill["label"]))
            break
    # first-turn search phrases that led to no suggestion and haven't been searched
    # since; specific ones ("manage dynamic memory") before generic ones
    found_by = {k["origin"] for k in known}
    searched = {e["topic"].lower() for e in state.get("events") or [] if e.get("type") == "search"}
    topics = [q for q in state.get("queries") or []
              if f"search '{q}'" not in found_by and q.lower() not in searched]
    topics.sort(key=lambda q: q.lower().startswith(GENERIC_START))
    for q in topics:
        if c["topic"].format(topic=q).lower() not in asked:
            add("topic", c["topic"].format(topic=q))
            break
    add("anything", c["anything"], send=False)
    return out


# --- sessions -----------------------------------------------------------------

class Sessions:
    """Owns the graph and every visitor's session."""

    def __init__(self, graph=None, model: str = "", results_dir: str = "results"):
        self._graph, self.model = graph, model
        self.dir = Path(results_dir) / "sessions"
        self.salt = load_salt(results_dir)
        self.sessions: dict[str, dict] = {}
        self._day, self._used_today = None, 0
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

    def owned(self, sid: str, user: dict | None) -> dict:
        """With the login on, a session can only be used by the person who started it."""
        s = self.get(sid)
        if user and hash_email(user["email"], self.salt) != s["participant"]:
            raise Problem("This session belongs to another user. / "
                          "Esta sessão pertence a outro utilizador.")
        return s

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

    @staticmethod
    def _asked(s: dict) -> set[str]:
        return {m["content"].strip().lower() for m in s["chat"]
                if m["role"] == "user" and not m.get("intake")}

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
            "login": s.get("login"), "role": s.get("role"),
            "language": s["language"], "course_name": s["course_name"],
            "started": s["created"], "finished": s.get("finished"),
            "expected_skills_before": s["expected"],
            "model": served or self.model,
            "chat": s["chat"],            # what the teacher saw, with button use marked
            "resumed": s.get("resumed", []),   # times the page was reloaded and picked up again
            "answers": s.get("answers"),
            "agent": export(state) if state else None,   # list, events, full transcript
        }
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = s["path"].with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(s["path"])

    # --- 1. start ---

    def create(self, email: str, language: str, expected: str, verified: bool = False,
               login_kind: str = "ua", role: str | None = None, course_name: str = "") -> dict:
        """email: from a login (verified=True; login_kind "ua" or "pof"), or typed
        (local development). With course_name (asked on the first screen) the chat starts
        at the learning outcomes; without it, the chat asks for the name first."""
        email, expected = (email or "").strip(), (expected or "").strip()
        if "@" not in email or len(email) > 200:
            raise Problem("Please enter your email. / Por favor indique o seu email.")
        if len([ln for ln in expected.splitlines() if ln.strip()]) < 1 or len(expected) < 5:
            raise Problem("Please write the skills you expect before starting. / "
                          "Escreva as competências que espera antes de começar.")
        if (course_name or "").strip() and not intake.check_name(course_name):
            raise Problem("Please write only the course's name. / "
                          "Escreva apenas o nome da unidade curricular.")
        course_name = " ".join((course_name or "").split())
        language = language if language in intake.MESSAGES else "pt"
        stage = "outcomes" if course_name else "name"
        sid = uuid.uuid4().hex
        participant = hash_email(email, self.salt)
        s = {"id": sid, "participant": participant, "login": login_kind if verified else "typed",
             "role": role,
             "language": language,
             "expected": expected[:5000], "stage": stage, "course_name": course_name or None,
             "sections": {}, "started": False, "turns": 0, "chat": [],
             "created": _now(), "lock": asyncio.Lock(),
             "path": self.dir / f"{datetime.now():%Y-%m-%d_%H%M%S}_{participant}_{sid[:6]}.json"}
        self.sessions[sid] = s
        hello = (intake.MESSAGES[language]["hello_named"].format(name=course_name) if course_name
                 else intake.MESSAGES[language]["hello"])
        self._say(s, "assistant", hello)
        self.save(s)
        return {"session": sid, "language": language, "stage": stage, "chat": s["chat"],
                "course_name": s["course_name"]}

    # --- 2. the chat ---

    async def message(self, sid: str, text: str, from_button: str | None = None) -> dict:
        result = None
        async for event in self.message_events(sid, text, from_button):
            if "done" in event:
                result = event["done"]
        return result

    async def message_events(self, sid: str, text: str, from_button: str | None = None):
        """Like message(), but yields events while the assistant works:
        {"progress": {...}} when a step starts, {"token": "..."} as the model writes
        its reply, then {"done": result}. The page shows them as they arrive."""
        s = self.get(sid)
        text = (text or "").strip()
        if not text:
            raise Problem("Write a message first. / Escreva primeiro uma mensagem.")
        if len(text) > (intake.MAX_SECTION_CHARS + 500 if s["stage"] != "chat"
                        else MAX_MESSAGE_CHARS):
            raise Problem("That message is too long. / Essa mensagem é demasiado longa.")
        if s["stage"] in ("finished",):
            raise Problem("This session is finished. / Esta sessão terminou.")
        if s["stage"] in ("chat", "contents"):
            self._count_turn()
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
                    yield getter.result()
                else:
                    getter.cancel()
            while not queue.empty():
                yield queue.get_nowait()
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
                            "course_name": s["course_name"], **panel(state),
                            "chips": suggestions(state, s["language"], self._asked(s))}}

    async def _intake(self, s: dict, text: str) -> dict:
        r = intake.step(s["stage"], text, s["language"])
        if "name" in r:
            s["course_name"] = r["name"]
        if r.get("outcomes"):
            s["sections"]["Learning outcomes"] = r["outcomes"]
        if r.get("contents"):
            s["sections"]["Contents"] = r["contents"]
        if r.get("sections"):                 # a whole page, split by its headings
            s["sections"] = dict(r["sections"])
        s["stage"] = r["stage"]
        if r["stage"] != "ready" or r.get("sections"):
            self._say(s, "assistant", r["reply"])    # for a page: which sections were used
        if r["stage"] != "ready":
            return {}
        s["started"] = True
        state = await self._run(s, {"sections": s["sections"], "language": s["language"],
                                    "messages": [HumanMessage(course_message(s["sections"]))]})
        self._say(s, "assistant", state["messages"][-1].content)
        s["stage"] = "chat"
        return state

    async def _converse(self, s: dict, text: str) -> dict:
        graph = await self.graph()
        before = len((await graph.aget_state(self._config(s))).values["messages"])
        state = await self._run(s, {"messages": [HumanMessage(text)]})
        replies = [m.content for m in state["messages"][before + 1:]
                   if m.type == "ai" and m.content]
        reply = "\n\n".join(replies) or "…"
        issues = check_reply(reply, state, earlier=before)
        self._say(s, "assistant", reply, **({"grounding_issues": issues} if issues else {}))
        await graph.aupdate_state(self._config(s), {"events": [
            {"t": _now(), "type": "grounding", "ok": not issues, "issues": issues}]},
            as_node="first_turn")
        return (await graph.aget_state(self._config(s))).values

    def _count_turn(self) -> None:
        """Server-wide daily cap on model turns, a ceiling on cost whatever happens."""
        today = datetime.now().date()
        if self._day != today:
            self._day, self._used_today = today, 0
        if self._used_today >= DAILY_LIMIT:
            raise Problem("The assistant has reached today's limit; please try again tomorrow. / "
                          "O assistente atingiu o limite de hoje; tente novamente amanhã.")
        self._used_today += 1

    async def _run(self, s: dict, graph_input: dict) -> dict:
        """Run the graph, passing progress steps and the reply's tokens to the
        session's listener. Only the conversation model's tokens are passed on
        (node "agent"); the judge and drafting calls inside tools write JSON."""
        graph = await self.graph()
        report = s.get("progress") or (lambda _: None)
        async for mode, chunk in graph.astream(graph_input, self._config(s),
                                               stream_mode=["custom", "messages"]):
            if mode == "custom":
                report({"progress": chunk})
            else:
                message, meta = chunk
                text = message.content if isinstance(message.content, str) else ""
                if text and meta.get("langgraph_node") == "agent":
                    report({"token": text})
        return (await graph.aget_state(self._config(s))).values

    async def decide(self, sid: str, number: int, action: str, source: str = "button") -> dict:
        """Keep or remove a skill from the list panel, without the LLM.
        source: "button" (the list) or "finish" (the check when finishing)."""
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
                               "source": source if source in ("button", "finish") else "button"})
                skill["status"] = status
            # as_node="first_turn": a node that leads to END, so the next chat
            # message starts a normal turn.
            await graph.aupdate_state(self._config(s), {"known": known, "events": events},
                                      as_node="first_turn")
            word = {("pt", "accepted"): "mantida", ("pt", "rejected"): "removida",
                    ("en", "accepted"): "kept", ("en", "rejected"): "removed"}[(s["language"], status)]
            self._say(s, "action", f"{number}. {skill['label']}: {word}",
                      source=source if source in ("button", "finish") else "button")
            state = (await graph.aget_state(self._config(s))).values
            self.save(s, state)
            return {"stage": s["stage"], "new": s["chat"][-1:], **panel(state),
                    "chips": suggestions(state, s["language"], self._asked(s))}

    # --- resume (the page was reloaded, e.g. by POF's session dialog) ---

    async def resume(self, sid: str) -> dict:
        """Everything the page needs to redraw a session where it was. Waits for a
        reply still being written, so it comes back with the page."""
        s = self.get(sid)
        async with s["lock"]:
            state = await self._state(s)
            out = {"session": sid, "language": s["language"], "stage": s["stage"],
                   "course_name": s["course_name"], "chat": s["chat"],
                   "answered": bool(s.get("answers")),
                   "skills": [], "text_changes": [], "occupations": [], "chips": []}
            if state:
                out.update(panel(state), chips=suggestions(state, s["language"], self._asked(s)))
            s.setdefault("resumed", []).append(_now())
            self.save(s, state or None)
            return out

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
                    "course_name": s["course_name"], "questions": questions(s["language"]),
                    "questions_by_language": {lang: questions(lang) for lang in ("pt", "en")},
                    "text_changes": [c for c in panel(state)["text_changes"]
                                     if c["status"] == "accepted"],
                    "sections_after": sections_after(state)}

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
    email: str = ""
    language: str = "pt"
    expected: str
    course_name: str = ""


class MessageIn(BaseModel):
    session: str
    text: str
    from_button: str | None = None


class SkillIn(BaseModel):
    session: str
    number: int
    action: str
    source: str = "button"


class SessionIn(BaseModel):
    session: str


class AnswersIn(BaseModel):
    session: str
    language: str = ""
    sus: list[int]
    statements: dict
    open: dict = {}


def create_app(sessions: Sessions | None = None, oauth_config=None, pof_config=None) -> FastAPI:
    """Login, from the environment unless given: POF's login (ESCO_AUTH=pof), the
    university's (UA_OAUTH_*), or none (a typed email, for development).
    Pass False to turn one off in tests."""
    sessions = sessions or Sessions()
    app = FastAPI(title="ESCO skill assistant", docs_url=None, redoc_url=None)
    pof = pof_auth.PofConfig.from_env() if pof_config is None else pof_config or None
    oauth = None if pof else (auth.OAuthConfig.from_env() if oauth_config is None
                              else oauth_config or None)
    if oauth and not os.environ.get("ESCO_SESSION_SECRET"):
        raise RuntimeError("ESCO_SESSION_SECRET must be set when the university login is on")
    app.add_middleware(SessionMiddleware,
                       secret_key=os.environ.get("ESCO_SESSION_SECRET") or secrets.token_hex(32),
                       session_cookie="esco_session", same_site="lax", max_age=8 * 3600,
                       https_only=bool(oauth and oauth.redirect_uri.startswith("https")))

    @app.middleware("http")
    async def same_origin_frames(request: Request, call_next):
        # POF shows the page in a frame on the same address; no other site may frame it
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = "frame-ancestors 'self'"
        response.headers["X-Frame-Options"] = "SAMEORIGIN"
        return response

    def current_user(request: Request) -> dict | None:
        return request.session.get("user")

    async def require_user(request: Request) -> dict | None:
        """With a login on, every API call needs a logged-in user."""
        if pof:
            try:
                return await pof_auth.current_user(pof, request.headers.get("authorization"))
            except pof_auth.NotLoggedIn:
                raise HTTPException(401, "Please log in. / Por favor inicie sessão.") from None
            except pof_auth.NotAllowed:
                raise HTTPException(403, "This experiment isn't open to your account. / "
                                         "Esta experiência não está aberta à sua conta.") from None
        user = current_user(request)
        if oauth and not user:
            raise HTTPException(401, "Please log in. / Por favor inicie sessão.")
        return user

    @app.get("/auth/login")
    async def login_start(request: Request):
        if not oauth:
            return RedirectResponse("/")
        return RedirectResponse(auth.start(oauth, request.session), status_code=302)

    @app.get("/auth/callback")
    async def login_callback(request: Request, code: str | None = None, state: str | None = None):
        if not oauth:
            return RedirectResponse("/")
        try:
            user = await auth.finish(oauth, request.session, code, state)
        except auth.LoginError as e:
            request.session["login_error"] = str(e)
            return RedirectResponse("/", status_code=302)
        # keep only what the page needs; the email is hashed when a session starts
        request.session["user"] = {"email": user["email"],
                                   "first_name": user["name"].split(" ")[0] if user["name"] else ""}
        return RedirectResponse("/", status_code=302)

    @app.get("/auth/logout")
    async def logout(request: Request):
        request.session.clear()
        return RedirectResponse("/", status_code=302)

    @app.get("/api/me")
    async def me(request: Request):
        if pof:
            out = {"login_required": True, "mode": "pof", "logged_in": False, "allowed": False,
                   "first_name": "", "error": None,
                   "login_url": pof.login_url, "logout_url": pof.logout_url}
            try:
                user = await pof_auth.current_user(pof, request.headers.get("authorization"))
                out.update(logged_in=True, allowed=True, first_name=user["first_name"])
            except pof_auth.NotLoggedIn:
                pass
            except pof_auth.NotAllowed:
                out.update(logged_in=True, error="Esta experiência não está aberta à sua conta. / "
                                                 "This experiment isn't open to your account.")
            return out
        user = current_user(request)
        return {"login_required": bool(oauth), "mode": "ua" if oauth else "none",
                "logged_in": bool(user), "allowed": bool(user) or not oauth,
                "first_name": (user or {}).get("first_name", ""),
                "error": request.session.pop("login_error", None),
                "login_url": "auth/login", "logout_url": "auth/logout"}
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
    async def start(body: StartIn, user: dict | None = Depends(require_user)):
        if user:       # the login provides the email; nothing typed is used
            return await sync(sessions.create, user["email"], body.language, body.expected, True,
                              "pof" if pof else "ua", user.get("role"), body.course_name)
        return await sync(sessions.create, body.email, body.language, body.expected,
                          False, "ua", None, body.course_name)

    def check(sid: str, user: dict | None) -> None:
        try:
            sessions.owned(sid, user)
        except Problem as e:
            raise HTTPException(403 if "another user" in str(e) else 400, str(e)) from None

    @app.post("/api/message", dependencies=[Depends(login)])
    async def message(body: MessageIn, user: dict | None = Depends(require_user)):
        check(body.session, user)
        return await run(sessions.message(body.session, body.text, body.from_button))

    @app.post("/api/message/stream", dependencies=[Depends(login)])
    async def message_stream(body: MessageIn, user: dict | None = Depends(require_user)):
        """Newline-delimited JSON: {"progress": ...} and {"token": ...} lines, then
        {"done": ...} or {"error": ...}."""
        check(body.session, user)
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
                log.exception("message failed")      # the cause goes to the server log only
                yield json.dumps({"error": "Something went wrong. / Ocorreu um erro."}) + "\n"

        return StreamingResponse(lines(), media_type="application/x-ndjson")

    @app.post("/api/skill", dependencies=[Depends(login)])
    async def skill(body: SkillIn, user: dict | None = Depends(require_user)):
        check(body.session, user)
        return await run(sessions.decide(body.session, body.number, body.action, body.source))

    @app.post("/api/resume", dependencies=[Depends(login)])
    async def resume(body: SessionIn, user: dict | None = Depends(require_user)):
        check(body.session, user)
        return await run(sessions.resume(body.session))

    @app.post("/api/finish", dependencies=[Depends(login)])
    async def finish(body: SessionIn, user: dict | None = Depends(require_user)):
        check(body.session, user)
        return await run(sessions.finish(body.session))

    @app.post("/api/answers", dependencies=[Depends(login)])
    async def answers(body: AnswersIn, user: dict | None = Depends(require_user)):
        check(body.session, user)
        return await run(sessions.answers(body.session, body.model_dump()))

    @app.get("/api/download/{sid}", dependencies=[Depends(login)])
    async def download(sid: str, user: dict | None = Depends(require_user)):
        check(sid, user)
        s = sessions.get(sid)
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
