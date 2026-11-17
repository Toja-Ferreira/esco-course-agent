"""The skill assistant as a web page: a chat on the left, the skill list on the right.

    python -m esco_agent.webapp          # http://127.0.0.1:7860

One process serves every visitor. It starts ONE ESCO MCP server (so the models
load once) and builds ONE graph; each visitor gets their own conversation
thread in it. Every session is saved to results/sessions/ after each turn.

Before the conversation starts, the teacher is asked to write the skills they
expect, in their own words. That list is saved with the session and is never
shown to the assistant: it is what the suggestions are compared with afterwards.

Environment:
    DEEPSEEK_API_KEY, DEEPSEEK_MODEL, ESCO_DATA_DIR   as for the command line
    HOST, PORT                 where to listen (default 127.0.0.1:7860)
    ESCO_WEB_USER, ESCO_WEB_PASSWORD   if both are set, the page asks for a login
"""

from __future__ import annotations

import asyncio
import os
import re
import uuid
from datetime import datetime
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage

from esco_agent.assistant import _shorten, save_session
from esco_agent.suggest import course_text

MAX_COURSE_CHARS = 12000    # both sections together
MAX_MESSAGE_CHARS = 2000
MAX_TURNS = 40              # teacher messages per session
SESSIONS_DIR = "results/sessions"

UI = {
    "en": {"list": "Skill list", "empty": "Nothing yet. Tell the assistant which skills to keep.",
           "undecided": "Suggested, not decided", "not_in_text": "not in the course text",
           "changes": "Suggested text changes", "passed": "check passed", "failed": "check failed",
           "status": {"proposed": "proposed", "accepted": "accepted", "rejected": "rejected"},
           "add_to": "add to"},
    "pt": {"list": "Lista de competências", "empty": "Ainda vazia. Diga ao assistente quais quer manter.",
           "undecided": "Sugeridas, por decidir", "not_in_text": "não está no texto",
           "changes": "Alterações de texto sugeridas", "passed": "verificação passou",
           "failed": "verificação falhou",
           "status": {"proposed": "proposta", "accepted": "aceite", "rejected": "rejeitada"},
           "add_to": "acrescentar a"},
}


class UserError(Exception):
    """A problem with what the visitor entered; the message is shown to them."""


# --- what the page shows ------------------------------------------------------

def chat_history(state: dict) -> list[dict]:
    """The conversation without the course text, tool calls or tool results."""
    out = []
    for m in state["messages"][1:]:
        if isinstance(m, HumanMessage):
            out.append({"role": "user", "content": m.content})
        elif isinstance(m, AIMessage) and m.content:
            out.append({"role": "assistant", "content": m.content})
    return out


def list_panel(state: dict) -> str:
    t = UI[state.get("language") or "en"]
    known = list(enumerate(state.get("known") or [], 1))
    accepted = [(n, s) for n, s in known if s["status"] == "accepted"]
    pending = [(n, s) for n, s in known if s["status"] == "pending"]
    lines = [f"### {t['list']} ({len(accepted)})"]
    if not accepted:
        lines.append(f"*{t['empty']}*")
    for n, s in accepted:
        if s["in_text"]:
            e = s["evidence"][0]
            note = f"\n   <small>“{_shorten(e['text'], 110)}” ({e['section']})</small>"
        else:
            note = f" ⚠ *{t['not_in_text']}*"
        lines.append(f"{n}. **{s['label']}**{note}")
    if pending:
        lines += ["", f"### {t['undecided']}"]
        lines += [f"{n}. {s['label']}" for n, s in pending]
    changes = state.get("text_changes") or []
    if changes:
        lines += ["", f"### {t['changes']}"]
        for n, c in enumerate(changes, 1):
            check = t["passed"] if c["verified"] else t["failed"]
            lines.append(f"{n}. *{c['label']}* · {t['add_to']} **{c['section']}** · "
                         f"{check} · {t['status'][c['status']]}\n   > {c['text']}")
    return "\n".join(lines)


# --- sessions -----------------------------------------------------------------

class Backend:
    """Owns the MCP server connection and the graph, and runs the sessions."""

    def __init__(self, graph=None, model: str = "", sessions_dir: str = SESSIONS_DIR):
        self._graph, self.model, self.sessions_dir = graph, model, sessions_dir
        self._lock = asyncio.Lock()
        self._holder = None

    async def graph(self):
        async with self._lock:
            if self._graph is None:
                ready = asyncio.get_running_loop().create_future()
                self._holder = asyncio.create_task(self._hold(ready))
                self._graph, self.model = await ready
        return self._graph

    async def _hold(self, ready) -> None:
        """Runs for the life of the process: keeps the MCP server connection open
        in a task of its own, so every request can use it."""
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

    def _save(self, session: dict, state: dict) -> None:
        served = state["messages"][-1].response_metadata.get("model_name") or self.model
        save_session(session["path"], state, served, "web", extra=session["meta"])

    async def start(self, language: str, participant: str, expected: str,
                    outcomes: str, contents: str) -> tuple[dict, dict]:
        sections = {name: text.strip() for name, text in
                    (("Learning outcomes", outcomes or ""), ("Contents", contents or ""))
                    if text.strip()}
        if not sections:
            raise UserError("Paste the course's learning outcomes or contents first. / "
                            "Cole primeiro os resultados de aprendizagem ou os conteúdos.")
        if sum(map(len, sections.values())) > MAX_COURSE_CHARS:
            raise UserError(f"The course text is over {MAX_COURSE_CHARS} characters. / "
                            f"O texto tem mais de {MAX_COURSE_CHARS} caracteres.")
        graph = await self.graph()
        thread = uuid.uuid4().hex
        who = re.sub(r"[^A-Za-z0-9_-]", "", participant or "")[:20] or "web"
        Path(self.sessions_dir).mkdir(parents=True, exist_ok=True)
        session = {
            "thread_id": thread, "turns": 0,
            "path": str(Path(self.sessions_dir) /
                        f"{datetime.now():%Y-%m-%d_%H%M%S}_{who}_{thread[:6]}.json"),
            "meta": {"participant": who, "started": datetime.now().isoformat(timespec="seconds"),
                     "expected_skills_before": (expected or "").strip()},
        }
        state = await graph.ainvoke(
            {"sections": sections, "language": language if language in UI else "en",
             "messages": [HumanMessage(course_text(sections))]},
            {"configurable": {"thread_id": thread}})
        self._save(session, state)
        return session, state

    async def send(self, session: dict, text: str) -> dict:
        text = (text or "").strip()
        if not text:
            raise UserError("Write a message first. / Escreva primeiro uma mensagem.")
        if len(text) > MAX_MESSAGE_CHARS:
            raise UserError(f"Messages are limited to {MAX_MESSAGE_CHARS} characters. / "
                            f"As mensagens têm um limite de {MAX_MESSAGE_CHARS} caracteres.")
        if session["turns"] >= MAX_TURNS:
            raise UserError("This session has reached its message limit; the list is saved. / "
                            "Esta sessão atingiu o limite de mensagens; a lista está guardada.")
        graph = await self.graph()
        session["turns"] += 1
        state = await graph.ainvoke({"messages": [HumanMessage(text)]},
                                    {"configurable": {"thread_id": session["thread_id"]}})
        self._save(session, state)
        return state


# --- the page -----------------------------------------------------------------

INTRO = """## ESCO skill assistant · Assistente de competências ESCO
Paste a course unit's text and build its list of [ESCO](https://esco.ec.europa.eu) skills \
in conversation. Each suggestion comes with the sentence of your text that supports it.

Cole o texto de uma unidade curricular e construa, em conversa, a sua lista de \
competências ESCO. Cada sugestão vem com a frase do seu texto que a suporta."""


def build_app(backend: Backend):
    import gradio as gr

    with gr.Blocks(title="ESCO skill assistant") as demo:
        session = gr.State(None)
        with gr.Column() as setup:
            gr.Markdown(INTRO)
            language = gr.Radio([("Português", "pt"), ("English", "en")], value="pt",
                                label="Língua da conversa / Chat language")
            participant = gr.Textbox(label="Código de participante (opcional) / Participant code (optional)",
                                     max_lines=1)
            expected = gr.Textbox(
                lines=5,
                label="Antes de começar: pelas suas palavras, que competências desenvolve esta "
                      "unidade curricular? Uma por linha. / Before you start: in your own words, "
                      "which skills does this course develop? One per line.")
            outcomes = gr.Textbox(lines=7, label="Resultados de aprendizagem, em inglês / "
                                                 "Learning outcomes, in English")
            contents = gr.Textbox(lines=7, label="Conteúdos, em inglês / Contents, in English")
            start = gr.Button("Começar / Start", variant="primary")
        with gr.Row(visible=False) as work:
            with gr.Column(scale=3):
                chat = gr.Chatbot(height=560, show_label=False)
                box = gr.Textbox(show_label=False, max_lines=4,
                                 placeholder="manter 1, 2 e 4 · porquê a 3? · há alguma coisa sobre…? "
                                             "· que profissões? · o que falta para a primeira?")
            with gr.Column(scale=2):
                panel = gr.Markdown()
                download = gr.DownloadButton("Descarregar a sessão / Download the session")

        async def on_start(lang, who, exp, lo, co):
            try:
                sess, state = await backend.start(lang, who, exp, lo, co)
            except UserError as e:
                raise gr.Error(str(e))
            return (sess, gr.update(visible=False), gr.update(visible=True),
                    chat_history(state), list_panel(state), sess["path"])

        async def on_send(text, sess):
            if sess is None:
                raise gr.Error("Start a session first. / Comece primeiro uma sessão.")
            try:
                state = await backend.send(sess, text)
            except UserError as e:
                raise gr.Error(str(e))
            return "", sess, chat_history(state), list_panel(state), sess["path"]

        start.click(on_start, [language, participant, expected, outcomes, contents],
                    [session, setup, work, chat, panel, download], api_name="start")
        box.submit(on_send, [box, session], [box, session, chat, panel, download],
                   api_name="send")
    return demo


def main() -> None:
    from dotenv import load_dotenv
    load_dotenv()
    user, password = os.environ.get("ESCO_WEB_USER"), os.environ.get("ESCO_WEB_PASSWORD")
    demo = build_app(Backend())
    demo.queue(default_concurrency_limit=4).launch(
        server_name=os.environ.get("HOST", "127.0.0.1"),
        server_port=int(os.environ.get("PORT", "7860")),
        auth=(user, password) if user and password else None)


if __name__ == "__main__":
    main()
