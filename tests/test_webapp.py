"""The web page's backend and rendering, with the fakes from test_assistant. No data or API."""

import asyncio
import json

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from test_assistant import FAKE_ESCO, Agent, Judge, call

from esco_agent.assistant import build_assistant
from esco_agent.webapp import Backend, UserError, build_app, chat_history, list_panel

LO = "Students write programs in C. They test their code."
CO = "- pointers and dynamic memory\n- the Linux command line"


def backend(tmp_path, *replies):
    graph = build_assistant(FAKE_ESCO, Agent(*replies), Judge(), InMemorySaver())
    return Backend(graph=graph, model="fake-model", sessions_dir=str(tmp_path))


def test_a_session_from_start_to_saved_file(tmp_path):
    Judge.drafts[:] = [json.dumps({"section": "Contents", "text": "Students use gdb."})]
    b = backend(tmp_path,
                call("update_skill_list", accept=[1]), AIMessage("Mantive a 1."),
                call("find_occupations"), AIMessage("."),
                call("check_occupation_gap", occupation=1), AIMessage("."),
                call("suggest_text_change", skill="debug software"), AIMessage("Sugestão."))

    async def go():
        session, state = await b.start("pt", "T 01!", "programar em C\ntestar", LO, CO)
        for q in ["manter 1", "profissões?", "o que falta?", "cobrir depuração"]:
            state = await b.send(session, q)
        return session, state

    session, state = asyncio.run(go())
    assert chat_history(state)[0]["content"].startswith("Li a unidade curricular")
    assert chat_history(state)[1:3] == [{"role": "user", "content": "manter 1"},
                                        {"role": "assistant", "content": "Mantive a 1."}]
    assert all("course" != m["content"] and m["content"] for m in chat_history(state))

    panel = list_panel(state)
    assert "### Lista de competências (1)" in panel and "1. **computer programming**" in panel
    assert "“Students write programs in C.” (Learning outcomes)" in panel
    assert "### Sugeridas, por decidir" in panel and "2. perform software unit testing" in panel
    assert "### Alterações de texto sugeridas" in panel
    assert "acrescentar a **Contents** · verificação passou · proposta\n   > Students use gdb." in panel

    assert "_T01_" in session["path"]                      # participant code made filename-safe
    saved = json.load(open(session["path"], encoding="utf-8"))
    assert saved["expected_skills_before"] == "programar em C\ntestar"
    assert saved["participant"] == "T01" and saved["model"] == "fake-model"
    assert saved["course"] == {"Learning outcomes": LO, "Contents": CO}
    assert [s["label"] for s in saved["skills"]] == ["computer programming"]
    assert len(saved["transcript"]) == len(state["messages"])
    # the teacher's own list never reaches the assistant
    assert "programar em C" not in json.dumps(saved["transcript"])


def test_two_visitors_do_not_share_a_list(tmp_path):
    b = backend(tmp_path, call("update_skill_list", accept=[1]), AIMessage("ok"))

    async def go():
        s1, _ = await b.start("en", "a", "", LO, CO)
        s2, st2 = await b.start("en", "b", "", LO, CO)
        st1 = await b.send(s1, "keep 1")
        return st1, st2

    st1, st2 = asyncio.run(go())
    assert st1["known"][0]["status"] == "accepted" and st2["known"][0]["status"] == "pending"
    assert "*Nothing yet." in list_panel(st2)


def test_bad_input_is_refused_with_a_message(tmp_path):
    b = backend(tmp_path)

    async def go():
        with pytest.raises(UserError, match="Paste the course"):
            await b.start("en", "", "", "  ", "")
        with pytest.raises(UserError, match="over 12000"):
            await b.start("en", "", "", "x" * 13000, "")
        session, _ = await b.start("en", "", "", LO, "")
        with pytest.raises(UserError, match="Write a message"):
            await b.send(session, "   ")
        with pytest.raises(UserError, match="limited to 2000"):
            await b.send(session, "x" * 2001)
        session["turns"] = 40
        with pytest.raises(UserError, match="message limit"):
            await b.send(session, "one more")

    asyncio.run(go())


def test_page_builds(tmp_path):
    pytest.importorskip("gradio")
    assert build_app(backend(tmp_path)) is not None
