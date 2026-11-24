"""The web page's API, end to end with the fakes from test_assistant. No data or API key."""

import json

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from test_assistant import FAKE_ESCO, SECTIONS, Agent, Judge, call

from esco_agent.assistant import build_assistant
from esco_agent.web import intake
from esco_agent.web.app import Sessions, create_app, hash_email
from esco_agent.web.questionnaire import STATEMENTS, sus_score

OUTCOMES = SECTIONS["Learning outcomes"] + " Students learn to structure programs into functions."
CONTENTS = SECTIONS["Contents"] + "\n- arrays, strings and structures in the C language"


def client(tmp_path, *replies):
    graph = build_assistant(FAKE_ESCO, Agent(*replies), Judge(), InMemorySaver())
    sessions = Sessions(graph=graph, model="fake-model", results_dir=str(tmp_path))
    return TestClient(create_app(sessions)), sessions


def begin(c, lang="pt"):
    r = c.post("/api/session", json={"email": "Teacher@ua.pt ", "language": lang,
                                     "expected": "programar em C\ntestar código"})
    assert r.status_code == 200, r.text
    sid = r.json()["session"]
    for text in ["Programação", OUTCOMES, CONTENTS]:
        r = c.post("/api/message", json={"session": sid, "text": text})
        assert r.status_code == 200, r.text
    return sid, r.json()


def saved(sessions, sid):
    return json.loads(sessions.sessions[sid]["path"].read_text(encoding="utf-8"))


def test_whole_session_from_start_to_answers(tmp_path):
    c, sessions = client(tmp_path,
                         call("update_skill_list", accept=[2]), AIMessage("Mantive a 2."))
    sid, r = begin(c)
    assert r["stage"] == "chat" and r["course_name"] == "Programação"
    assert [m["role"] for m in r["new"]] == ["user", "assistant"]   # no "working…" message
    assert r["new"][-1]["content"].startswith("Li a unidade curricular")
    assert [s["label"] for s in r["skills"]] == ["computer programming",
                                                 "perform software unit testing"]
    first = r["skills"][0]
    assert first["url"] == ("https://esco.ec.europa.eu/en/classification/skill?uri="
                            "http%3A%2F%2Fdata.europa.eu%2Fesco%2Fskill%2Fprog")
    assert first["evidence"][0]["section"] == "Learning outcomes" and first["definition"]

    # list panel button, then a chat message started from a suggestion button
    r = c.post("/api/skill", json={"session": sid, "number": 1, "action": "accept"}).json()
    assert r["new"][0] == {**r["new"][0], "role": "action", "source": "button"}
    assert [s["status"] for s in r["skills"]] == ["accepted", "pending"]
    r = c.post("/api/message", json={"session": sid, "text": "manter a 2",
                                     "from_button": "why"}).json()
    assert r["new"][-1]["content"] == "Mantive a 2."
    assert [s["status"] for s in r["skills"]] == ["accepted", "accepted"]

    # the page was reloaded (e.g. POF's session dialog): the session comes back as it was
    back = c.post("/api/resume", json={"session": sid}).json()
    assert back["stage"] == "chat" and back["course_name"] == "Programação" and not back["answered"]
    assert back["chat"][-1]["content"] == "Mantive a 2."
    assert [s["status"] for s in back["skills"]] == ["accepted", "accepted"] and "chips" in back
    assert len(saved(sessions, sid)["resumed"]) == 1
    assert c.post("/api/resume", json={"session": "nope"}).status_code == 400

    data = saved(sessions, sid)
    assert data["participant"] == hash_email("teacher@ua.pt", sessions.salt)
    assert "teacher@ua.pt" not in json.dumps(data).lower()
    assert data["course_name"] == "Programação"
    user_msgs = [m for m in data["chat"] if m["role"] == "user"]
    assert user_msgs[-1]["from_button"] == "why" and user_msgs[1]["intake"] == "outcomes"
    events = [e for e in data["agent"]["events"] if e["type"] != "grounding"]
    assert [e["type"] for e in events] == ["suggested", "accepted", "accepted"]
    assert events[1]["source"] == "button" and "source" not in events[2]
    # the teacher's own list never reaches the assistant
    assert "programar em C" not in json.dumps(data["agent"]["transcript"])

    review = c.post("/api/finish", json={"session": sid}).json()
    assert review["expected"] == "programar em C\ntestar código"
    assert [s["label"] for s in review["skills"]] == ["computer programming",
                                                      "perform software unit testing"]
    assert not any(m.get("intake") for m in review["chat"])
    assert len(review["questions"]["sus"]) == 10 and len(review["questions"]["statements"]) == 5
    both = review["questions_by_language"]
    assert both["pt"]["scale"][0] == "Discordo totalmente" and both["en"]["scale"][0] == "Strongly disagree"
    assert [o["key"] for o in both["pt"]["open"]] == ["most_useful", "distrust", "comments"]
    assert both["pt"]["sus"][0] == "Acho que gostaria de utilizar este produto com frequência."
    assert both["pt"]["sus_language"] == "pt" and both["en"]["sus_language"] == "en"
    assert len(both["pt"]["sus"]) == 10
    assert review["text_changes"] == []

    assert c.post("/api/message", json={"session": sid, "text": "mais"}).status_code == 400
    bad = c.post("/api/answers", json={"session": sid, "sus": [3] * 10, "statements": {}})
    assert bad.status_code == 400
    answers = {"session": sid, "sus": [5, 1, 5, 1, 5, 1, 5, 1, 5, 1],
               "statements": {k: ("na" if v["na"] else 4) for k, v in STATEMENTS.items()},
               "open": {"most_useful": "as frases", "distrust": ""}, "language": "en"}
    assert c.post("/api/answers", json=answers).json() == {"ok": True}
    data = saved(sessions, sid)
    assert data["answers"]["sus_score"] == 100.0
    assert data["answers"]["statements"]["vs_pof"] == "na"
    assert data["answers"]["open"]["most_useful"] == "as frases"
    assert data["answers"]["open"]["comments"] == ""                # optional
    assert data["answers"]["questions_language"] == "en" and data["answers"]["sus_language"] == "en"

    assert c.post("/api/resume", json={"session": sid}).json()["answered"] is True

    d = c.get(f"/api/download/{sid}")
    assert d.status_code == 200 and json.loads(d.content)["session"] == sid


def test_intake_checks_each_answer(tmp_path):
    c, sessions = client(tmp_path)
    sid = c.post("/api/session", json={"email": "a@b.pt", "language": "en",
                                       "expected": "something useful"}).json()["session"]

    def say(text):
        r = c.post("/api/message", json={"session": sid, "text": text}).json()
        return r["stage"], r["new"][-1]["content"]

    assert say("Programming\nwith\nlong text")[0] == "name"
    assert say("Programming")[0] == "outcomes"
    stage, reply = say("Os estudantes aprendem a programar em C e a usar a linha de comandos "
                       "para compilação e execução dos programas.")
    assert stage == "outcomes" and "Portuguese" in reply
    stage, reply = say("Learning outcomes\nWrite C.\nAssessment\nTwo tests.\nBibliography\nK&R.")
    assert stage == "outcomes" and reply.startswith("This text is too short")   # a page, too thin
    assert say("Write C.")[1].startswith("This text is too short")
    assert say(OUTCOMES)[0] == "contents"
    stage, _ = say("none")
    assert stage == "chat"
    assert list(sessions.sessions[sid]["sections"]) == ["Learning outcomes"]


def test_start_needs_email_and_expected_skills(tmp_path):
    c, _ = client(tmp_path)
    assert c.post("/api/session", json={"email": "x", "expected": "abcdef"}).status_code == 400
    assert c.post("/api/session", json={"email": "a@b.pt", "expected": " "}).status_code == 400
    assert c.post("/api/message", json={"session": "nope", "text": "hi"}).status_code == 400


def test_email_hash_is_stable_and_salted():
    assert hash_email("A@B.pt ", b"s") == hash_email("a@b.pt", b"s")
    assert hash_email("a@b.pt", b"s") != hash_email("a@b.pt", b"t")


def test_page_is_served_and_login_can_be_required(tmp_path, monkeypatch):
    c, _ = client(tmp_path)
    assert "Assistente de competências" in c.get("/").text
    monkeypatch.setenv("ESCO_WEB_USER", "u")
    monkeypatch.setenv("ESCO_WEB_PASSWORD", "p")
    c2, _ = client(tmp_path)
    assert c2.get("/").status_code == 401
    assert c2.get("/", auth=("u", "p")).status_code == 200


def test_sus_scoring():
    assert sus_score([3] * 10) == 50.0
    assert sus_score([1, 5, 1, 5, 1, 5, 1, 5, 1, 5]) == 0.0
    with pytest.raises(ValueError):
        sus_score([3] * 9)


def test_portuguese_detection():
    assert intake.looks_portuguese("Os alunos aprendem programação e a utilização de funções.")
    assert not intake.looks_portuguese("Students learn programming and the use of functions.")


def test_stream_reports_progress_then_the_result(tmp_path):
    c, _ = client(tmp_path, call("find_skills", topic="Linux"), AIMessage("Encontrei."))
    sid = c.post("/api/session", json={"email": "a@b.pt", "language": "pt",
                                       "expected": "programar"}).json()["session"]

    def stream(text):
        r = c.post("/api/message/stream", json={"session": sid, "text": text})
        assert r.status_code == 200, r.text
        return [json.loads(line) for line in r.text.splitlines() if line.strip()]

    stream("Programação"), stream(OUTCOMES)
    events = stream(CONTENTS)                        # the first suggestions
    stages = [e["progress"]["stage"] for e in events if "progress" in e]
    assert stages == ["reading", "searching", "checking"]
    assert events[-1]["done"]["stage"] == "chat"

    events = stream("há alguma coisa sobre linux?")
    stages = [e["progress"]["stage"] for e in events if "progress" in e]
    assert stages == ["thinking", "find_skills", "thinking"]
    assert events[1]["progress"] == {"stage": "find_skills", "topic": "Linux"}
    assert events[-1]["done"]["new"][-1]["content"] == "Encontrei."

    bad = c.post("/api/message/stream", json={"session": sid, "text": "  "})
    assert bad.status_code == 400


def test_suggested_messages_follow_the_conversation():
    from esco_agent.web.app import suggestions

    def k(i, status, origin="search 'x'"):
        return {"id": i, "label": i, "status": status, "origin": origin}

    state = {"queries": ["identify sources of information on genetics advances",
                         "use pointers", "manipulate DNA", "x"],
             "known": [k("a", "pending"), k("b", "pending")]}
    texts = [c["text"] for c in suggestions(state, "pt")]
    # a missed search phrase, as a search: short and specific first; long ones never
    assert texts == ["Porquê a número 1?", "Manter todas as sugeridas",
                     "Procurar na ESCO: «manipulate DNA»", "Procurar outro tema na ESCO: "]
    asked = {"procurar na esco: «manipulate dna»"}
    assert "Procurar na ESCO: «use pointers»" in [c["text"] for c in suggestions(state, "pt", asked)]
    state["events"] = [{"type": "search", "topic": "Manipulate DNA"}]      # searched since
    assert not any("manipulate" in c["text"] for c in suggestions(state, "pt"))
    state["events"] = []

    state["known"] = [k("a", "accepted"), k("b", "rejected")]
    assert [c["key"] for c in suggestions(state, "pt")][0] == "jobs"

    state["occupations"] = [{"label": "software developer"}]
    state["offered"] = {"d": {"id": "d", "label": "debug software", "origin": "required by 'software developer'"}}
    chips = suggestions(state, "en")
    assert [c["text"] for c in chips][:2] == ["What is missing for software developer?",
                                              'Suggest text to cover "debug software"']

    state["text_changes"] = [{"skill_id": "d", "status": "proposed"}]
    chips = suggestions(state, "en")
    assert chips[0] == {"key": "accept_change", "text": "I accept text change 1", "send": True}
    assert chips[1]["send"] is False                      # rewording: the teacher finishes it
    assert not any(c["key"] == "text" for c in chips)     # that skill already has a change
    assert len(chips) <= 5


def test_responses_carry_suggestions(tmp_path):
    c, _ = client(tmp_path)
    sid, r = begin(c)
    assert [x["key"] for x in r["chips"]][:2] == ["why", "keep_all"]
    r = c.post("/api/skill", json={"session": sid, "number": 1, "action": "accept"}).json()
    assert r["chips"][0]["text"] == "Porquê a número 2?"


def test_reply_tokens_are_streamed_from_the_conversation_model_only(tmp_path):
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessageChunk
    from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

    class Streaming(BaseChatModel):
        reply: str = "Mantive a número 1."

        @property
        def _llm_type(self):
            return "streaming-test"

        def _generate(self, messages, stop=None, run_manager=None, **kw):
            return ChatResult(generations=[ChatGeneration(message=AIMessage(self.reply))])

        async def _astream(self, messages, stop=None, run_manager=None, **kw):
            for word in self.reply.split(" "):
                chunk = ChatGenerationChunk(message=AIMessageChunk(content=word + " "))
                if run_manager:
                    await run_manager.on_llm_new_token(word + " ", chunk=chunk)
                yield chunk

        def bind_tools(self, tools, **kw):
            return self

    graph = build_assistant(FAKE_ESCO, Streaming(), Judge(), InMemorySaver())
    c = TestClient(create_app(Sessions(graph=graph, model="m", results_dir=str(tmp_path))))
    sid, _ = begin(c)
    r = c.post("/api/message/stream", json={"session": sid, "text": "olá"})
    events = [json.loads(line) for line in r.text.splitlines() if line.strip()]
    tokens = [e["token"] for e in events if "token" in e]
    assert "".join(tokens).strip() == "Mantive a número 1." and len(tokens) == 4
    assert events[-1]["done"]["new"][-1]["content"].strip() == "Mantive a número 1."



def test_suggestions_are_not_repeated_and_follow_searches_and_failed_drafts():
    from esco_agent.web.app import suggestions

    state = {"queries": ["x"],
             "known": [{"id": "a", "label": "a", "status": "pending", "origin": "x"},
                       {"id": "b", "label": "b", "status": "pending", "origin": "x"}],
             "events": []}
    texts = [c["text"] for c in suggestions(state, "pt")]
    assert "Porquê a número 1?" in texts
    texts = [c["text"] for c in suggestions(state, "pt", {"porquê a número 1?"})]
    assert "Porquê a número 2?" in texts and "Porquê a número 1?" not in texts

    # the latest search found skills the text doesn't show: offer text for the first
    state["events"] = [{"type": "search", "topic": "old"}, {"type": "search", "topic": "manipulate DNA"}]
    state["offered"] = {
        "o": {"id": "o", "label": "older one", "origin": "search 'old'"},
        "g": {"id": "g", "label": "genetic engineering", "origin": "search 'manipulate DNA'"}}
    texts = [c["text"] for c in suggestions(state, "pt")]
    assert "Sugere texto para cobrir «genetic engineering»" in texts
    assert not any("older one" in x for x in texts)

    # a draft that only described the course failed: offer to teach the skill instead
    state["text_changes"] = [{"skill_id": "g", "label": "genetic engineering",
                              "status": "failed", "mode": "show"}]
    chips = suggestions(state, "pt")
    assert [c["key"] for c in chips][:2] == ["teach", "reword_change"]
    assert chips[0]["text"] == "Sugere uma alteração para a UC passar a ensinar «genetic engineering»"
    assert "accept_change" not in [c["key"] for c in chips]
    assert not any(c["key"] == "text" for c in chips)     # that skill already has a change
    state["text_changes"][0]["mode"] = "teach"            # a failed teaching draft: reword only
    assert [c["key"] for c in suggestions(state, "en")][0] == "reword_change"



def test_decisions_from_the_finish_check_are_marked(tmp_path):
    c, sessions = client(tmp_path)
    sid, _ = begin(c)
    c.post("/api/skill", json={"session": sid, "number": 2, "action": "reject", "source": "finish"})
    events = saved(sessions, sid)["agent"]["events"]
    assert events[-1]["type"] == "rejected" and events[-1]["source"] == "finish"



def test_replies_are_checked_and_flagged_replies_logged(tmp_path):
    c, sessions = client(tmp_path,
                         AIMessage('The course says "students build compilers from scratch in Rust".'),
                         AIMessage('It says "Students write programs in C." (Learning outcomes)'))
    sid, _ = begin(c)
    r = c.post("/api/message", json={"session": sid, "text": "what does it teach?"}).json()
    assert r["new"][-1]["grounding_issues"][0]["kind"] == "quote_not_found"
    r = c.post("/api/message", json={"session": sid, "text": "and?"}).json()
    assert "grounding_issues" not in r["new"][-1]
    checks = [e for e in saved(sessions, sid)["agent"]["events"] if e["type"] == "grounding"]
    assert [e["ok"] for e in checks] == [False, True]


def test_daily_limit(tmp_path, monkeypatch):
    import esco_agent.web.app as webapp
    monkeypatch.setattr(webapp, "DAILY_LIMIT", 2)
    c, _ = client(tmp_path, AIMessage("one"))
    sid, _ = begin(c)                                    # the first suggestions use 1
    assert c.post("/api/message", json={"session": sid, "text": "a"}).status_code == 200
    r = c.post("/api/message", json={"session": sid, "text": "b"})
    assert r.status_code == 400 and "today's limit" in r.json()["detail"]


def test_page_can_be_framed_only_by_the_same_site(tmp_path):
    c, _ = client(tmp_path)
    for path in ("/", "/?embed=1", "/api/me"):
        r = c.get(path)
        assert r.headers["content-security-policy"] == "frame-ancestors 'self'"
        assert r.headers["x-frame-options"] == "SAMEORIGIN"
    page = c.get("/?embed=1").text
    assert 'id="embedbar"' in page and "body.embed header" in page


def test_course_name_on_the_first_screen_skips_that_question(tmp_path):
    c, sessions = client(tmp_path)
    r = c.post("/api/session", json={"email": "t@ua.pt", "language": "pt", "course_name": "  Programação  em C ",
                                     "expected": "programar em C"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["stage"] == "outcomes" and body["course_name"] == "Programação em C"
    assert "**Programação em C**" in body["chat"][0]["content"]
    r = c.post("/api/message", json={"session": body["session"], "text": OUTCOMES})
    assert r.json()["stage"] == "contents"
    assert saved(sessions, body["session"])["course_name"] == "Programação em C"
    bad = c.post("/api/session", json={"email": "t@ua.pt", "course_name": "line\nline\nline",
                                       "expected": "programar em C"})
    assert bad.status_code == 400


def test_sections_after_add_accepted_changes_at_the_end():
    from esco_agent.web.app import sections_after
    state = {"sections": {"Contents": "Pointers.\n", "Learning outcomes": "Write C."},
             "text_changes": [{"section": "Contents", "text": "Unit tests.", "status": "accepted"},
                              {"section": "Contents", "text": "Rejected.", "status": "rejected"},
                              {"section": "Contents", "text": "Test reports.", "status": "accepted"}]}
    assert sections_after(state) == {"Contents": "Pointers.\nUnit tests.\nTest reports."}


PAGE = """Programming
Objectives
The course aims to develop students' ability to understand, design and implement programmes in C language.
Learning Outcomes
The course aims to develop students' ability to understand, design and implement programmes in C language.
Requisites
Mathematics knowledge at the level of the 12th grade of secondary education.
Assessment
TP component:
- 4 quizzes taken throughout the semester
Teaching methods
The theoretical-practical classes are heavily based on practical exercises.
Contents
The C Programming course introduces the fundamentals of structured programming, integrating Unix/Linux.
Recommended bibliography
Kernighan, Brian W., e Dennis M. Ritchie. 2014. The C Programming Language."""


def test_a_whole_page_is_split_by_its_headings():
    from esco_agent.web.intake import split_page
    kept, dropped = split_page(PAGE)
    assert list(kept) == ["Objectives", "Learning outcomes", "Contents"]
    assert kept["Contents"].startswith("The C Programming course")
    assert dropped == ["requisites", "assessment", "teaching methods", "bibliography"]
    assert split_page("Students write C programs.\nThey use make.") is None
    # a qualified heading of a left-out section ends the section above it (20 Nov genetics
    # page: the bibliography had been judged and drafted against as part of Contents)
    kept, dropped = split_page("Objectives\nTo manipulate DNA.\nContents\nTHEME I. PCR\n"
                               "TOPIC V. Mutagenesis - Ames Test\nAbbreviated bibliography\n"
                               "Palladino, M.A. (2015) Concepts of Genetics. Pearson")
    assert kept["Contents"] == "THEME I. PCR\nTOPIC V. Mutagenesis - Ames Test"
    assert dropped == ["bibliography"]
    # but a long line ending in such a word is text, not a heading
    kept, _ = split_page("Objectives\nStudents learn to read the scientific bibliography\n"
                         "Contents\nPCR")
    assert "scientific bibliography" in kept["Objectives"]

    c, sessions = client(tmp_path := __import__("pathlib").Path(__import__("tempfile").mkdtemp()))
    sid = c.post("/api/session", json={"email": "a@b.pt", "language": "pt", "course_name": "Programação",
                                       "expected": "programar em C"}).json()["session"]
    r = c.post("/api/message", json={"session": sid, "text": PAGE}).json()
    assert r["stage"] == "chat"
    used = next(m["content"] for m in r["new"] if m["role"] == "assistant")
    assert used.startswith("Usei objetivos, resultados de aprendizagem e conteúdos da página")
    assert "deixei de fora requisitos, avaliação, métodos de ensino e bibliografia" in used
    assert list(sessions.sessions[sid]["sections"]) == ["Objectives", "Learning outcomes", "Contents"]



def test_removing_from_the_list_needs_a_reason(tmp_path):
    c, sessions = client(tmp_path)
    sid = c.post("/api/session", json={"email": "a@b.pt", "language": "pt",
                                       "expected": "programar em C"}).json()["session"]
    for text in ("Programação", OUTCOMES, "não tenho"):
        c.post("/api/message", json={"session": sid, "text": text})
    r = c.post("/api/skill", json={"session": sid, "number": 1, "action": "reject"})
    assert r.status_code == 400 and "porque" in r.json()["detail"]
    r = c.post("/api/skill", json={"session": sid, "number": 1, "action": "reject",
                                   "reason": "not_taught"}).json()
    assert r["skills"][0]["reject_reason"] == "not_taught"
    assert r["new"][-1]["content"].endswith("removida (a UC não ensina isto)")
    r = c.post("/api/skill", json={"session": sid, "number": 2, "action": "reject",
                                   "source": "finish"}).json()
    assert r["skills"][1]["reject_reason"] == "not_specified"
    r = c.post("/api/skill", json={"session": sid, "number": 1, "action": "accept"}).json()
    assert r["skills"][0]["reject_reason"] is None
    r = c.post("/api/skill", json={"session": sid, "number": 1, "action": "reject",
                                   "reason": "misleading_name"}).json()
    assert r["skills"][0]["reject_reason"] == "misleading_name"
    assert r["new"][-1]["content"].endswith("removida (o nome engana)")
    r = c.post("/api/skill", json={"session": sid, "number": 1, "action": "reject",
                                   "reason": "other", "note": " "})
    assert r.status_code == 400 and "poucas palavras" in r.json()["detail"]
    r = c.post("/api/skill", json={"session": sid, "number": 1, "action": "reject",
                                   "reason": "other", "note": "demasiado genérica"}).json()
    assert r["skills"][0]["reject_note"] == "demasiado genérica"
    assert r["new"][-1]["content"].endswith("removida (outro motivo: demasiado genérica)")
    r = c.post("/api/skill", json={"session": sid, "number": 1, "action": "reject",
                                   "reason": "not_in_text", "note": "ignored"}).json()
    assert r["skills"][0]["reject_note"] is None


def test_session_file_records_familiarity_version_usage_and_opened_cards(tmp_path):
    reply = AIMessage("Mantive a 2.", usage_metadata={
        "input_tokens": 1000, "output_tokens": 50, "total_tokens": 1050,
        "input_token_details": {"cache_read": 800}})
    c, sessions = client(tmp_path, call("update_skill_list", accept=[2]), reply)
    bad = c.post("/api/session", json={"email": "a@b.pt", "expected": "programar em C",
                                       "esco_familiarity": "expert"})
    assert bad.status_code == 400 and "ESCO" in bad.json()["detail"]
    r = c.post("/api/session", json={"email": "a@b.pt", "language": "pt", "course_name": "Programação",
                                     "expected": "programar em C", "esco_familiarity": "medium"})
    sid = r.json()["session"]
    for text in [OUTCOMES, CONTENTS, "manter a 2"]:
        assert c.post("/api/message", json={"session": sid, "text": text}).status_code == 200

    assert c.post("/api/ui", json={"session": sid, "event": "open", "kind": "skill",
                                   "number": 1}).status_code == 200
    assert c.post("/api/ui", json={"session": sid, "event": "open", "kind": "script"}).status_code == 400
    data = saved(sessions, sid)
    assert data["esco_familiarity"] == "medium"
    assert data["ui"] == [{**data["ui"][0], "event": "open", "kind": "skill", "number": 1}]
    assert data["agent"]["skills"]            # saving the card event kept the agent's state
    assert len(data["app_version"]["code"]) == 12
    u = data["usage"]
    # first turn: query step + judge (fakes, no usage); chat turn: two agent calls, one with usage
    assert u["calls"] >= 4 and u["input_tokens"] == 1000 and u["output_tokens"] == 50
    assert u["cached_input_tokens"] == 800


def test_familiarity_is_optional_for_the_api():
    """The page requires it; the API accepts older clients and stores nothing."""
    from esco_agent.web.app import FAMILIARITY
    assert FAMILIARITY == ("none", "low", "medium", "high")


def test_accepted_text_is_shown_and_kept_where_it_was_placed():
    from esco_agent.web.app import panel, sections_after
    state = {"sections": {"Contents": "THEME I. PCR\nTOPIC V. Ames\nnotes"}, "known": [],
             "text_changes": [{"skill_id": "g", "label": "genetic engineering", "mode": "teach",
                               "section": "Contents", "after": "TOPIC V. Ames",
                               "text": "TOPIC VI. Cloning", "verified": True, "status": "accepted"}]}
    assert panel(state)["text_changes"][0]["around"] == ("THEME I. PCR\nTOPIC V. Ames", "notes")
    assert sections_after(state)["Contents"] == "THEME I. PCR\nTOPIC V. Ames\nTOPIC VI. Cloning\nnotes"
    assert state["sections"]["Contents"] == "THEME I. PCR\nTOPIC V. Ames\nnotes"   # original kept
