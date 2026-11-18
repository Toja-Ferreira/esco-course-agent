"""The skill assistant graph, with scripted LLMs and fake ESCO lookups. No data or API."""

import asyncio
import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from esco_agent.assistant import (Esco, build_assistant, export, render_state,
                                  save_session, session_path)

SECTIONS = {"Learning outcomes": "Students write programs in C. They test their code.",
            "Contents": "- pointers and dynamic memory\n- the Linux command line"}


def skill(i, label, kind="knowledge", description=""):
    return {"id": i, "label": label, "type": kind, "score": 0.9, "description": description}


ESCO = {s["id"]: s for s in [
    skill("prog", "computer programming", description="The techniques of software development."),
    skill("test", "perform software unit testing", "skill", "Test single units of source code."),
    skill("cpp", "C++", description="The C++ language."),
    skill("os", "operating systems", description="Features of operating systems such as Linux."),
    skill("kali", "Kali Linux", description="A penetration testing tool."),
]}
OCCUPATIONS = [
    {"id": "dev", "label": "software developer", "isco_group": "2512", "score": 0.5,
     "top_matching_skills": [{"id": "prog", "label": "computer programming", "relation": "essential"}]},
    {"id": "lect", "label": "computer science lecturer", "isco_group": "2310", "score": 0.3,
     "top_matching_skills": []},
]
ESSENTIAL = {"dev": [("prog", "computer programming"), ("test", "perform software unit testing"),
                     ("os", "operating systems"), ("dbg", "debug software")]}
searches = []


async def link(text, k):
    if "linux" in text.lower() and "\n" not in text:
        return [ESCO["os"], ESCO["kali"], ESCO["prog"]]
    return [ESCO["prog"], ESCO["test"], ESCO["cpp"]]


async def search_occupations(skill_ids, isco_prefix, k):
    searches.append((skill_ids, isco_prefix))
    return [o for o in OCCUPATIONS if not isco_prefix or o["isco_group"].startswith(isco_prefix)]


async def get_occupation_skills(occupation_id):
    return {"id": occupation_id, "label": "software developer", "isco_group": "2512",
            "description": "", "optional": [],
            "essential": [{"id": i, "label": l, "type": "skill"} for i, l in ESSENTIAL[occupation_id]]}


FAKE_ESCO = Esco(link, search_occupations, get_occupation_skills)


class Judge:
    """Query step: one query. Judge steps: verdict by skill name."""
    COVERED = {"computer programming": [1], "perform software unit testing": [2],
               "operating systems": [4]}

    drafts = []     # replies to the text-change prompt, in order

    async def ainvoke(self, messages):
        prompt = messages[0][1]
        if '{"queries"' in prompt:
            return AIMessage(json.dumps({"queries": ["test code"]}))
        if '"section": "..."' in prompt:          # a text-change draft
            Judge.prompts.append(prompt)
            return AIMessage(Judge.drafts.pop(0))
        if '{"reason"' in prompt:                  # why a draft failed the check
            Judge.prompts.append(prompt)
            return AIMessage(json.dumps({"reason": "It does not mention a debugger."}))
        covered = dict(self.COVERED)
        if "gdb" in prompt:            # the revised course mentions a debugger
            line = next(l for l in prompt.splitlines() if "gdb" in l)
            covered["debug software"] = [int(line.split(".")[0])]
        block = prompt.split("For EVERY")[0].rstrip().rsplit("\n\n", 1)[-1]
        out = []
        for line in block.splitlines():
            head, _, rest = line.partition(". ")
            if head.isdigit():
                sents = covered.get(rest.split(":")[0], [])
                out.append({"skill": int(head), "verdict": "covered" if sents else "missing",
                            "sentences": sents})
        return AIMessage(json.dumps({"judgements": out}))


Judge.prompts = []


class Agent:
    """Plays the conversation LLM: each reply is scripted, prompts are recorded."""
    def __init__(self, *replies):
        self.replies, self.systems = list(replies), []

    def bind_tools(self, tools):
        self.tools = [t["function"]["name"] for t in tools]
        return self

    async def ainvoke(self, messages):
        self.systems.append(messages[0].content)
        return self.replies.pop(0)


def call(name, **args):
    return AIMessage("", tool_calls=[{"name": name, "args": args, "id": name, "type": "tool_call"}])


def run(agent, turns, language="en"):
    async def go():
        graph = build_assistant(FAKE_ESCO, agent, Judge(), InMemorySaver())
        cfg = {"configurable": {"thread_id": "t"}}
        state = await graph.ainvoke({"sections": SECTIONS, "language": language,
                                     "messages": [HumanMessage("course")]}, cfg)
        states = [state]
        for q in turns:
            states.append(await graph.ainvoke({"messages": [HumanMessage(q)]}, cfg))
        return states
    return asyncio.run(go())


def last_result(state):
    return json.loads(next(m.content for m in reversed(state["messages"])
                           if isinstance(m, ToolMessage)))


def test_first_turn_suggests_supported_skills_with_evidence():
    (s,) = run(Agent(), [])
    assert [k["label"] for k in s["known"]] == ["computer programming",
                                                "perform software unit testing"]  # C++ judged out
    text = s["messages"][-1].content
    assert '1. **computer programming** - "Students write programs in C." (Learning outcomes)' in text
    assert s["known"][1]["origin"] == "search 'test code'"
    assert s["events"][0]["type"] == "suggested" and s["events"][0]["judge_rejected"] == ["cpp"]


def test_first_turn_in_portuguese():
    (s,) = run(Agent(), [], language="pt")
    assert s["messages"][-1].content.startswith("Li a unidade curricular (Learning outcomes, Contents)")


def test_accept_reject_search_and_add_a_skill_not_in_the_text():
    agent = Agent(
        call("update_skill_list", accept=[1], reject=[2]), AIMessage("Kept 1."),
        call("find_skills", topic="Linux"), AIMessage("Found 3; Kali Linux is not in the text."),
        call("update_skill_list", accept=[3], add_not_in_text=["kali linux"]), AIMessage("Added."),
        AIMessage("Because sentence X."))
    _, s1, s2, s3, s4 = run(agent, ["keep 1, drop 2", "anything on Linux?",
                                    "add both", "why number 3?"])
    assert agent.tools == ["update_skill_list", "find_skills", "find_occupations",
                           "check_occupation_gap", "suggest_text_change", "decide_text_change"]
    assert [k["status"] for k in s1["known"]] == ["accepted", "rejected"]

    found = last_result(s2)
    assert found["supported_by_course_text"] == [{
        "number": 3, "label": "operating systems",
        "definition": "Features of operating systems such as Linux.",
        "evidence": [{"section": "Contents", "text": "the Linux command line"}]}]
    # not in the text: named, but no number until the teacher asks for it
    assert found["not_in_course_text"] == [{"label": "Kali Linux",
                                            "definition": "A penetration testing tool."}]
    assert found["already_known"] == [{"number": 1, "label": "computer programming",
                                       "status": "accepted"}]
    assert len(s2["known"]) == 3 and list(s2["offered"]) == ["kali"]

    assert [(k["label"], k["status"], k["in_text"]) for k in s3["known"]] == [
        ("computer programming", "accepted", True),
        ("perform software unit testing", "rejected", True),
        ("operating systems", "accepted", True),
        ("Kali Linux", "accepted", False)]
    assert s3["offered"] == {}

    # the LLM sees the current list, evidence and origin on every turn
    last = agent.systems[-1]
    assert "ON THE LIST" in last and "4. Kali Linux [knowledge]" in last and "NOT IN TEXT" in last
    assert '"the Linux command line" (Contents)' in last and "found by: search 'Linux'" in last
    assert "2. perform software unit testing" in last.split("REJECTED BY THE TEACHER")[1]

    out = export(s4)
    assert [(x["number"], x["label"], x["in_text"]) for x in out["skills"]] == [
        (1, "computer programming", True), (3, "operating systems", True), (4, "Kali Linux", False)]
    assert out["skills"][0]["uri"] == "http://data.europa.eu/esco/skill/prog"
    assert [e["type"] for e in out["events"]] == ["suggested", "accepted", "rejected", "search",
                                                  "accepted", "accepted"]
    assert out["events"][-1] == {**out["events"][-1], "skill": "kali", "was": "offered",
                                 "in_text": False}
    assert [m["role"] for m in out["transcript"]][:3] == ["human", "ai", "human"]
    assert out["transcript"][2]["content"] == "keep 1, drop 2"
    assert out["transcript"][3]["tool_calls"] == [{"name": "update_skill_list",
                                                   "args": {"accept": [1], "reject": [2]}}]


def test_numbers_stay_consecutive_across_searches():
    agent = Agent(call("find_skills", topic="Linux"), AIMessage("."),
                  call("find_skills", topic="Linux tools"), AIMessage("."))
    _, _, s = run(agent, ["linux?", "again?"])
    assert [k["label"] for k in s["known"]] == ["computer programming",
                                                "perform software unit testing",
                                                "operating systems"]
    again = last_result(s)       # second search: nothing new is numbered or re-judged
    assert again["supported_by_course_text"] == []
    assert [x["number"] for x in again["already_known"]] == [3, 1]
    assert [x["label"] for x in again["not_in_course_text"]] == ["Kali Linux"]


def test_occupations_come_from_the_accepted_list_and_gap_adds_numbered_skills():
    agent = Agent(
        call("find_occupations"), AIMessage("List is empty."),
        call("update_skill_list", accept=[1]), AIMessage("Kept."),
        call("find_occupations", isco_prefix="25"), AIMessage("1. software developer"),
        call("check_occupation_gap", occupation=1), AIMessage("Gap."),
        call("update_skill_list", accept=[3], add_not_in_text=["debug software"]), AIMessage("Ok."))
    searches.clear()
    _, s1, s2, s3, s4, s5 = run(agent, ["jobs?", "keep 1", "ict jobs?", "missing for 1?", "add"])
    assert "list is empty" in last_result(s1)["error"]   # and no search was made, see below

    occ = last_result(s3)
    assert searches == [(["prog"], "25")]
    assert occ["occupations"] == [{"rank": 1, "label": "software developer", "isco_group": "2512",
                                   "score": 0.5, "matched_on": ["computer programming"]}]
    assert "LATEST OCCUPATIONS (rank. name): 1. software developer" in agent.systems[-1]

    gap = last_result(s4)
    assert gap["occupation"] == "software developer" and gap["essential_skills"] == 4
    assert gap["on_the_list"] == ["1. computer programming"]
    # already suggested (2) keeps its number; newly found in the text gets the next one (3)
    assert [(x["number"], x["label"]) for x in gap["in_course_text_but_not_on_list"]] == [
        (2, "perform software unit testing"), (3, "operating systems")]
    assert gap["in_course_text_but_not_on_list"][1]["evidence"] == [
        {"section": "Contents", "text": "the Linux command line"}]
    assert gap["not_in_course_text"] == ["debug software"]
    assert s4["known"][2]["origin"] == "required by 'software developer'"
    # the page's occupation card, worked out from the list as it is now
    from esco_agent.web.app import panel
    shown = panel(s4)["occupations"]
    assert shown[0]["gap"] == {
        "essential": 4, "on_list": 1,
        "in_text": [{"number": 2, "label": "perform software unit testing"},
                    {"number": 3, "label": "operating systems"}],
        "not_in_text": ["debug software"]}
    assert shown[0]["url"].startswith("https://esco.ec.europa.eu/en/classification/occupation?uri=")
    # after keeping 3 and adding "debug software" (s5), the card updates without a new check
    assert panel(s5)["occupations"][0]["gap"] == {
        "essential": 4, "on_list": 3,
        "in_text": [{"number": 2, "label": "perform software unit testing"}], "not_in_text": []}

    assert [(k["label"], k["status"], k["in_text"]) for k in s5["known"]][2:] == [
        ("operating systems", "accepted", True), ("debug software", "accepted", False)]
    assert [e["type"] for e in s5["events"]] == ["suggested", "accepted", "occupations", "gap",
                                                 "accepted", "accepted"]


def gap_then(*replies):
    """Conversation up to a gap check that leaves 'debug software' not in the text."""
    return Agent(call("update_skill_list", accept=[1]), AIMessage("."),
                 call("find_occupations"), AIMessage("."),
                 call("check_occupation_gap", occupation=1), AIMessage("."), *replies)


def test_show_mode_draft_is_verified_and_accepting_it_adds_the_skill():
    Judge.drafts[:] = [json.dumps({"section": "contents",
                                   "text": "Students find and fix faults in their programs with gdb."})]
    Judge.prompts.clear()
    agent = gap_then(call("suggest_text_change", skill="Debug Software", mode="show"),
                     AIMessage("Suggestion."),
                     call("decide_text_change", change=1, decision="accept"), AIMessage("Done."))
    *_, s4, s5 = run(agent, ["keep 1", "jobs?", "gap?", "it already covers debugging", "ok"])

    out = last_result(s4)
    assert out == {"change": 1, "mode": "show", "target_skill": "debug software",
                   "add_to_section": "Contents",
                   "suggested_text": "Students find and fix faults in their programs with gdb.",
                   "check_passed": True, "status": "proposed", "check": out["check"]}
    assert "already develops" in Judge.prompts[0] and "Skill: debug software" in Judge.prompts[0]
    assert s4["sections"] == SECTIONS                       # the page itself is not changed
    assert any("page text for 'debug software'" in x and "[check passed; proposed]" in x
               for x in agent.systems)

    assert last_result(s5)["list_now"][-1] == "4. debug software (NOT IN TEXT)"
    saved = export(s5)
    assert saved["text_changes"] == [{
        "number": 1, "skill_id": "dbg", "label": "debug software", "mode": "show",
        "add_to_course": [], "section": "Contents",
        "text": "Students find and fix faults in their programs with gdb.", "verified": True,
        "reason": "", "attempts": 1, "edited_by_teacher": False, "status": "accepted"}]
    assert saved["skills"][-1]["label"] == "debug software" and saved["skills"][-1]["text_change"] == 1
    assert [e["type"] for e in saved["events"]][-3:] == [
        "text_change_proposed", "text_change_accepted", "accepted"]


def test_teach_mode_proposes_what_to_add_and_the_page_text():
    Judge.drafts[:] = [json.dumps({
        "add_to_course": ["a lab on using the gdb debugger", "tracing faults in pointer code"],
        "section": "Contents", "text": "Students learn to locate and fix faults using gdb."})]
    Judge.prompts.clear()
    agent = gap_then(call("suggest_text_change", skill="debug software", mode="teach"),
                     AIMessage("Change to the course."))
    *_, s = run(agent, ["keep 1", "jobs?", "gap?", "I want it to teach debugging"])
    out = last_result(s)
    assert out["mode"] == "teach" and out["check_passed"] is True
    assert out["add_to_course"] == ["a lab on using the gdb debugger", "tracing faults in pointer code"]
    assert "START developing" in Judge.prompts[0]
    assert "course change for 'debug software'" in render_state(
        s["known"], s["occupations"], s["text_changes"])


def test_failed_draft_gets_a_reason_one_redraft_and_cannot_be_accepted_as_is():
    Judge.drafts[:] = [json.dumps({"section": "Contents", "text": "Students debug software."}),
                       json.dumps({"section": "Contents", "text": "Students correct their code."})]
    Judge.prompts.clear()
    agent = gap_then(
        call("suggest_text_change", skill="debug software", mode="show", section="Contents"),
        AIMessage("It failed because..."),
        call("decide_text_change", change=1, decision="accept"), AIMessage("."),
        call("decide_text_change", change=1, decision="accept",
             edited_text="Students fix faults in their programs using gdb."), AIMessage("."))
    *_, s4, s5, s6 = run(agent, ["keep 1", "jobs?", "gap?", "cover debugging", "accept it",
                                 "accept with my words"])
    out = last_result(s4)
    assert out["check_passed"] is False and out["status"] == "failed"
    assert out["reason"] == "It does not mention a debugger." and "FAILED" in out["check"]
    assert out["suggested_text"] == "Students correct their code."     # the redraft
    drafts = [p for p in Judge.prompts if '"section": "..."' in p]
    assert len(drafts) == 2 and "Its reason: It does not mention a debugger." in drafts[1]
    assert "fine for the check to fail" in drafts[1]                   # show mode stays honest
    assert "can only be accepted in the teacher's own wording" in last_result(s5)["error"]

    c = s6["text_changes"][0]
    assert (c["status"], c["edited_by_teacher"], c["verified"]) == ("accepted", True, True)
    assert c["text"] == "Students fix faults in their programs using gdb."


def test_teacher_rewording_is_checked_again_and_rejection_leaves_the_list_alone():
    Judge.drafts[:] = [json.dumps({"section": "Contents", "text": "Students use gdb."})]
    agent = gap_then(
        call("suggest_text_change", skill="debug software", mode="show"), AIMessage("."),
        call("decide_text_change", change=1, decision="reject",
             edited_text="Students learn about tools."), AIMessage("."))
    *_, s = run(agent, ["keep 1", "jobs?", "gap?", "cover debugging", "no, and I'd say it like this"])
    c = s["text_changes"][0]
    assert (c["status"], c["text"], c["edited_by_teacher"], c["verified"]) == (
        "rejected", "Students learn about tools.", True, False)
    assert c["reason"] == "It does not mention a debugger."
    assert "debug software" not in [k["label"] for k in s["known"] if k["status"] == "accepted"]


def test_text_change_needs_a_mode_and_a_skill_missing_from_the_text():
    agent = Agent(call("suggest_text_change", skill="computer programming", mode="show"), AIMessage("."),
                  call("suggest_text_change", skill="quantum computing", mode="show"), AIMessage("."),
                  call("decide_text_change", change=1, decision="accept"), AIMessage("."))
    *_, s = run(agent, ["a", "b", "c"])
    errors = [json.loads(m.content)["error"] for m in s["messages"] if isinstance(m, ToolMessage)]
    assert "already shows" in errors[0] and "not a skill that was found missing" in errors[1]
    assert errors[2] == "there are no suggested text changes yet"
    agent = gap_then(call("suggest_text_change", skill="debug software"), AIMessage("."))
    *_, s = run(agent, ["keep 1", "jobs?", "gap?", "cover debugging"])
    assert "mode must be 'show'" in last_result(s)["error"]
    assert s["text_changes"] == []


def test_mistakes_are_reported_to_the_llm_and_change_nothing():
    agent = Agent(call("update_skill_list", accept=[1, 9]), AIMessage("."),
                  call("update_skill_list", accept=[1], reject=[1]), AIMessage("."),
                  call("update_skill_list", accept=[1], add_not_in_text=["made up skill"]), AIMessage("."),
                  call("check_occupation_gap", occupation=1), AIMessage("."),
                  call("find_skills", subject="x"), AIMessage("."))
    states = run(agent, ["a", "b", "c", "d", "e"])
    errors = [json.loads(m.content)["error"] for m in states[-1]["messages"]
              if isinstance(m, ToolMessage)]
    assert "no skill with number [9]" in errors[0]
    assert "both accept and reject" in errors[1]
    assert "did not match" in errors[2]
    assert errors[3] == "call find_occupations first"
    assert "bad arguments for find_skills" in errors[4]
    assert all(k["status"] == "pending" for k in states[-1]["known"])
    assert len(states[-1]["events"]) == 1


def test_two_tool_calls_in_one_turn_both_apply():
    both = AIMessage("", tool_calls=[
        {"name": "update_skill_list", "args": {"accept": [1]}, "id": "a", "type": "tool_call"},
        {"name": "update_skill_list", "args": {"reject": [2]}, "id": "b", "type": "tool_call"}])
    _, s = run(Agent(both, AIMessage("Done.")), ["keep 1, drop 2"])
    assert [k["status"] for k in s["known"]] == ["accepted", "rejected"]


def test_session_is_saved_with_course_list_log_and_conversation(tmp_path):
    _, s = run(Agent(call("update_skill_list", accept=[1]), AIMessage("Kept.")), ["keep 1"])
    path = session_path("courses/c-programming.json", folder=str(tmp_path / "sessions"))
    assert path.endswith("_c-programming.json")
    save_session(path, s, "deepseek-x", "courses/c-programming.json")
    saved = json.load(open(path, encoding="utf-8"))
    assert saved["model"] == "deepseek-x" and saved["course"] == SECTIONS
    assert [x["label"] for x in saved["skills"]] == ["computer programming"]
    assert saved["transcript"][-1] == {"role": "ai", "content": "Kept."}
    assert saved["events"][-1]["type"] == "accepted"


def test_render_state_empty():
    assert render_state([]) == "No skills yet."
