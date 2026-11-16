"""The skill assistant graph, with scripted LLMs and a fake linker. No data or API."""

import asyncio
import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from esco_agent.assistant import build_assistant, export, render_state

SECTIONS = {"Learning outcomes": "Students write programs in C. They test their code.",
            "Contents": "- pointers and dynamic memory\n- the Linux command line"}
ESCO = {
    "prog": {"id": "prog", "label": "computer programming", "type": "knowledge",
             "score": 0.9, "description": "The techniques of software development."},
    "test": {"id": "test", "label": "perform software unit testing", "type": "skill",
             "score": 0.8, "description": "Test single units of source code."},
    "cpp": {"id": "cpp", "label": "C++", "type": "knowledge", "score": 0.7,
            "description": "The C++ language."},
    "os": {"id": "os", "label": "operating systems", "type": "knowledge", "score": 0.9,
           "description": "Features of operating systems such as Linux."},
    "kali": {"id": "kali", "label": "Kali Linux", "type": "knowledge", "score": 0.6,
             "description": "A penetration testing tool."},
}


async def link(text, k):
    if "linux" in text.lower() and "\n" not in text:
        return [ESCO["os"], ESCO["kali"], ESCO["prog"]]
    return [ESCO["prog"], ESCO["test"], ESCO["cpp"]]


class Judge:
    """Query step: one query. Judge step: verdict by label."""
    COVERED = {"computer programming": [1], "perform software unit testing": [2],
               "operating systems": [4]}

    async def ainvoke(self, messages):
        prompt = messages[0][1]
        if '{"queries"' in prompt:
            return AIMessage(json.dumps({"queries": ["test code"]}))
        lines = prompt.split("definition follows")[1].split("For EVERY")[0].splitlines()
        out = []
        for line in lines:
            head, _, rest = line.partition(". ")
            if head.isdigit():
                sents = self.COVERED.get(rest.split(":")[0], [])
                out.append({"skill": int(head), "verdict": "covered" if sents else "missing",
                            "sentences": sents})
        return AIMessage(json.dumps({"judgements": out}))


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
        graph = build_assistant(link, agent, Judge(), InMemorySaver())
        cfg = {"configurable": {"thread_id": "t"}}
        state = await graph.ainvoke({"sections": SECTIONS, "language": language,
                                     "messages": [HumanMessage("course")]}, cfg)
        states = [state]
        for q in turns:
            states.append(await graph.ainvoke({"messages": [HumanMessage(q)]}, cfg))
        return states
    return asyncio.run(go())


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
        call("find_skills", topic="Linux"), AIMessage("Found 3; 4 is not in the text."),
        call("update_skill_list", accept=[3, 4]), AIMessage("Added."),
        AIMessage("Because sentence X."))
    _, s1, s2, s3, s4 = run(agent, ["keep 1, drop 2", "anything on Linux?",
                                    "add both", "why number 3?"])
    assert agent.tools == ["update_skill_list", "find_skills"]
    assert [k["status"] for k in s1["known"]] == ["accepted", "rejected"]

    found = json.loads(next(m.content for m in reversed(s2["messages"])
                            if isinstance(m, ToolMessage)))
    assert found["supported_by_course_text"][0] == {
        "number": 3, "label": "operating systems",
        "definition": "Features of operating systems such as Linux.",
        "evidence": [{"section": "Contents", "text": "the Linux command line"}]}
    assert [x["number"] for x in found["not_in_course_text"]] == [4]
    assert found["already_known"] == [{"number": 1, "label": "computer programming",
                                       "status": "accepted"}]

    assert [(k["label"], k["status"], k["in_text"]) for k in s3["known"]] == [
        ("computer programming", "accepted", True),
        ("perform software unit testing", "rejected", True),
        ("operating systems", "accepted", True),
        ("Kali Linux", "accepted", False)]

    # the LLM sees the current list, evidence and origin on every turn
    last = agent.systems[-1]
    assert "ON THE LIST" in last and "4. Kali Linux [knowledge]" in last and "NOT IN TEXT" in last
    assert '"the Linux command line" (Contents)' in last and "found by: search 'Linux'" in last
    assert "2. perform software unit testing" in last.split("REJECTED BY THE TEACHER")[1]

    out = export(s4)
    assert [(x["label"], x["in_text"]) for x in out["skills"]] == [
        ("computer programming", True), ("operating systems", True), ("Kali Linux", False)]
    assert out["skills"][0]["uri"] == "http://data.europa.eu/esco/skill/prog"
    assert [e["type"] for e in out["events"]] == ["suggested", "accepted", "rejected", "search",
                                                  "accepted", "accepted"]
    assert out["events"][-1] == {**out["events"][-1], "skill": "kali", "in_text": False}


def test_bad_numbers_are_reported_to_the_llm_and_change_nothing():
    agent = Agent(call("update_skill_list", accept=[1, 9]), AIMessage("Sorry."),
                  call("update_skill_list", accept=[1], reject=[1]), AIMessage("Sorry."))
    _, s1, s2 = run(agent, ["keep 9", "keep and drop 1"])
    errors = [json.loads(m.content)["error"] for m in s2["messages"] if isinstance(m, ToolMessage)]
    assert "no skill with number [9]" in errors[0] and "both accept and reject" in errors[1]
    assert all(k["status"] == "pending" for k in s2["known"]) and len(s2["events"]) == 1


def test_two_tool_calls_in_one_turn_both_apply():
    both = AIMessage("", tool_calls=[
        {"name": "update_skill_list", "args": {"accept": [1]}, "id": "a", "type": "tool_call"},
        {"name": "update_skill_list", "args": {"reject": [2]}, "id": "b", "type": "tool_call"}])
    _, s = run(Agent(both, AIMessage("Done.")), ["keep 1, drop 2"])
    assert [k["status"] for k in s["known"]] == ["accepted", "rejected"]


def test_render_state_empty():
    assert render_state([]) == "No skills yet."