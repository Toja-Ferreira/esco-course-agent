"""The skill assistant: a teacher builds a course's ESCO skill list in conversation.

    START --(first message)--> first_turn --> END
    START --(follow-ups)-----> agent <--> tools --> END

first_turn is fixed, not LLM-driven: it runs the suggestion pipeline
(esco_agent.suggest) and shows the supported skills with their evidence.

Every skill the conversation knows about lives in state["known"] and has a
number (its position + 1). The teacher and the LLM refer to skills by number,
so the LLM never has to copy an ESCO id. A skill's status is one of:
    pending     suggested, the teacher hasn't decided
    accepted    on the course's skill list
    rejected    the teacher said no
A skill also has in_text: True if the judge found a supporting sentence.
The teacher can accept a skill that is not in the text (their decision); it is
exported with in_text false, which is the "taught but not on the page" case.

The LLM has two tools, executed by the tools node one call at a time so that
two calls in the same turn can't overwrite each other's changes:
    update_skill_list(accept, reject)   change statuses, by number
    find_skills(topic)                  "anything on X?": search ESCO, then
                                        judge the results against the course

state["events"] records what was suggested, accepted, rejected and searched,
with timestamps. It is the log the teacher test will be analysed from.

Try it:
    python -m esco_agent.assistant --course course.json --lang pt
"""

from __future__ import annotations

import json
import operator
from datetime import datetime, timezone
from typing import Annotated

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, MessagesState, StateGraph

from esco_agent.suggest import (Sections, _shorten, course_text, gather_candidates,
                                propose_queries, suggest_skills)

SKILL_URI = "http://data.europa.eu/esco/skill/"
FIND_K = 8   # ESCO skills fetched per "anything on X?" search

TOOLS = [
    {"type": "function", "function": {
        "name": "update_skill_list",
        "description": (
            "Change the course's skill list. Call this whenever the teacher wants "
            "skills kept, added, removed or rejected; saying it is done is not enough. "
            "Use the skill numbers from CURRENT STATE or from find_skills."),
        "parameters": {"type": "object", "properties": {
            "accept": {"type": "array", "items": {"type": "integer"},
                       "description": "Numbers of skills to put on the list."},
            "reject": {"type": "array", "items": {"type": "integer"},
                       "description": "Numbers of skills to reject or take off the list."},
        }}}},
    {"type": "function", "function": {
        "name": "find_skills",
        "description": (
            "Search ESCO for skills about a topic and check each against the course "
            "text. Use when the teacher asks 'anything on X?' or expects a skill "
            "that isn't shown. Returns numbered skills: those the course text "
            "supports (with the sentence) and those it doesn't."),
        "parameters": {"type": "object", "properties": {
            "topic": {"type": "string",
                      "description": "A short phrase in English, like an ESCO skill "
                                     "name: 'operating systems', 'work in teams'."},
        }, "required": ["topic"]}}},
]

LANGUAGES = {"en": "English", "pt": "European Portuguese"}

SYSTEM = """You help a teacher choose the ESCO skills that describe their course. \
The course text is the first message of this conversation.

Reply in {language}. Keep ESCO skill names exactly as given, in English.

Rules:
- Every skill has a number. Only mention skills that appear in CURRENT STATE below \
or in a tool result of this conversation. Never invent ESCO skills or recall them \
from memory.
- When the teacher wants skills kept, added, removed or rejected, call \
update_skill_list with their numbers.
- When the teacher asks whether there is anything on a topic, or expects a skill \
you haven't shown, call find_skills. If it finds nothing suitable, say so; ESCO \
does not have a skill for everything.
- "Why number N?": answer from CURRENT STATE with the skill's ESCO definition, the \
course sentence(s) that support it, quoted word for word, and the search that found it.
- A skill marked NOT IN TEXT can go on the list if the teacher wants it, but tell \
them the course text doesn't show it.
- Only quote the course word for word.
- Be brief. After changing the list, say in one line what it now holds.

CURRENT STATE
{state}"""

TEXT = {
    "en": {
        "intro": "I read the course ({sections}) and found {n} ESCO skills that the text supports:",
        "none": "I read the course ({sections}) but found no ESCO skill that the text clearly supports.",
        "ask": ("Tell me which to keep (for example \"keep 1, 2 and 4\"), ask \"why number 3?\", "
                "or ask \"anything on …?\" for a topic you expected to see."),
    },
    "pt": {
        "intro": "Li a unidade curricular ({sections}) e encontrei {n} competências ESCO que o texto suporta:",
        "none": "Li a unidade curricular ({sections}), mas não encontrei nenhuma competência ESCO que o texto suporte claramente.",
        "ask": ("Diga-me quais quer manter (por exemplo \"manter 1, 2 e 4\"), pergunte \"porquê a número 3?\", "
                "ou pergunte \"há alguma coisa sobre …?\" para um tema que esperava ver."),
    },
}


class AssistantState(MessagesState):
    sections: dict                      # section name -> text
    language: str                       # "en" or "pt"
    known: list[dict]                   # every skill seen; number = index + 1
    queries: list[str]                  # search phrases of the first turn
    events: Annotated[list[dict], operator.add]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _entry(skill: dict, origin: str, in_text: bool) -> dict:
    return {"id": skill["id"], "label": skill["label"], "type": skill.get("type", ""),
            "description": skill.get("description", ""),
            "evidence": skill.get("evidence", []), "in_text": in_text,
            "origin": origin, "status": "pending"}


def _evidence_text(skill: dict, limit: int = 160) -> str:
    return " | ".join(f"\"{_shorten(e['text'], limit)}\" ({e['section']})"
                      for e in skill["evidence"][:2])


def render_state(known: list[dict]) -> str:
    """The skills as the LLM sees them each turn."""
    if not known:
        return "No skills yet."
    groups = {"accepted": "ON THE LIST", "pending": "SUGGESTED, UNDECIDED",
              "rejected": "REJECTED BY THE TEACHER"}
    lines = []
    for status, title in groups.items():
        items = [(n, s) for n, s in enumerate(known, 1) if s["status"] == status]
        lines.append(f"{title}: {'none' if not items else ''}")
        for n, s in items:
            line = f"  {n}. {s['label']} [{s['type']}]"
            if status != "rejected":
                line += f" - definition: {_shorten(s['description'], 160) or 'n/a'}"
                line += (f" - evidence: {_evidence_text(s)}" if s["in_text"]
                         else " - NOT IN TEXT")
                line += f" - found by: {s['origin']}"
            lines.append(line)
    return "\n".join(lines)


def format_first_answer(known: list[dict], sections: Sections, language: str) -> str:
    t = TEXT[language]
    names = ", ".join(sections)
    if not known:
        return t["none"].format(sections=names) + "\n\n" + t["ask"]
    lines = [t["intro"].format(sections=names, n=len(known)), ""]
    for n, s in enumerate(known, 1):
        lines.append(f"{n}. **{s['label']}** - {_evidence_text(s, 140)}")
    return "\n".join(lines + ["", t["ask"]])


def export(state: dict) -> dict:
    """The final list and the log, for saving."""
    def out(s):
        return {"uri": SKILL_URI + s["id"], "label": s["label"], "type": s["type"],
                "in_text": s["in_text"], "evidence": s["evidence"], "origin": s["origin"]}
    known = state.get("known", [])
    return {"language": state.get("language"), "sections": list(state.get("sections", {})),
            "skills": [out(s) for s in known if s["status"] == "accepted"],
            "rejected": [out(s) for s in known if s["status"] == "rejected"],
            "undecided": [out(s) for s in known if s["status"] == "pending"],
            "events": state.get("events", [])}


def build_assistant(link, llm, judge_llm, checkpointer=None):
    """link: async (text, k) -> link_text's "skills" list, with descriptions.
    llm: chat model for the conversation. judge_llm: chat model bound to JSON
    output, for the query and judge steps."""
    llm_with_tools = llm.bind_tools(TOOLS)

    async def first_turn(state: AssistantState):
        sections = state["sections"]
        queries = await propose_queries(judge_llm, sections)
        candidates = await gather_candidates(link, sections, queries)
        result = await suggest_skills(candidates, judge_llm, sections)
        known = []
        for s in result["suggested"]:
            via = [q for q in s.get("found_by", []) if q != "whole text"]
            known.append(_entry(s, f"search '{via[0]}'" if via else "the whole course text", True))
        language = state.get("language") or "en"
        return {
            "known": known, "queries": queries, "language": language,
            "messages": [AIMessage(format_first_answer(known, sections, language))],
            "events": [{"t": _now(), "type": "suggested", "queries": queries,
                        "n_candidates": result["n_candidates"],
                        "skills": [s["id"] for s in known],
                        "judge_rejected": [s["id"] for s in result["rejected"]],
                        "first_attempt_valid": result["first_attempt_valid"]}],
        }

    async def agent(state: AssistantState):
        system = SYSTEM.format(language=LANGUAGES[state["language"]],
                               state=render_state(state["known"]))
        return {"messages": [await llm_with_tools.ainvoke(
            [SystemMessage(system), *state["messages"]])]}

    def update_skill_list(known: list[dict], accept=None, reject=None) -> tuple[dict, list[dict]]:
        accept, reject = list(accept or []), list(reject or [])
        bad = [n for n in accept + reject
               if not isinstance(n, int) or not 1 <= n <= len(known)]
        if bad:
            return {"error": f"no skill with number {bad}; valid numbers are 1-{len(known)}"}, []
        both = sorted(set(accept) & set(reject))
        if both:
            return {"error": f"numbers {both} are in both accept and reject"}, []
        events = []
        for numbers, status in ((accept, "accepted"), (reject, "rejected")):
            for n in numbers:
                skill = known[n - 1]
                if skill["status"] != status:
                    events.append({"t": _now(), "type": status, "skill": skill["id"],
                                   "was": skill["status"], "in_text": skill["in_text"]})
                    skill["status"] = status
        on_list = [f"{n}. {s['label']}" + ("" if s["in_text"] else " (NOT IN TEXT)")
                   for n, s in enumerate(known, 1) if s["status"] == "accepted"]
        return {"list_now": on_list}, events

    async def find_skills(known: list[dict], sections: Sections, topic: str
                          ) -> tuple[dict, list[dict]]:
        topic = (topic or "").strip()
        if not topic:
            return {"error": "topic is empty"}, []
        hits = await link(topic, FIND_K)
        number = {s["id"]: n for n, s in enumerate(known, 1)}
        already = [{"number": number[h["id"]], "label": h["label"],
                    "status": known[number[h["id"]] - 1]["status"]}
                   for h in hits if h["id"] in number]
        new = [h for h in hits if h["id"] not in number]
        result = await suggest_skills(new, judge_llm, sections) if new else {
            "suggested": [], "rejected": [], "unverified": []}
        out = {"topic": topic, "supported_by_course_text": [], "not_in_course_text": [],
               "already_known": already}
        for group, in_text in (("suggested", True), ("rejected", False), ("unverified", False)):
            for s in result[group]:
                known.append(_entry(s, f"search '{topic}'", in_text))
                item = {"number": len(known), "label": s["label"],
                        "definition": _shorten(s.get("description", ""), 160)}
                if in_text:
                    item["evidence"] = [{"section": e["section"], "text": e["text"]}
                                        for e in s["evidence"][:2]]
                out["supported_by_course_text" if in_text else "not_in_course_text"].append(item)
        event = {"t": _now(), "type": "search", "topic": topic,
                 "supported": [s["id"] for s in result["suggested"]],
                 "not_in_text": [s["id"] for s in result["rejected"] + result["unverified"]],
                 "already_known": [h["id"] for h in hits if h["id"] in number]}
        return out, [event]

    async def tools(state: AssistantState):
        known = [dict(s) for s in state["known"]]   # work on a copy, one call at a time
        messages, events = [], []
        for call in state["messages"][-1].tool_calls:
            args = call["args"] if isinstance(call["args"], dict) else {}
            if call["name"] == "update_skill_list":
                result, new = update_skill_list(known, args.get("accept"), args.get("reject"))
            elif call["name"] == "find_skills":
                result, new = await find_skills(known, state["sections"], args.get("topic"))
            else:
                result, new = {"error": f"unknown tool {call['name']}"}, []
            events += new
            messages.append(ToolMessage(json.dumps(result, ensure_ascii=False),
                                        tool_call_id=call["id"], name=call["name"]))
        return {"known": known, "messages": messages, "events": events}

    def route_start(state: AssistantState) -> str:
        return "agent" if state.get("known") is not None else "first_turn"

    def route_agent(state: AssistantState) -> str:
        return "tools" if state["messages"][-1].tool_calls else END

    g = StateGraph(AssistantState)
    g.add_node("first_turn", first_turn)
    g.add_node("agent", agent)
    g.add_node("tools", tools)
    g.add_conditional_edges(START, route_start, ["first_turn", "agent"])
    g.add_edge("first_turn", END)
    g.add_conditional_edges("agent", route_agent, ["tools", END])
    g.add_edge("tools", "agent")
    return g.compile(checkpointer=checkpointer)


# --- command line -------------------------------------------------------------

def _main() -> None:
    import argparse
    import asyncio
    import os
    import uuid

    from dotenv import load_dotenv
    from langchain_core.messages import HumanMessage
    from langchain_deepseek import ChatDeepSeek
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from langchain_mcp_adapters.tools import load_mcp_tools
    from langgraph.checkpoint.memory import InMemorySaver

    from esco_agent.graph import parse_tool_result
    from esco_agent.mcp_client import server_config
    from esco_agent.suggest import load_sections

    load_dotenv()
    ap = argparse.ArgumentParser(description="Build a course's ESCO skill list in conversation.")
    ap.add_argument("--course", required=True, help=".json of named sections, or .txt")
    ap.add_argument("--lang", choices=sorted(LANGUAGES), default="en", help="chat language")
    ap.add_argument("--out", default="skill_list.json", help="where the list and log are saved")
    ap.add_argument("--trace", action="store_true", help="show tool calls")
    args = ap.parse_args()
    sections = load_sections(args.course)

    def show(messages):
        for m in messages:
            if args.trace and isinstance(m, AIMessage) and m.tool_calls:
                for tc in m.tool_calls:
                    print(f"  [tool] {tc['name']}({tc['args']})")
            elif args.trace and isinstance(m, ToolMessage):
                print(f"  [result] {str(m.content)[:200]}")
            elif isinstance(m, AIMessage) and m.content:
                print(f"\n{m.content}\n")

    async def chat():
        model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
        llm = ChatDeepSeek(model=model, temperature=0)
        judge = llm.bind(response_format={"type": "json_object"})
        async with MultiServerMCPClient(server_config()).session("esco") as session:
            tools = {t.name: t for t in await load_mcp_tools(session)}

            async def link(text, k):
                return parse_tool_result(await tools["link_text"].ainvoke(
                    {"text": text, "k": k, "include_descriptions": True}))["skills"]

            graph = build_assistant(link, llm, judge, InMemorySaver())
            config = {"configurable": {"thread_id": uuid.uuid4().hex}}
            print("Reading the course (the first run loads the models)...")
            state = await graph.ainvoke({"sections": sections, "language": args.lang,
                                         "messages": [HumanMessage(course_text(sections))]}, config)
            show(state["messages"][1:])
            seen = len(state["messages"])
            while True:
                try:
                    q = (await asyncio.to_thread(input, "you> ")).strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if not q:
                    break
                state = await graph.ainvoke({"messages": [HumanMessage(q)]}, config)
                show(state["messages"][seen + 1:])
                seen = len(state["messages"])
            result = export(state)
            result["model"] = state["messages"][-1].response_metadata.get("model_name") or model
            return result

    result = asyncio.run(chat())
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nSkill list ({len(result['skills'])}) saved to {args.out}:")
    for s in result["skills"]:
        print(f"  - {s['label']}" + ("" if s["in_text"] else "  (not in the course text)"))


if __name__ == "__main__":
    _main()