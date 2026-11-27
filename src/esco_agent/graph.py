"""The LangGraph agent.

    START --(first message)--> first_turn --> END
    START --(follow-ups)-----> agent <--> tools   (tool-calling loop) --> END

first_turn is fixed, not LLM-driven: link_text on the course, then
search_occupations with the linker's scores as weights. Its tool calls are
written into the history as if the agent had made them, so follow-ups can
refer to "the 2nd one" by the ids in those results. The first answer is
formatted by code, so it is grounded by construction.

Follow-ups go to the LLM with four tools: the three MCP tools plus
check_coverage, which is local because it needs an LLM (the MCP server stays
pure data).
"""

from __future__ import annotations

import json
import uuid
from typing import Annotated

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import InjectedState, ToolNode, tools_condition

from esco_agent import llm_calls
from esco_agent.coverage import judge_coverage, split_sentences

LINK_K = 20
SEARCH_K = 10

SYSTEM = """You help someone designing a university course see which ESCO occupations \
the course prepares people for, and what it is missing.

The course text and a first analysis are already in this conversation: the ESCO \
skills found in the course (link_text) and the best-matching occupations \
(search_occupations), in the order shown to the user.

Rules:
- Only name skills and occupations that appear in a tool result in this conversation. \
Never invent ESCO skills or occupations, and never rely on memory of ESCO.
- Only quote the course word for word. Quote from check_coverage's sentences.
- "The 2nd one", "that one" etc. refer to the numbered occupations last shown to the user.
- What a course is missing for an occupation: call check_coverage. Do not compare \
skill ids yourself; the course's linked skills are only a top-20 list.
- Restricting to a sector ("only ICT jobs"): call search_occupations again with the \
same skill_ids and weights as the first search_occupations call, plus isco_prefix \
(ISCO-08: "25" ICT professionals, "35" ICT technicians, "2" professionals, \
"3" technicians and associate professionals, "1" managers).
- Why an occupation matched: use its top_matching_skills from search_occupations.
- If no tool can answer, say so plainly.
Be concise. Refer to skills and occupations by their labels."""


class AgentState(MessagesState):
    course: str
    sentences: list[str]


def parse_tool_result(result) -> dict:
    """MCP tool results arrive via langchain-mcp-adapters as a JSON string or a
    list of content blocks; return the decoded JSON."""
    if isinstance(result, str):
        return json.loads(result)
    if isinstance(result, list):
        text = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in result)
        return json.loads(text)
    if isinstance(result, dict):
        return result
    raise TypeError(f"unexpected tool result: {type(result).__name__}")


def _call(name: str, args: dict) -> dict:
    return {"name": name, "args": args, "id": f"call_{uuid.uuid4().hex[:12]}",
            "type": "tool_call"}


def format_first_answer(linked: dict, occ: dict) -> str:
    skills = linked["skills"]
    lines = ["**ESCO skills found in the course** (top 10 of "
             f"{len(skills)}): " + "; ".join(s["label"] for s in skills[:10]), "",
             "**Best-matching ESCO occupations:**"]
    for i, o in enumerate(occ["occupations"], 1):
        why = ", ".join(s["label"] for s in o["top_matching_skills"])
        lines.append(f"{i}. {o['label']} (ISCO {o['isco_group']}, score {o['score']:.2f})"
                     f" - matched on: {why}")
    lines += ["", "Ask, for example: \"what's missing for the 2nd one?\", "
              "\"only ICT jobs\", or \"why the first one?\""]
    return "\n".join(lines)


def build_graph(mcp_tools: list, llm, coverage_llm, checkpointer=None):
    """mcp_tools: tools from load_mcp_tools on ONE persistent session.
    llm: chat model for the follow-up loop. coverage_llm: chat model bound to
    JSON output, for check_coverage."""
    by_name = {t.name: t for t in mcp_tools}

    @tool
    async def check_coverage(occupation_id: str,
                             state: Annotated[dict, InjectedState]) -> str:
        """Judge which of an occupation's essential skills the course covers,
        by reading the course text. Returns covered skills with the course
        sentences that show them, missing skills, and any the check could not
        verify. Use this for any question about what the course is missing."""
        raw = await by_name["get_occupation_skills"].ainvoke(
            {"occupation_id": occupation_id})
        try:
            occ = parse_tool_result(raw)
        except (ValueError, TypeError):
            return f"Could not look up occupation {occupation_id!r}: {raw}"
        sentences = state["sentences"]
        essential = occ["essential"]
        res = await judge_coverage(coverage_llm, sentences, occ["label"],
                                   [s["label"] for s in essential])
        out = {"occupation": occ["label"], "occupation_id": occ["id"],
               "covered": [], "missing": [], "unverified": [],
               "first_attempt_valid": res.first_attempt_valid}
        for n, skill in enumerate(essential, 1):
            jd = res.judgements[n]
            entry = {"id": skill["id"], "label": skill["label"]}
            if jd.verdict == "covered":
                entry["evidence"] = [{"sentence": s, "text": sentences[s - 1]}
                                     for s in jd.sentences]
            out[jd.verdict].append(entry)
        return json.dumps(out, ensure_ascii=False)

    tools = [*mcp_tools, check_coverage]
    llm_with_tools = llm.bind_tools(tools)

    async def first_turn(state: AgentState):
        course = state["course"]
        link_call = _call("link_text", {"text": course, "k": LINK_K})
        linked = parse_tool_result(await by_name["link_text"].ainvoke(link_call["args"]))
        ids = [s["id"] for s in linked["skills"]]
        weights = [s["score"] for s in linked["skills"]]
        search_call = _call("search_occupations",
                            {"skill_ids": ids, "weights": weights, "k": SEARCH_K})
        occ = parse_tool_result(await by_name["search_occupations"].ainvoke(search_call["args"]))
        return {
            "sentences": split_sentences(course),
            "messages": [
                AIMessage(content="", tool_calls=[link_call, search_call]),
                ToolMessage(json.dumps(linked, ensure_ascii=False),
                            tool_call_id=link_call["id"], name="link_text"),
                ToolMessage(json.dumps(occ, ensure_ascii=False),
                            tool_call_id=search_call["id"], name="search_occupations"),
                AIMessage(content=format_first_answer(linked, occ)),
            ],
        }

    async def agent(state: AgentState):
        reply = await llm_calls.call(llm_with_tools, [SystemMessage(SYSTEM), *state["messages"]],
                                     "conversation", streamed=True)
        return {"messages": [reply]}

    def route_start(state: AgentState) -> str:
        return "agent" if state.get("sentences") else "first_turn"

    g = StateGraph(AgentState)
    g.add_node("first_turn", first_turn)
    g.add_node("agent", agent)
    g.add_node("tools", ToolNode(tools))
    g.add_conditional_edges(START, route_start, ["first_turn", "agent"])
    g.add_edge("first_turn", END)
    g.add_conditional_edges("agent", tools_condition)
    g.add_edge("tools", "agent")
    return g.compile(checkpointer=checkpointer)
