"""Spike: does DeepSeek tool calling work in a LangGraph loop?

Uses fake stand-ins with the three real tool signatures. No linker, no MCP yet.
Simulates the state after the fixed first turn, then sends three follow-ups.

PASS for each probe means:
  - the model calls the expected tool (or none), with valid, parseable arguments
  - the tool result goes back to the model, and the run ends with a plain answer

Run from the repo root (DEEPSEEK_API_KEY in .env):
      python spikes/spike_deepseek_tools.py
      (optional) DEEPSEEK_MODEL=... in .env to pin a model instead of the alias
"""
import os
import sys

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_deepseek import ChatDeepSeek
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()  # reads DEEPSEEK_API_KEY (and optional DEEPSEEK_MODEL) from .env
MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")


# --- Fake tools: real signatures, fixed outputs ------------------------------

@tool
def link_text(text: str) -> list[dict]:
    """Find the ESCO skills mentioned in a piece of text. Returns skill IDs and labels."""
    return [{"id": "sk-1", "label": "SQL"},
            {"id": "sk-2", "label": "Python (computer programming)"}]


@tool
def search_occupations(skill_ids: list[str], sector: str | None = None) -> list[dict]:
    """Rank ESCO occupations by how well they match a set of ESCO skill IDs.
    Optionally restrict to one sector (e.g. 'ICT')."""
    return [{"rank": 1, "id": "occ-1", "label": "software developer"},
            {"rank": 2, "id": "occ-2", "label": "database designer"},
            {"rank": 3, "id": "occ-3", "label": "data analyst"}]


@tool
def get_occupation_skills(occupation_id: str) -> dict:
    """Return the essential and optional ESCO skills of one occupation."""
    return {"occupation_id": occupation_id,
            "essential": [{"id": "sk-1", "label": "SQL"},
                          {"id": "sk-9", "label": "design database scheme"}],
            "optional": []}


TOOLS = [link_text, search_occupations, get_occupation_skills]


# --- Graph: the follow-up loop from the spec ---------------------------------

llm = ChatDeepSeek(model=MODEL, temperature=0).bind_tools(TOOLS)


def agent(state: MessagesState):
    return {"messages": [llm.invoke(state["messages"])]}


builder = StateGraph(MessagesState)
builder.add_node("agent", agent)
builder.add_node("tools", ToolNode(TOOLS))
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")
graph = builder.compile()


# --- Conversation state after the fixed first turn ---------------------------

SYSTEM = ("You help a course designer relate a course to ESCO occupations. "
          "Use the tools for any ESCO fact not already in this conversation. "
          "Refer to occupations by the IDs shown.")
COURSE = ("Databases (6 ECTS). Students design relational schemas, write SQL "
          "queries, and build small data pipelines in Python.")
FIRST_TURN = ("Skills found in the course: sk-1 SQL; sk-2 Python (computer programming).\n"
              "Best-matching occupations: 1. software developer (occ-1); "
              "2. database designer (occ-2); 3. data analyst (occ-3).")
HISTORY = [SystemMessage(SYSTEM), HumanMessage(COURSE), AIMessage(FIRST_TURN)]

# (follow-up, expected tool or None, check on that tool's arguments)
PROBES = [
    ("What's missing for the 2nd one?", "get_occupation_skills",
     lambda a: a.get("occupation_id") == "occ-2"),
    ("Show me only ICT jobs.", "search_occupations",
     lambda a: bool(a.get("sector")) and bool(a.get("skill_ids"))),
    ("Thanks, that's all for now.", None, lambda a: True),
]


def run_probe(question, expected, args_ok):
    print(f"\n{'=' * 70}\n{question}\n{'=' * 70}")

    # Stream node by node, printing each message as it is produced.
    new = []
    for step in graph.stream({"messages": HISTORY + [HumanMessage(question)]},
                             stream_mode="updates"):
        for node, update in step.items():
            print(f"\n--- node: {node}")
            for m in update["messages"]:
                m.pretty_print()
                new.append(m)

    first, last = new[0], new[-1]

    called = [tc["name"] for tc in first.tool_calls]
    problems = []
    if first.invalid_tool_calls:
        problems.append(f"malformed tool calls: {first.invalid_tool_calls}")
    if expected is None and called:
        problems.append(f"expected no tool, got {called}")
    if expected is not None:
        if called != [expected]:
            problems.append(f"expected [{expected}], got {called}")
        elif not args_ok(first.tool_calls[0]["args"]):
            problems.append(f"bad args: {first.tool_calls[0]['args']}")
    if not (isinstance(last, AIMessage) and not last.tool_calls and last.content):
        problems.append("run did not end with a plain answer")

    served = last.response_metadata.get("model_name", "unknown")
    print(f"\n[served model: {served}]")
    print("PASS" if not problems else "FAIL: " + "; ".join(problems))
    return not problems


if __name__ == "__main__":
    print(f"model: {MODEL}")
    results = [run_probe(*p) for p in PROBES]
    print(f"\n{sum(results)}/{len(results)} probes passed")
    sys.exit(0 if all(results) else 1)