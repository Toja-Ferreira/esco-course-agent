"""Command-line chat with the ESCO course agent.

    python -m esco_agent.cli --course course.txt
    python -m esco_agent.cli            # paste the course, end with an empty line
    python -m esco_agent.cli --course course.txt --trace   # show tool calls

Needs DEEPSEEK_API_KEY (in .env). DEEPSEEK_MODEL pins the model (default
deepseek-chat, an alias that moves). ESCO data from ESCO_DATA_DIR (default ./data).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import uuid

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.checkpoint.memory import InMemorySaver

from esco_agent import llm_calls
from esco_agent.graph import build_graph
from esco_agent.mcp_client import server_config


def read_course(path: str | None) -> str:
    if path:
        with open(path, encoding="utf-8") as f:
            return f.read().strip()
    print("Paste the course description, then an empty line:")
    lines = []
    while (line := input()).strip():
        lines.append(line)
    return "\n".join(lines)


def show(messages, trace: bool) -> None:
    for m in messages:
        if trace and isinstance(m, AIMessage) and m.tool_calls:
            for tc in m.tool_calls:
                args = {k: (v if len(str(v)) < 80 else str(v)[:77] + "...")
                        for k, v in tc["args"].items()}
                print(f"  [tool] {tc['name']}({args})")
        elif trace and isinstance(m, ToolMessage):
            print(f"  [result] {m.name}: {str(m.content)[:150]}...")
        elif isinstance(m, AIMessage) and m.content:
            print(f"\n{m.content}\n")


async def chat(course: str, trace: bool) -> None:
    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
    llm = llm_calls.deepseek(streamed=True, model=model)
    coverage_llm = llm_calls.deepseek(json_output=True, model=model)

    client = MultiServerMCPClient(server_config())
    async with client.session("esco") as session:   # one server for the whole chat
        tools = await load_mcp_tools(session)
        graph = build_graph(tools, llm, coverage_llm, checkpointer=InMemorySaver())
        config = {"configurable": {"thread_id": uuid.uuid4().hex}}

        print("Analysing the course (the first run loads the models)...")
        state = await graph.ainvoke(
            {"course": course, "messages": [HumanMessage(course)]}, config)
        show(state["messages"][1:], trace)
        seen = len(state["messages"])

        while True:
            try:
                q = (await asyncio.to_thread(input, "you> ")).strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q:
                break
            state = await graph.ainvoke({"messages": [HumanMessage(q)]}, config)
            show(state["messages"][seen + 1:], trace)
            seen = len(state["messages"])
            served = state["messages"][-1].response_metadata.get("model_name")
            if trace and served:
                print(f"  [model] {served}")


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser(description="Chat about which ESCO occupations "
                                             "a course prepares people for.")
    ap.add_argument("--course", help="text file with the course description")
    ap.add_argument("--trace", action="store_true", help="show tool calls")
    args = ap.parse_args()
    course = read_course(args.course)
    if not course:
        sys.exit("no course text")
    asyncio.run(chat(course, args.trace))


if __name__ == "__main__":
    main()
