"""Agent plumbing: the real MCP server, with scripted LLMs standing in for DeepSeek.

Checks the graph, not the model: fixed first turn, "the 2nd one" reaching
check_coverage (with a forced retry), a sector filter, and a no-tool reply.
Skipped when data/ is missing. Run:  pytest tests/test_agent_graph.py -v -s
"""
import pytest
from esco_core.linker import EMBEDDINGS_FILE, data_dir

pytestmark = pytest.mark.skipif(not (data_dir() / EMBEDDINGS_FILE).exists(),
                                reason=f"no data in {data_dir()}")
import asyncio, json, os, re, sys
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.checkpoint.memory import InMemorySaver
from esco_agent.graph import build_graph, parse_tool_result as P

def last_tool(messages, name):
    return P(next(m.content for m in reversed(messages) if isinstance(m, ToolMessage) and m.name == name))

class ScriptedAgentLLM:
    """Stands in for DeepSeek in the follow-up loop: decides tool calls from the question."""
    def bind_tools(self, tools):
        self.tool_names = {t.name for t in tools}; return self
    async def ainvoke(self, messages):
        last = messages[-1]
        if isinstance(last, ToolMessage):                      # tool done -> answer
            if last.name == "check_coverage":
                r = P(last.content)
                return AIMessage(f"For {r['occupation']}: {len(r['covered'])} covered, {len(r['missing'])} missing, {len(r['unverified'])} unverified.")
            occ = P(last.content)["occupations"]
            return AIMessage("ICT only: " + "; ".join(o["label"] + f" ({o['isco_group']})" for o in occ))
        q = last.content.lower()
        if "missing" in q:
            occ = last_tool(messages, "search_occupations")["occupations"]
            return AIMessage("", tool_calls=[{"name": "check_coverage", "args": {"occupation_id": occ[1]["id"]}, "id": "c1", "type": "tool_call"}])
        if "ict" in q:
            first = next(m for m in messages if isinstance(m, AIMessage) and any(tc["name"] == "search_occupations" for tc in m.tool_calls))
            args = dict(next(tc["args"] for tc in first.tool_calls if tc["name"] == "search_occupations"), isco_prefix="25")
            return AIMessage("", tool_calls=[{"name": "search_occupations", "args": args, "id": "c2", "type": "tool_call"}])
        return AIMessage("You're welcome.")

class ScriptedCoverageLLM:
    """First reply drops the last skill (forces a retry); the retry fills it in."""
    calls = 0
    async def ainvoke(self, messages):
        ScriptedCoverageLLM.calls += 1
        prompt = messages[0][1]
        n_skills = len(re.findall(r"^\d+\. ", prompt.split("Skills required")[1].split("For EVERY")[0], re.M))
        if len(messages) == 1:
            js = [{"skill": i, "verdict": "covered" if i % 2 else "missing", "sentences": [1] if i % 2 else []} for i in range(1, n_skills)]
        else:
            js = [{"skill": n_skills, "verdict": "missing", "sentences": []}]
        return type("R", (), {"content": json.dumps({"judgements": js})})()


async def run():
    client = MultiServerMCPClient({"esco": {"command": sys.executable, "args": ["-m", "esco_mcp.server"],
                                            "transport": "stdio", "env": {**os.environ, "ESCO_DATA_DIR": str(data_dir())}}})
    async with client.session("esco") as session:
        graph = build_graph(await load_mcp_tools(session), ScriptedAgentLLM(), ScriptedCoverageLLM(), InMemorySaver())
        cfg = {"configurable": {"thread_id": "t1"}}
        course = ("Students design relational database schemas. They write SQL queries with joins.\n"
                  "- version control with git\n- unit testing in Java")
        s = await graph.ainvoke({"course": course, "messages": [HumanMessage(course)]}, cfg)
        assert len(s["sentences"]) == 4
        assert "Best-matching ESCO occupations" in s["messages"][-1].content
        second = last_tool(s["messages"], "search_occupations")["occupations"][1]["id"]
        print("\n" + s["messages"][-1].content)

        n = len(s["messages"])
        s = await graph.ainvoke({"messages": [HumanMessage("What's missing for the 2nd one?")]}, cfg)
        new = s["messages"][n:]; n = len(s["messages"])
        calls = [tc for m in new if isinstance(m, AIMessage) for tc in m.tool_calls]
        assert [c["name"] for c in calls] == ["check_coverage"]
        assert calls[0]["args"]["occupation_id"] == second
        cov = P(next(m.content for m in new if isinstance(m, ToolMessage)))
        assert cov["first_attempt_valid"] is False and ScriptedCoverageLLM.calls == 2
        assert not cov["unverified"]
        for c in cov["covered"]:
            assert all(e["text"] in course for e in c["evidence"])

        s = await graph.ainvoke({"messages": [HumanMessage("Show me only ICT jobs")]}, cfg)
        ict = last_tool(s["messages"][n:], "search_occupations")["occupations"]
        assert ict and all(o["isco_group"].startswith("25") for o in ict)
        print("ICT:", [o["label"] for o in ict[:5]])
        n = len(s["messages"])

        s = await graph.ainvoke({"messages": [HumanMessage("thanks")]}, cfg)
        assert not [tc for m in s["messages"][n:] if isinstance(m, AIMessage) for tc in m.tool_calls]


def test_agent_graph():
    asyncio.run(run())
