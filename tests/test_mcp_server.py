"""The MCP server, exercised the way the agent will use it: as a subprocess
over stdio, through a real MCP client. Skipped when data/ is missing.
Run:  pytest tests/test_mcp_server.py -v -s
"""

import asyncio
import json
import os
import sys

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from esco_core.linker import EMBEDDINGS_FILE, data_dir

SERVER = StdioServerParameters(
    command=sys.executable, args=["-m", "esco_mcp.server"],
    env={**os.environ, "ESCO_DATA_DIR": str(data_dir())},
)
COURSE = ("Students design relational database schemas, apply normalisation and "
          "write SQL queries with joins and aggregation, and build small data "
          "pipelines in Python.")

pytestmark = pytest.mark.skipif(not (data_dir() / EMBEDDINGS_FILE).exists(),
                                reason=f"no data in {data_dir()}")


def call(session, name, **args):
    async def _go():
        r = await session.call_tool(name, args)
        text = r.content[0].text if r.content else ""
        return r.isError, (text if r.isError else json.loads(text))
    return _go()


def test_server_end_to_end():
    async def run():
        async with stdio_client(SERVER) as (read, write):
            async with ClientSession(read, write) as s:
                await s.initialize()

                tools = {t.name for t in (await s.list_tools()).tools}
                assert tools == {"link_text", "search_occupations",
                                 "get_occupation_skills"}

                err, linked = await call(s, "link_text", text=COURSE, k=20)
                assert not err and len(linked["skills"]) == 20
                ids = [x["id"] for x in linked["skills"]]
                w = [x["score"] for x in linked["skills"]]
                print("\nskills:", [x["label"] for x in linked["skills"][:6]])

                err, occ = await call(s, "search_occupations", skill_ids=ids, weights=w)
                assert not err and occ["occupations"]
                top = occ["occupations"][0]
                assert top["top_matching_skills"]
                print("occupations:", [o["label"] for o in occ["occupations"][:5]])

                err, ict = await call(s, "search_occupations", skill_ids=ids,
                                      weights=w, isco_prefix="25")
                assert not err and all(o["isco_group"].startswith("25")
                                       for o in ict["occupations"])

                err, req = await call(s, "get_occupation_skills",
                                      occupation_id=top["id"])
                assert not err and req["essential"] and req["label"] == top["label"]
                have = set(ids)
                missing = [x["label"] for x in req["essential"] if x["id"] not in have]
                print(f"{top['label']}: {len(req['essential'])} essential, "
                      f"missing by id: {missing[:6]}")

                # full URIs are accepted too
                err, _ = await call(s, "get_occupation_skills", occupation_id=
                                    "http://data.europa.eu/esco/occupation/" + top["id"])
                assert not err

                # errors come back as tool errors, not crashes
                err, msg = await call(s, "get_occupation_skills", occupation_id="nope")
                assert err and "not an ESCO occupation id" in msg
                err, msg = await call(s, "search_occupations", skill_ids=["made-up"])
                assert err and "none of these" in msg
                err, part = await call(s, "search_occupations",
                                       skill_ids=[ids[0], "made-up"])
                assert not err and part["unknown_skill_ids"] == ["made-up"]
                err, msg = await call(s, "search_occupations", skill_ids=ids[:2],
                                      weights=[1.0])
                assert err and "weights" in msg

    asyncio.run(run())
