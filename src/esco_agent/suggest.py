"""Evidence-backed skill suggestions for a course text.

Three stages:
  1. Queries: the LLM reads the course and proposes short skill phrases in the
     style of ESCO labels ("debug software", "Linux"). These are search
     queries, never shown as skills.
  2. Candidates (recall): the linker is run on the whole course and on each
     query. Only the linker's results, which are real ESCO skills, go forward.
  3. Judge (precision): the LLM reads the course and keeps only the candidates
     the text actually supports, each with the sentence(s) that show it. Same
     judge and validator as check_coverage. The linker ranks by similarity, so
     near-misses (Objective-C, C++ for a C course) are expected and rejected here.

Why LLM queries (13 Nov 2026, one C course, so a lead and not a result): the
whole text alone collapsed to its main topic (28 programming languages, 2
supported skills). Raw sentences as queries added word-sense junk ("demolish
structures", "file documents") and reached 4. POF's current pipeline showed
the same weak shortlist, and its useful suggestions were ones the LLM added
from memory, unchecked. Here the LLM's proposals are only queries; every
suggestion still comes from the linker and is checked against the text.
--candidates whole|sentences|llm keeps all three for comparison.

Try it:
    python -m esco_agent.suggest --course course.txt
"""

from __future__ import annotations

import asyncio

from esco_agent.coverage import judge_coverage, split_sentences

import json

WHOLE_TEXT_K = 15       # candidates from the course as one query
PER_QUERY_K = 4         # candidates from each sentence or LLM query
MAX_QUERIES = 25        # LLM queries used
MIN_SENTENCE_WORDS = 5  # skip headings like "Objectives" or "TP component:"
JUDGE_BATCH = 30        # candidates per judge call (calls run in parallel)


def unique_sentences(course: str) -> list[str]:
    """Course pages often repeat a paragraph (objectives = learning outcomes)."""
    return list(dict.fromkeys(split_sentences(course)))


QUERY_PROMPT = """Read this university course description and list the distinct \
skills and knowledge areas it teaches.

Course:
{course}

Write each as a short phrase in the style of a skills taxonomy label: verb + \
object for skills ("prepare financial statements", "manage a team"), a noun \
phrase for knowledge ("organic chemistry", "contract law"). Be specific: one \
phrase per distinct thing taught, including tools, methods and practices the \
text names. Only what the text states; nothing about how the course is assessed \
or taught. Between 8 and 20 phrases.

Reply with JSON only: {{"queries": ["...", "..."]}}"""


def sentence_queries(course: str) -> list[str]:
    return [s for s in unique_sentences(course) if len(s.split()) >= MIN_SENTENCE_WORDS]


async def propose_queries(llm, course: str) -> list[str]:
    """llm: chat model bound to JSON output. Returns [] if the reply is unusable
    (the caller then falls back to whole-text candidates only)."""
    reply = await llm.ainvoke([("user", QUERY_PROMPT.format(course=course))])
    try:
        items = json.loads(reply.content)["queries"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return []
    if not isinstance(items, list):
        return []
    clean = [q.strip() for q in items if isinstance(q, str) and q.strip()]
    return list(dict.fromkeys(clean))[:MAX_QUERIES]


async def gather_candidates(link, course: str, queries: list[str],
                            whole_k: int = WHOLE_TEXT_K,
                            per_query_k: int = PER_QUERY_K) -> list[dict]:
    """link: async callable (text, k) -> link_text's "skills" list.
    Runs the linker on the whole course and on each query. Returns unique
    candidates. Scores are only comparable within one query, so each candidate
    keeps its best score; order is whole-text hits first, then query order.
    Each candidate records the queries that found it ("found_by")."""
    seen: dict[str, dict] = {}
    for text, k, source in [(course, whole_k, "whole text")] + [
            (q, per_query_k, q) for q in queries]:
        for skill in await link(text, k):
            old = seen.get(skill["id"])
            if old is None:
                seen[skill["id"]] = {**skill, "found_by": [source]}
            else:
                old["score"] = max(old["score"], skill["score"])
                old["found_by"].append(source)
    return list(seen.values())


SUGGEST_PROMPT = """You are helping a teacher choose the skills that describe a \
university course.

Course sentences:
{sentences}

Candidate skills (from a search engine, so some are near-misses):
{skills}

For EVERY candidate above, decide from the course sentences alone:
- "covered": the course explicitly teaches or practises this skill. Cite every \
sentence number that shows it.
- "missing": the sentences don't show it. A candidate that is only similar to \
what the course teaches (another programming language, a neighbouring field, \
teaching the subject rather than learning it) is "missing".

Reply with JSON only, one entry per skill number, in this form:
{{"judgements": [{{"skill": 1, "verdict": "covered", "sentences": [2, 5]}}, \
{{"skill": 2, "verdict": "missing", "sentences": []}}]}}"""


async def suggest_skills(candidates: list[dict], judge_llm, course: str) -> dict:
    """candidates: skills as returned by link_text (id, label, type, score).
    Returns suggested (with evidence), rejected and unverified candidates."""
    sentences = unique_sentences(course)
    batches = [candidates[i:i + JUDGE_BATCH] for i in range(0, len(candidates), JUDGE_BATCH)]
    results = await asyncio.gather(*[
        judge_coverage(judge_llm, sentences, "", [c["label"] for c in batch],
                       prompt_template=SUGGEST_PROMPT) for batch in batches])
    out = {"sentences": sentences, "suggested": [], "rejected": [], "unverified": [],
           "n_candidates": len(candidates), "judge_calls": len(batches),
           "first_attempt_valid": all(r.first_attempt_valid for r in results)}
    for batch, res in zip(batches, results):
        for n, cand in enumerate(batch, 1):
            jd = res.judgements[n]
            entry = dict(cand)
            if jd.verdict == "covered":
                entry["evidence"] = [{"sentence": s, "text": sentences[s - 1]}
                                     for s in jd.sentences]
                out["suggested"].append(entry)
            else:
                out["rejected" if jd.verdict == "missing" else "unverified"].append(entry)
    return out


def _main() -> None:
    import argparse
    import asyncio
    import os
    import sys

    from dotenv import load_dotenv
    from langchain_deepseek import ChatDeepSeek
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from langchain_mcp_adapters.tools import load_mcp_tools

    from esco_agent.graph import parse_tool_result

    load_dotenv()
    ap = argparse.ArgumentParser(description="Suggest ESCO skills for a course, with evidence.")
    ap.add_argument("--course", required=True, help="text file with the course description")
    ap.add_argument("--candidates", choices=["llm", "sentences", "whole"], default="llm",
                    help="where candidates come from besides the whole text: LLM-proposed "
                         "queries (default), each sentence, or nothing")
    args = ap.parse_args()
    course = open(args.course, encoding="utf-8").read().strip()

    async def run():
        llm = ChatDeepSeek(model=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
                           temperature=0).bind(response_format={"type": "json_object"})
        client = MultiServerMCPClient({"esco": {
            "command": sys.executable, "args": ["-m", "esco_mcp.server"],
            "transport": "stdio", "env": dict(os.environ)}})
        async with client.session("esco") as session:
            tools = {t.name: t for t in await load_mcp_tools(session)}

            async def link(text, k):
                return parse_tool_result(await tools["link_text"].ainvoke(
                    {"text": text, "k": k}))["skills"]

            if args.candidates == "llm":
                queries = await propose_queries(llm, course)
            elif args.candidates == "sentences":
                queries = sentence_queries(course)
            else:
                queries = []
            cands = await gather_candidates(
                link, course, queries, whole_k=WHOLE_TEXT_K if queries else 30)
        result = await suggest_skills(cands, llm, course)
        result["queries"] = queries if args.candidates == "llm" else []
        return result

    r = asyncio.run(run())
    if r["queries"]:
        print(f"\nLLM QUERIES ({len(r['queries'])}): " + "; ".join(r["queries"]))
    print(f"\nSUGGESTED ({len(r['suggested'])} of {r['n_candidates']} candidates, "
          f"{r['judge_calls']} judge calls)")
    for s in r["suggested"]:
        via = [q for q in s.get("found_by", []) if q != "whole text"][:2]
        print(f"  + {s['label']}  [{s['type']}]" + (f"  <- query: {' | '.join(via)}" if via else ""))
        for e in s["evidence"][:3]:
            print(f"      {e['sentence']}. {e['text'][:110]}")
    print(f"\nREJECTED by the judge ({len(r['rejected'])})")
    print("  " + "; ".join(f"{s['label']} ({s['score']:.2f})" for s in r["rejected"]))
    if r["unverified"]:
        print(f"\nUNVERIFIED ({len(r['unverified'])}): "
              + "; ".join(s["label"] for s in r["unverified"]))
    print(f"\njudge reply valid on first attempt: {r['first_attempt_valid']}")


if __name__ == "__main__":
    _main()
