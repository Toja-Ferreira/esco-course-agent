"""Hand-labelling of the judge's suggestions, for evidence precision.

    make   runs the assistant's first-turn pipeline once per course (LLM search
           phrases -> linker, 6 per phrase -> judge with ESCO definitions) and writes
           one row per suggested skill: the skill, its ESCO definition and the course
           sentences the judge cited. Rows are shuffled within each course, so the
           judge's order doesn't steer the labelling. Fill in `verdict`:
               right       the sentence shows the course teaches what the definition describes
               borderline  related or partly covered, or too generic to be sure
               wrong       a different sense (shared words), or the sentence doesn't support it
    label  goes through the sheet in the terminal, one suggestion at a time: press
           1 right, 2 borderline, 3 wrong, b back, s skip, q quit. Saves after every
           key, so it can be stopped and resumed. --per-course N labels only rows 1..N
           of each course (rows are shuffled, so that is a random sample).
    score  reads the filled sheet: precision (strict: right; lenient: right + borderline)
           with 95% bootstrap intervals over courses, also split into generic
           (ESCO "transversal") and course-specific skills.

    python -m esco_agent.labelsheet make courses/*.json --out results/2026-11-20_labels.csv
    python -m esco_agent.labelsheet label results/2026-11-20_labels.csv --per-course 8
    python -m esco_agent.labelsheet score results/2026-11-20_labels.csv
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import random
from pathlib import Path

import numpy as np

FIELDS = ["course", "row", "skill", "esco_definition", "evidence", "verdict", "note",
          "skill_id", "reuse_level"]
VERDICTS = ("right", "borderline", "wrong")


def rows_for(course: str, suggested: list[dict], seed: int) -> list[dict]:
    rows = [{"course": course, "skill": s["label"],
             "esco_definition": s.get("description", ""),
             "evidence": " | ".join(f"[{e['section']}] {e['text']}" for e in s.get("evidence", [])),
             "verdict": "", "note": "", "skill_id": s["id"], "reuse_level": s.get("reuse", "")}
            for s in suggested]
    random.Random(f"{seed}:{course}").shuffle(rows)
    for n, r in enumerate(rows, 1):
        r["row"] = n
    return rows


def read_sheet(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["verdict"] = (r.get("verdict") or "").strip().lower()
    return rows


def precision(rows: list[dict]) -> dict:
    done = [r for r in rows if r["verdict"] in VERDICTS]
    if not done:
        return {"labelled": 0}
    n = len(done)
    count = {v: sum(r["verdict"] == v for r in done) for v in VERDICTS}
    return {"labelled": n, **count, "strict": count["right"] / n,
            "lenient": (count["right"] + count["borderline"]) / n}


def bootstrap(rows: list[dict], key: str, n_boot: int = 2000, seed: int = 0):
    courses = sorted({r["course"] for r in rows})
    by = {c: [r for r in rows if r["course"] == c] for c in courses}
    rng = np.random.default_rng(seed)
    vals = [precision([r for c in rng.choice(courses, len(courses)) for r in by[c]]).get(key)
            for _ in range(n_boot)]
    vals = [v for v in vals if v is not None]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def score(path: str) -> dict:
    rows = read_sheet(path)
    bad = sorted({r["verdict"] for r in rows if r["verdict"] and r["verdict"] not in VERDICTS})
    if bad:
        raise SystemExit(f"unknown verdicts {bad}; use {', '.join(VERDICTS)}")
    out = {}
    groups = {"all": rows,
              "course-specific": [r for r in rows if r["reuse_level"] != "transversal"],
              "transversal": [r for r in rows if r["reuse_level"] == "transversal"]}
    for name, group in groups.items():
        p = precision(group)
        if p["labelled"]:
            p["strict_ci95"] = bootstrap(group, "strict")
            p["lenient_ci95"] = bootstrap(group, "lenient")
        out[name] = p
    out["per_course"] = {c: precision([r for r in rows if r["course"] == c])
                         for c in sorted({r["course"] for r in rows})}
    out["unlabelled"] = sum(1 for r in rows if not r["verdict"])
    return out


def write_sheet(path: str, rows: list[dict]) -> None:
    tmp = Path(path).with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    tmp.replace(path)                         # never leaves a half-written sheet


def _getch() -> str:
    """One key press, without Enter (falls back to a line of input)."""
    import sys
    try:
        import termios, tty
        fd = sys.stdin.fileno(); old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd); return sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
    except Exception:
        return (input("> ").strip() or " ")[0]


def _label(args) -> None:
    import shutil
    import textwrap
    rows = read_sheet(args.sheet)
    todo = [i for i, r in enumerate(rows)
            if args.per_course is None or int(r["row"]) <= args.per_course]
    todo.sort(key=lambda i: (rows[i]["course"], int(rows[i]["row"])))
    pos = next((k for k, i in enumerate(todo) if not rows[i]["verdict"]), len(todo))
    width = min(100, shutil.get_terminal_size().columns - 2)
    wrap = lambda t, ind="  ": "\n".join(textwrap.wrap(t, width, initial_indent=ind,
                                                       subsequent_indent=ind)) or ind + "—"
    keys = {"1": "right", "2": "borderline", "3": "wrong"}
    while 0 <= pos < len(todo):
        r = rows[todo[pos]]
        done = sum(1 for i in todo if rows[i]["verdict"])
        print("\033[2J\033[H", end="")
        print(f"{pos + 1}/{len(todo)}   ({done} labelled)   course: {r['course']}, row {r['row']}\n")
        print(f"SKILL: {r['skill']}\n")
        print("ESCO definition:"); print(wrap(r["esco_definition"])); print()
        print("Course sentence(s) the judge cited:")
        for e in r["evidence"].split(" | "):
            print(wrap(e)); print()
        print(f"current: {r['verdict'] or '(none)'}\n")
        print("1 right   2 borderline   3 wrong   b back   s skip   q quit")
        k = _getch().lower()
        if k in keys:
            r["verdict"] = keys[k]; write_sheet(args.sheet, rows); pos += 1
        elif k == "b":
            pos = max(0, pos - 1)
        elif k == "s":
            pos += 1
        elif k in ("q", "\x03"):
            break
    done = sum(1 for i in todo if rows[i]["verdict"])
    print(f"\n{done}/{len(todo)} labelled, saved to {args.sheet}")


def _make(args) -> None:
    import os

    from dotenv import load_dotenv
    from langchain_deepseek import ChatDeepSeek
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from langchain_mcp_adapters.tools import load_mcp_tools

    from esco_agent.graph import parse_tool_result
    from esco_agent.mcp_client import server_config
    from esco_agent.suggest import (WHOLE_TEXT_K, gather_candidates, load_sections,
                                    propose_queries, suggest_skills)
    load_dotenv()

    async def run():
        llm = ChatDeepSeek(model=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
                           temperature=0).bind(response_format={"type": "json_object"})
        rows = []
        async with MultiServerMCPClient(server_config()).session("esco") as session:
            tools = {t.name: t for t in await load_mcp_tools(session)}

            async def link(text, k):
                return parse_tool_result(await tools["link_text"].ainvoke(
                    {"text": text, "k": k, "include_descriptions": True}))["skills"]

            for path in args.courses:
                course = load_sections(path)
                queries = await propose_queries(llm, course)
                cands = await gather_candidates(link, course, queries, whole_k=WHOLE_TEXT_K)
                result = await suggest_skills(cands, llm, course)
                rows += rows_for(Path(path).stem, result["suggested"], args.seed)
                print(f"{Path(path).stem}: {len(result['suggested'])} suggestions")
        return rows

    rows = asyncio.run(run())
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8-sig") as f:      # BOM: Excel shows accents
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
    print(f"\nwritten: {out} ({len(rows)} rows). Fill in 'verdict': right / borderline / wrong.")


def _score(args) -> None:
    res = score(args.sheet)
    for name in ("all", "course-specific", "transversal"):
        p = res[name]
        if not p["labelled"]:
            print(f"{name:<16} no labelled rows"); continue
        print(f"{name:<16} n={p['labelled']:<4} right {p['right']}, borderline {p['borderline']}, "
              f"wrong {p['wrong']} | strict {p['strict']:.3f} [{p['strict_ci95'][0]:.3f}, "
              f"{p['strict_ci95'][1]:.3f}] | lenient {p['lenient']:.3f} "
              f"[{p['lenient_ci95'][0]:.3f}, {p['lenient_ci95'][1]:.3f}]")
    print("\nper course:", json.dumps({c: (p.get("strict"), p["labelled"])
                                      for c, p in res["per_course"].items()}))
    if res["unlabelled"]:
        print(f"\n{res['unlabelled']} rows still without a verdict (not counted)")
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=2), encoding="utf-8")


def _main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make"); m.add_argument("courses", nargs="+")
    m.add_argument("--out", required=True); m.add_argument("--seed", type=int, default=0)
    lb = sub.add_parser("label"); lb.add_argument("sheet")
    lb.add_argument("--per-course", type=int, default=None)
    s = sub.add_parser("score"); s.add_argument("sheet"); s.add_argument("--json", default=None)
    args = ap.parse_args()
    {"make": _make, "label": _label, "score": _score}[args.cmd](args)


if __name__ == "__main__":
    _main()
