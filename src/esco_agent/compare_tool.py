"""Compare the ESCO Open Access Mapping tool (European Commission, 2026) with this
project's skill linker, line by line, and with the agent's checked list.

The tool splits a course text into lines, marks some as noise, and returns ranked
ESCO candidates for each other line; the user picks. Its results for the 6 study
courses were collected by hand from the web tool (English, Top K 15, its own split;
top 5 kept per line) into results/esco-tool/<course>.json.

Retriever against retriever (fair): the linker gets exactly the same lines, and the
top 3 of both are labelled blind (one sheet, preferred ESCO labels only, shuffled,
the source kept in a separate key file). The tool's top 3 is its first 3 candidates as
shown; two of them meaning one concept count once, at the better rank. Then precision
at 1 and at 3 per system, the paired difference (tool minus linker, same lines), and the
same with the tool's labels that are not English ESCO skills counted as wrong.

    python -m esco_agent.compare_tool sheet results/esco-tool/*.json \\
        --out results/2026-11-26_tool_vs_linker.csv
    python -m esco_agent.labelsheet label results/2026-11-26_tool_vs_linker.csv
    python -m esco_agent.compare_tool score results/2026-11-26_tool_vs_linker.csv
    # a sheet labelled before the top-3 rule changed: recompute the tool's ranks
    python -m esco_agent.compare_tool rekey results/esco-tool/*.json \
        --key results/2026-11-26_tool_vs_linker.key.json --out results/2026-11-26_tool_vs_linker.top3.key.json
    python -m esco_agent.compare_tool coverage results/esco-tool/*.json \\
        --agent results/2026-11-20_labels.csv

Labelling rule for this sheet: right if the line supports the skill (for activities:
students are shown doing it), borderline if the topic is named or only partly
covered, wrong if the words match in another sense or the line doesn't support it.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from esco_agent.labelsheet import VERDICTS, read_sheet, write_sheet

TOP = 3                     # candidates per line and system that are labelled


def load_tool(paths: list[str]) -> list[dict]:
    return [json.loads(Path(p).read_text(encoding="utf-8")) for p in sorted(paths)]


def resolve(label: str, esco) -> tuple[str | None, str]:
    """A label the tool shows (often an alternative or hidden one, even misspelt:
    "data anlysis") -> one ESCO concept. Preferred labels first; an alternative label
    shared by several concepts is ambiguous and resolves to the first in order."""
    status, uris = esco.label_status(label)
    if not uris:
        return None, "unknown"
    return sorted(uris)[0], ("preferred" if status == "preferred" else
                             "alt" if len(uris) == 1 else "alt, ambiguous")


def build_rows(tool: list[dict], linker, esco, seed: int = 0) -> tuple[list[dict], list[dict]]:
    """One labelling row per (course, line, concept) in either system's top 3, shuffled
    within each course; and a key saying which system ranked it where."""
    rows, key = [], []
    rng = random.Random(seed)
    for course in tool:
        name, items = course["course"], []
        for i, line in enumerate(l for l in course["lines"] if not l["noise"]):
            found: dict[str, dict] = {}
            for rank, c in enumerate(line["candidates"][:TOP], 1):
                uri, how = resolve(c["label"], esco)
                entry = found.setdefault(uri or f"unknown:{c['label']}", {"tool": None, "linker": None,
                                                                            "tool_label": c["label"], "how": how})
                if entry["tool"] is None:
                    entry["tool"] = rank
            for rank, hit in enumerate(linker.link(line["line"], k=TOP), 1):
                entry = found.setdefault(hit.uri, {"tool": None, "linker": None, "tool_label": "", "how": ""})
                if entry["linker"] is None:
                    entry["linker"] = rank
            for uri, e in found.items():
                known = not uri.startswith("unknown:")
                items.append(({"course": name, "skill": linker.label.get(uri, e["tool_label"]) if known
                               else e["tool_label"],
                               "esco_definition": (linker.description.get(uri) or "") if known
                               else "(not an English ESCO skill label)",
                               "evidence": line["line"], "verdict": "", "note": "",
                               "skill_id": uri, "reuse_level": linker.reuse_level.get(uri, "") if known else ""},
                              {"course": name, "line": i, "skill_id": uri, "tool_rank": e["tool"],
                               "linker_rank": e["linker"], "tool_label": e["tool_label"], "resolved": e["how"]}))
        rng.shuffle(items)
        for n, (row, k) in enumerate(items, 1):
            row["row"] = n; k["row"] = n
            rows.append(row); key.append(k)
    return rows, key


def tool_top(candidates: list[dict], esco, depth: int = TOP) -> dict[str, int]:
    """The tool's first `depth` candidates as shown -> concept: its best rank. Two labels
    meaning one concept count once; a label that is no English ESCO skill is "unknown:<label>"."""
    out: dict[str, int] = {}
    for rank, c in enumerate(candidates[:depth], 1):
        out.setdefault(resolve(c["label"], esco)[0] or f"unknown:{c['label']}", rank)
    return out


def rekey(key: list[dict], tool: list[dict], esco) -> list[dict]:
    """The key of an already labelled sheet with the tool's ranks recomputed by tool_top;
    rows and labels are kept. A concept the new top 3 needs but the sheet lacks is an error."""
    out = [dict(k, tool_rank=None) for k in key]
    by = {(k["course"], k["line"], k["skill_id"]): k for k in out}
    for course in tool:
        for i, line in enumerate(l for l in course["lines"] if not l["noise"]):
            for uri, rank in tool_top(line["candidates"], esco).items():
                k = by.get((course["course"], i, uri))
                if k is None:
                    raise ValueError(f"not on the sheet, needs labelling: {course['course']} line {i} {uri}")
                k["tool_rank"] = rank
    return out


def metrics(sample: list[dict], unknown_wrong: bool = False) -> dict:
    """P@1 and P@3 per system, strict and lenient, and tool minus linker on the same lines.
    unknown_wrong: the tool's labels that are no English ESCO skill count as wrong."""
    out = {}
    for name, ok in (("strict", {"right"}), ("lenient", {"right", "borderline"})):
        hit = lambda vu: vu[0] in ok and not (unknown_wrong and vu[1])
        for s in ("tool", "linker"):
            p1 = [hit(l[s][1]) for l in sample]
            p3 = [hit(vu) for l in sample for r, vu in l[s].items() if r <= TOP]
            out[f"{s} P@1 {name}"] = float(np.mean(p1)) if p1 else float("nan")
            out[f"{s} P@3 {name}"] = float(np.mean(p3)) if p3 else float("nan")
        for at in ("P@1", "P@3"):
            out[f"tool - linker {at} {name}"] = out[f"tool {at} {name}"] - out[f"linker {at} {name}"]
    return out


def score(rows: list[dict], key: list[dict], n_boot: int = 2000, seed: int = 0) -> dict:
    """Metrics on the lines where both systems' rank 1 is labelled, with 95% bootstrap
    intervals over lines; each draw resamples lines once, so differences are paired."""
    verdict = {(r["course"], int(r["row"])): r["verdict"] for r in rows}
    lines: dict[tuple, dict] = {}
    for k in key:
        v = verdict.get((k["course"], int(k["row"])), "")
        if v not in VERDICTS:
            continue
        ln = lines.setdefault((k["course"], k["line"]), {"tool": {}, "linker": {}})
        for system in ("tool", "linker"):
            if k[f"{system}_rank"]:
                ln[system][k[f"{system}_rank"]] = (v, k["skill_id"].startswith("unknown:"))
    full = [l for l in lines.values() if all(1 in l[s] for s in ("tool", "linker"))]
    rng = np.random.default_rng(seed)
    draws = [[full[i] for i in rng.integers(0, len(full), len(full))] for _ in range(n_boot)]
    res: dict = {"lines": len(full)}
    for name, unknown_wrong in (("as labelled", False), ("unknown tool labels as wrong", True)):
        point = metrics(full, unknown_wrong)
        boot = [metrics(d, unknown_wrong) for d in draws]
        res[name] = {m: {"value": v, "ci95": [float(np.percentile([b[m] for b in boot], 2.5)),
                                              float(np.percentile([b[m] for b in boot], 97.5))]}
                     for m, v in point.items()}
    return res


def short(uri: str) -> str:
    return uri.rsplit("/", 1)[-1]


def cid(uri: str) -> str:
    """Concept id across the sheets: the end of an ESCO URI (as the agent's sheet has it),
    or "unknown:<label>" for a tool label that is no English ESCO skill."""
    return uri if uri.startswith("unknown:") else short(uri)


def ratio(a: float, b: float) -> float | None:
    return a / b if b else None


BEST = {"right": 2, "borderline": 1, "wrong": 0}


def best_verdicts(labelled: list[dict]) -> dict:
    """(course, concept id) -> the best verdict a concept got on any line of the blind
    tool-vs-linker sheet (a skill counts if some line of the course supports it)."""
    out = {}
    for r in labelled:
        if r["verdict"] in BEST:
            k = (r["course"], cid(r["skill_id"]))
            if k not in out or BEST[r["verdict"]] > BEST[out[k]]:
                out[k] = r["verdict"]
    return out


def coverage(tool: list[dict], agent_rows: list[dict], esco, labelled: list[dict] | None = None) -> dict:
    """What a teacher reviews with each, how far the two overlap per course and, with
    the labelled blind sheet, how many skills only the tool has are right: skills the
    agent missed that a retriever found and a rater confirmed. The tool's labels that are
    no English ESCO skill count as tool skills (by name), as in score; TOTAL adds relative
    recall of the agent (also with them counted wrong, with the tool's ranks 4-5, and
    with partly right counted) and reviews per fully right skill."""
    best = best_verdicts(labelled or [])
    count = lambda name, ids: {v: sum(best.get((name, u), "") == v for u in ids) for v in BEST} | {
        "unlabelled": sum((name, u) not in best for u in ids)}
    out = {}
    for course in tool:
        name = course["course"]
        lines = [l for l in course["lines"] if not l["noise"]]
        top = {d: {cid(u) for l in lines for u in tool_top(l["candidates"], esco, d)} for d in (1, TOP, 5)}
        agent = {short(r["skill_id"]): r["verdict"] for r in agent_rows if r["course"] == name}
        c = {"lines": len(course["lines"]), "noise_lines": len(course["lines"]) - len(lines),
             "tool_candidates_to_review_top_k": len(lines) * course["settings"]["top_k"],
             "tool_distinct_in_top3": len(top[TOP]), "agent_suggestions": len(agent),
             "both": len(top[TOP] & set(agent)), "only_tool_top3": len(top[TOP] - set(agent)),
             "only_agent": len(agent.keys() - top[TOP]), "only_agent_in_tool_ranks_4_5": len(agent.keys() & top[5] - top[TOP])}
        if labelled is not None:
            only3 = top[TOP] - set(agent)
            c["agent_labelled"] = {v: sum(x == v for x in agent.values()) for v in BEST}
            c["only_tool_top3_labelled"] = count(name, only3)
            c["only_tool_top3_unknown_labelled"] = count(name, {u for u in only3 if u.startswith("unknown:")})
            c["only_tool_top5_labelled"] = count(name, top[5] - set(agent))
            c["tool_rank1_right"] = count(name, top[1])["right"]
            c["tool_top3_right"] = count(name, top[TOP])["right"]
            c["tool_top3_right_unknown"] = count(name, {u for u in top[TOP] if u.startswith("unknown:")})["right"]
        out[name] = c
    if out:
        total = {}
        for c in out.values():
            for k, v in c.items():
                total[k] = ({x: total.get(k, {}).get(x, 0) + n for x, n in v.items()} if isinstance(v, dict)
                            else total.get(k, 0) + v)
        if labelled is not None:
            a, t3, u3, t5 = (total["agent_labelled"], total["only_tool_top3_labelled"],
                             total["only_tool_top3_unknown_labelled"], total["only_tool_top5_labelled"])
            ar, aok = a["right"], a["right"] + a["borderline"]
            total["relative_recall"] = {
                "strict, tool top 3": ratio(ar, ar + t3["right"]),
                "strict, tool top 3, unknown tool labels as wrong": ratio(ar, ar + t3["right"] - u3["right"]),
                "partly right counted, tool top 3": ratio(aok, aok + t3["right"] + t3["borderline"]),
                "strict, tool top 5, unlabelled as not right (upper bound)": ratio(ar, ar + t5["right"]),
                "strict, tool top 5, unlabelled as right (lower bound)": ratio(ar, ar + t5["right"] + t5["unlabelled"]),
            }
            n_lines = total["lines"] - total["noise_lines"]
            total["reviews_per_right_skill"] = {
                "agent": ratio(total["agent_suggestions"], ar),
                "tool K=1": ratio(n_lines, total["tool_rank1_right"]),
                "tool K=3": ratio(TOP * n_lines, total["tool_top3_right"]),
                "tool K=3, unknown tool labels as wrong": ratio(TOP * n_lines, total["tool_top3_right"] - total["tool_top3_right_unknown"]),
            }
        out["TOTAL"] = total
    return out


def _main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sheet"); s.add_argument("tool", nargs="+"); s.add_argument("--out", required=True)
    s.add_argument("--seed", type=int, default=0)
    c = sub.add_parser("score"); c.add_argument("sheet"); c.add_argument("--json")
    c.add_argument("--key", help="default: the sheet's .key.json")
    r = sub.add_parser("rekey"); r.add_argument("tool", nargs="+"); r.add_argument("--key", required=True)
    r.add_argument("--out", required=True)
    v = sub.add_parser("coverage"); v.add_argument("tool", nargs="+"); v.add_argument("--agent", required=True)
    v.add_argument("--labels", help="the labelled blind tool-vs-linker sheet: counts the "
                                    "confirmed skills only the tool's top 3 has")
    v.add_argument("--json")
    a = ap.parse_args()
    if a.cmd == "sheet":
        from esco_agent.baseline import EscoLabels
        from esco_core.linker import SkillLinker, data_dir
        linker = SkillLinker()
        rows, key = build_rows(load_tool(a.tool), linker, EscoLabels(data_dir() / "labels.parquet"), a.seed)
        write_sheet(a.out, rows)
        Path(a.out).with_suffix(".key.json").write_text(json.dumps(key, indent=1), encoding="utf-8")
        unknown = sum(k["resolved"] == "unknown" for k in key)
        print(f"{a.out}: {len(rows)} rows to label ({unknown} tool labels not found in ESCO). "
              f"The key (which system ranked what) is in {Path(a.out).with_suffix('.key.json')}: "
              f"don't open it before labelling.")
    elif a.cmd == "rekey":
        from esco_agent.baseline import EscoLabels
        from esco_core.linker import data_dir
        key = json.loads(Path(a.key).read_text(encoding="utf-8"))
        new = rekey(key, load_tool(a.tool), EscoLabels(data_dir() / "labels.parquet"))
        Path(a.out).write_text(json.dumps(new, indent=1), encoding="utf-8")
        moved = sum(k["tool_rank"] != n["tool_rank"] for k, n in zip(key, new))
        print(f"{a.out}: {moved} of {len(key)} tool ranks changed")
    elif a.cmd == "score":
        key = json.loads(Path(a.key or Path(a.sheet).with_suffix(".key.json")).read_text(encoding="utf-8"))
        res = score(read_sheet(a.sheet), key)
        print(f"{res['lines']} lines with both systems' top candidate labelled")
        for part in ("as labelled", "unknown tool labels as wrong"):
            print(part)
            for m, r in res[part].items():
                print(f"  {m:30} {r['value']:+.3f} [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]")
        if a.json:
            Path(a.json).write_text(json.dumps(res, indent=2), encoding="utf-8")
    else:
        from esco_agent.baseline import EscoLabels
        from esco_core.linker import data_dir
        res = coverage(load_tool(a.tool), read_sheet(a.agent), EscoLabels(data_dir() / "labels.parquet"),
                       read_sheet(a.labels) if a.labels else None)
        for name, r in res.items():
            print(name, r)
        if a.json:
            Path(a.json).write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    _main()
