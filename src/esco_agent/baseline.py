"""No-tools baseline: the same LLM, asked directly for a course's ESCO skills.

The spec's baseline ("the same LLM with no tools"). For each course file, the model
gets the course text and is asked for ESCO skills with their exact preferred labels,
URIs if it knows them, and the supporting sentence. Each answer is then checked
against the ESCO data and the course text:

  label     "preferred"  the exact preferred label of an ESCO skill/knowledge concept
            "alt"        not a preferred label, but an alternative or hidden one
            "none"       not an ESCO label at all
  uri       given? a real ESCO skill URI? the same concept as the label?
  quote     is the supporting sentence actually in the course text?

The assistant passes all three by construction (its skills come from ESCO tool
results, its evidence from the numbered course sentences), so this measures what
the tools add. Intervals: 95% bootstrap over courses (courses resampled, all their
items kept), since items from one course are not independent.

Run (21 API calls for 7 courses x 3 runs):
    python -m esco_agent.baseline courses/*.json --runs 3 --out results/2026-11-20_baseline
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import re
from pathlib import Path

import numpy as np

from esco_agent import llm_calls
from esco_agent.grounding import _norm
from esco_agent.suggest import course_text, load_sections

PROMPT = """You are helping a university teacher describe a course with skills from ESCO \
(European Skills, Competences, Qualifications and Occupations), version 1.2.

Below is the course text. List the ESCO skills and knowledge concepts this course develops.

Rules:
- Use only real ESCO skills/knowledge, with their exact ESCO preferred label (English). \
Give the ESCO URI if you know it; otherwise use "unknown".
- For each, quote the sentence from the course text that supports it.
- Only include skills the text supports; don't add skills the course merely relates to.
- Aim for 5-15 items, most clearly supported first.

Answer with JSON only: {{"skills": [{{"label": "...", "uri": "...", "sentence": "..."}}]}}

<course>
{course}
</course>"""

SKILL_PREFIX = "http://data.europa.eu/esco/skill/"


class EscoLabels:
    """English labels of ESCO skill concepts: label -> {(uri, labelType)}."""

    def __init__(self, labels_parquet: Path):
        import pandas as pd
        df = pd.read_parquet(labels_parquet)
        df = df[(df["language"] == "en") & df["conceptUri"].str.startswith(SKILL_PREFIX)]
        self.by_label: dict[str, set] = {}
        for uri, kind, label in zip(df["conceptUri"], df["labelType"], df["label"]):
            self.by_label.setdefault(label.strip().lower(), set()).add((uri, kind))
        self.uris = set(df["conceptUri"])

    def label_status(self, label: str) -> tuple[str, set]:
        hits = self.by_label.get((label or "").strip().lower(), set())
        if any(kind == "preferred" for _, kind in hits):
            return "preferred", {u for u, k in hits if k == "preferred"}
        if hits:
            return "alt", {u for u, _ in hits}
        return "none", set()


def quote_found(sentence: str, course_norm: str) -> bool:
    """The sentence (or each part of it, if shortened with "..." or "…") is in the text."""
    parts = [p for p in re.split(r"\.\.\.|…", sentence or "") if len(_norm(p).split()) >= 3]
    return bool(parts) and all(_norm(p) in course_norm for p in parts)


def check(items: list[dict], esco: EscoLabels, course_norm: str) -> list[dict]:
    out = []
    for it in items:
        label = str(it.get("label", "")).strip()
        uri = str(it.get("uri", "")).strip()
        status, concepts = esco.label_status(label)
        given = uri.lower() not in ("", "unknown", "n/a", "none")
        out.append({"label": label, "label_status": status, "uri": uri,
                    "uri_given": given, "uri_real": given and uri in esco.uris,
                    "uri_matches_label": given and uri in concepts,
                    "quote_found": quote_found(str(it.get("sentence", "")), course_norm)})
    return out


def summarise(rows: list[dict]) -> dict:
    n = len(rows)
    if not n:
        return {"items": 0}
    share = lambda f: sum(1 for r in rows if f(r)) / n
    given = [r for r in rows if r["uri_given"]]
    return {"items": n,
            "label_preferred": share(lambda r: r["label_status"] == "preferred"),
            "label_any": share(lambda r: r["label_status"] != "none"),
            "quote_found": share(lambda r: r["quote_found"]),
            "uri_given": len(given) / n,
            "uri_correct_of_given": (sum(r["uri_matches_label"] for r in given) / len(given)
                                     if given else None)}


def bootstrap_courses(rows: list[dict], key: str, n_boot: int, seed: int) -> tuple:
    courses = sorted({r["course"] for r in rows})
    by = {c: [r for r in rows if r["course"] == c] for c in courses}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = [r for c in rng.choice(courses, len(courses)) for r in by[c]]
        v = summarise(pick).get(key)
        if v is not None:
            vals.append(v)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


async def ask(llm, sections: dict) -> list[dict]:
    reply = await llm_calls.call(llm, PROMPT.format(course=course_text(sections)), "baseline")
    try:
        items = json.loads(reply.content).get("skills", [])
    except (json.JSONDecodeError, AttributeError):
        return []
    return [i for i in items if isinstance(i, dict)]


def _main() -> None:
    from dotenv import load_dotenv

    from esco_agent.assistant import deepseek_models
    from esco_core.linker import data_dir

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("courses", nargs="+", help="course files (.json sections or .txt)")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="write items (.csv) and summary (.json) here")
    args = ap.parse_args()
    load_dotenv()

    esco = EscoLabels(data_dir() / "labels.parquet")
    model, _, json_llm = deepseek_models()

    async def run_all():
        rows = []
        for path in args.courses:
            sections = load_sections(path)
            hay = _norm("\n".join(sections.values()))
            for run in range(1, args.runs + 1):
                for r in check(await ask(json_llm, sections), esco, hay):
                    rows.append({"course": Path(path).stem, "run": run, **r})
        return rows

    rows = asyncio.run(run_all())
    print(f"model: {model} | {len(args.courses)} courses x {args.runs} runs | "
          f"{len(rows)} items\n")
    print(f"{'course':<40} {'items/run':>9} {'ESCO pref.':>10} {'any label':>9} "
          f"{'quote ok':>8} {'URI given':>9} {'URI right':>9}")
    for c in sorted({r["course"] for r in rows}):
        s = summarise([r for r in rows if r["course"] == c])
        right = "-" if s["uri_correct_of_given"] is None else f"{s['uri_correct_of_given']:.2f}"
        print(f"{c:<40} {s['items'] / args.runs:>9.1f} {s['label_preferred']:>10.2f} "
              f"{s['label_any']:>9.2f} {s['quote_found']:>8.2f} {s['uri_given']:>9.2f} {right:>9}")
    overall = summarise(rows)
    summary = {"model": model, "runs": args.runs, "courses": len(args.courses),
               "overall": overall, "ci95_over_courses": {}}
    print("\noverall (95% bootstrap over courses):")
    for key in ("label_preferred", "label_any", "quote_found", "uri_given", "uri_correct_of_given"):
        if overall.get(key) is None:
            continue
        lo, hi = bootstrap_courses(rows, key, args.boot, args.seed)
        summary["ci95_over_courses"][key] = [lo, hi]
        print(f"  {key:<22} {overall[key]:.3f} [{lo:.3f}, {hi:.3f}]")
    print("\nnot ESCO labels:", sorted({r["label"] for r in rows if r["label_status"] == "none"})[:25])

    if args.out:
        out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
        with open(out.with_suffix(".csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        out.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nwritten: {out.with_suffix('.csv')}, {out.with_suffix('.json')}")


if __name__ == "__main__":
    _main()
