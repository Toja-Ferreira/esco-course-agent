import json
from types import SimpleNamespace

import pytest

from esco_agent import compare_tool as ct
from esco_agent.labelsheet import read_sheet, write_sheet


class Labels:
    """label -> concepts, like baseline.EscoLabels."""
    table = {"use html": ("preferred", {"u:html"}), "data anlysis": ("alt", {"u:da"}),
             "programming": ("alt", {"u:prog", "u:prog2"}),
             "web programming": ("preferred", {"http://data.europa.eu/esco/skill/web"})}

    def label_status(self, label):
        return self.table.get(label.lower(), ("none", set()))


class Linker:
    label = {"u:html": "use markup languages", "u:da": "data analysis", "u:prog": "computer programming",
             "http://data.europa.eu/esco/skill/web": "web programming", "u:css": "CSS"}
    description = {u: f"definition of {l}" for u, l in label.items()}
    reuse_level = {}

    def link(self, text, k):
        return [SimpleNamespace(uri=u) for u in ["http://data.europa.eu/esco/skill/web", "u:css", "u:html"][:k]]


WEB = "http://data.europa.eu/esco/skill/web"
# shown: use HTML, use html (same concept), JSSS (no ESCO skill), web programming (rank 4)
TOOL = [{"course": "web", "settings": {"top_k": 15}, "lines": [
    {"line": "Contents", "noise": True, "candidates": []},
    {"line": "3.1 HTML", "noise": False, "candidates": [
        {"label": "use HTML"}, {"label": "use html"}, {"label": "JSSS"}, {"label": "web programming"}]}]}]


def test_tool_top3_is_the_first_three_shown_one_concept_once():
    assert ct.resolve("data anlysis", Labels()) == ("u:da", "alt")
    assert ct.resolve("Programming", Labels()) == ("u:prog", "alt, ambiguous")
    assert ct.resolve("JSSS", Labels()) == (None, "unknown")
    # the repeat of "use HTML" does not pull rank 4 (web programming) into the top 3
    assert ct.tool_top(TOOL[0]["lines"][1]["candidates"], Labels()) == {"u:html": 1, "unknown:JSSS": 3}
    assert ct.tool_top(TOOL[0]["lines"][1]["candidates"], Labels(), 5)[WEB] == 4
    rows, key = ct.build_rows(TOOL, Linker(), Labels())
    by = {k["skill_id"]: k for k in key}
    assert by["u:html"]["tool_rank"] == 1 and by["u:html"]["linker_rank"] == 3
    assert by[WEB]["tool_rank"] is None and by[WEB]["linker_rank"] == 1
    assert by["u:css"]["tool_rank"] is None and by["unknown:JSSS"]["linker_rank"] is None
    assert len(rows) == 4                                    # one row per concept, not per system
    skills = {r["skill"] for r in rows}
    assert "use markup languages" in skills and "use HTML" not in skills   # preferred label: blind
    assert all(r["evidence"] == "3.1 HTML" and r["verdict"] == "" for r in rows)


def test_rekey_recomputes_tool_ranks_of_a_labelled_sheet():
    _, key = ct.build_rows(TOOL, Linker(), Labels())
    old = [dict(k, tool_rank={"u:html": 1, "unknown:JSSS": 2, WEB: 3}.get(k["skill_id"])) for k in key]
    new = {k["skill_id"]: k["tool_rank"] for k in ct.rekey(old, TOOL, Labels())}
    assert new == {"u:html": 1, "unknown:JSSS": 3, WEB: None, "u:css": None}
    with pytest.raises(ValueError):                          # a top-3 concept with no row to label
        ct.rekey([k for k in old if k["skill_id"] != "unknown:JSSS"], TOOL, Labels())


def _scored(tmp_path, verdict):
    rows, key = ct.build_rows(TOOL, Linker(), Labels())
    for r in rows:
        r["verdict"] = verdict[r["skill_id"]]
    p = tmp_path / "s.csv"
    write_sheet(str(p), rows)
    return ct.score(read_sheet(str(p)), key, n_boot=20)


def test_score_precision_paired_difference_and_unknown_sensitivity(tmp_path):
    res = _scored(tmp_path, {"u:html": "right", WEB: "borderline", "u:css": "wrong", "unknown:JSSS": "right"})
    assert res["lines"] == 1
    m = res["as labelled"]
    assert m["tool P@1 strict"]["value"] == 1.0              # use HTML
    assert m["linker P@1 strict"]["value"] == 0.0            # web programming: borderline
    assert m["linker P@1 lenient"]["value"] == 1.0
    assert m["tool P@3 strict"]["value"] == 1.0              # use HTML and JSSS: 2 concepts in 3 slots
    assert abs(m["linker P@3 strict"]["value"] - 1 / 3) < 1e-9
    assert abs(m["tool - linker P@3 strict"]["value"] - 2 / 3) < 1e-9
    assert len(m["tool - linker P@1 strict"]["ci95"]) == 2
    s = res["unknown tool labels as wrong"]
    assert s["tool P@3 strict"]["value"] == 0.5 and s["linker P@3 strict"] == m["linker P@3 strict"]


def test_metrics_differences_are_paired_per_line():
    lines = [{"tool": {1: ("right", False)}, "linker": {1: ("wrong", False)}},
             {"tool": {1: ("wrong", False)}, "linker": {1: ("right", False)}}]
    m = ct.metrics(lines)
    assert m["tool - linker P@1 strict"] == 0.0 and m["tool P@1 strict"] == 0.5


def test_coverage_counts_review_load_and_overlap():
    # the agent's ids are short (the end of the URI), the tool's resolve to full URIs
    agent = [{"course": "web", "skill_id": "u:html", "verdict": ""}, {"course": "web", "skill_id": "web", "verdict": ""}]
    r = ct.coverage(TOOL, agent, Labels())["web"]
    assert r["tool_candidates_to_review_top_k"] == 15 and r["noise_lines"] == 1
    # tool top 3: u:html, unknown:JSSS; "web" is only at rank 4
    assert (r["both"], r["only_tool_top3"], r["only_agent"], r["only_agent_in_tool_ranks_4_5"]) == (1, 1, 1, 1)


def test_coverage_recall_variants_and_reviews_per_right_skill():
    tool = [{**TOOL[0], "lines": TOOL[0]["lines"] + [
        {"line": "3.2 CSS", "noise": False, "candidates": [{"label": "data anlysis"}]}]}]
    agent = [{"course": "web", "skill_id": "u:html", "verdict": "right"},
             {"course": "web", "skill_id": "other", "verdict": "borderline"}]
    labelled = [{"course": "web", "skill_id": "u:html", "verdict": "right"},
                {"course": "web", "skill_id": "unknown:JSSS", "verdict": "right"},
                {"course": "web", "skill_id": "u:da", "verdict": "wrong"},
                {"course": "web", "skill_id": "u:da", "verdict": "borderline"}]  # best over lines
    t = ct.coverage(tool, agent, Labels(), labelled)["TOTAL"]
    assert t["only_tool_top3_labelled"] == {"right": 1, "borderline": 1, "wrong": 0, "unlabelled": 0}
    assert t["only_tool_top5_labelled"]["unlabelled"] == 1     # web programming, rank 4, never labelled
    rec = t["relative_recall"]
    assert rec["strict, tool top 3"] == 1 / 2                  # agent 1 right, JSSS right
    assert rec["strict, tool top 3, unknown tool labels as wrong"] == 1.0
    assert rec["partly right counted, tool top 3"] == 2 / 4
    assert rec["strict, tool top 5, unlabelled as not right (upper bound)"] == 1 / 2
    assert rec["strict, tool top 5, unlabelled as right (lower bound)"] == 1 / 3
    load = t["reviews_per_right_skill"]
    assert load["agent"] == 2.0 and load["tool K=1"] == 2 / 1 and load["tool K=3"] == 6 / 2
    assert load["tool K=3, unknown tool labels as wrong"] == 6.0
