"""Labelling sheet: rows, shuffling and scoring (no API calls)."""

import csv

import pytest

from esco_agent.labelsheet import FIELDS, rows_for, score

SUGG = [{"id": f"s{i}", "label": f"skill {i}", "description": "d", "reuse": "transversal" if i == 0 else "",
         "evidence": [{"section": "Contents", "text": f"sentence {i}"}]} for i in range(4)]


def test_rows_are_shuffled_reproducibly_and_numbered():
    a, b = rows_for("c", SUGG, 0), rows_for("c", SUGG, 0)
    assert [r["skill"] for r in a] == [r["skill"] for r in b]
    assert [r["row"] for r in a] == [1, 2, 3, 4]
    assert a[0]["evidence"].startswith("[Contents] sentence")
    assert set(a[0]) == set(FIELDS)


def test_score_counts_verdicts_and_splits_generic(tmp_path):
    rows = rows_for("c1", SUGG, 0) + rows_for("c2", SUGG, 0)
    verdicts = {"skill 0": "Right", "skill 1": "wrong", "skill 2": "borderline", "skill 3": ""}
    for r in rows:
        r["verdict"] = verdicts[r["skill"]]
    path = tmp_path / "sheet.csv"
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
    res = score(str(path))
    assert res["all"]["labelled"] == 6 and res["unlabelled"] == 2
    assert res["all"]["strict"] == pytest.approx(2 / 6) and res["all"]["lenient"] == pytest.approx(4 / 6)
    assert res["transversal"]["strict"] == 1.0 and res["course-specific"]["strict"] == 0.0


def test_label_keys_back_skip_quit_and_resume(tmp_path, monkeypatch):
    from esco_agent import labelsheet
    rows = rows_for("c1", SUGG, 0) + rows_for("c2", SUGG, 0)
    path = tmp_path / "sheet.csv"
    labelsheet.write_sheet(str(path), rows)
    args = type("A", (), {"sheet": str(path), "per_course": 2})

    keys = iter("1b3s2q")            # c1#1 right, back, c1#1 -> wrong, skip c1#2, c2#1 borderline, quit
    monkeypatch.setattr(labelsheet, "_getch", lambda: next(keys))
    labelsheet._label(args)
    got = {(r["course"], r["row"]): r["verdict"] for r in labelsheet.read_sheet(str(path))}
    assert got[("c1", "1")] == "wrong" and got[("c1", "2")] == "" and got[("c2", "1")] == "borderline"
    assert got[("c1", "3")] == ""                              # outside --per-course 2

    keys = iter("1q")                                          # resume: first unlabelled is c1#2
    monkeypatch.setattr(labelsheet, "_getch", lambda: next(keys))
    labelsheet._label(args)
    assert labelsheet.read_sheet(str(path))[[r["row"] for r in labelsheet.read_sheet(str(path))].index("2")]["verdict"] in ("right",)
