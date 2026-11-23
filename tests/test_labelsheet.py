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


def test_chance_corrected_matches_hand_computed_values():
    from esco_agent.labelsheet import chance_corrected
    # 10 items: both "right" 7 times, both "wrong" once, 2 disagreements
    a = ["right"] * 8 + ["wrong"] * 2
    b = ["right"] * 7 + ["wrong"] + ["wrong"] + ["right"]
    r = chance_corrected(a, b, ("right", "wrong"))
    assert r["agreement"] == pytest.approx(0.8)
    # kappa: chance = .8*.8 + .2*.2 = .68 -> (.8-.68)/.32
    assert r["kappa"] == pytest.approx(0.12 / 0.32)
    # AC1: pi = .8/.2, chance = 2*.8*.2/(2-1) = .32 -> (.8-.32)/.68
    assert r["ac1"] == pytest.approx(0.48 / 0.68)
    assert r["ac1"] > r["kappa"]                 # the kappa paradox: one verdict dominates


def test_agree_matches_rows_by_course_and_skill(tmp_path):
    from esco_agent import labelsheet
    a, b = rows_for("c1", SUGG, 0), rows_for("c1", SUGG, 1)      # same rows, other order
    va = {"skill 0": "right", "skill 1": "right", "skill 2": "wrong", "skill 3": ""}
    vb = {"skill 0": "right", "skill 1": "borderline", "skill 2": "wrong", "skill 3": "right"}
    for r in a: r["verdict"] = va[r["skill"]]
    for r in b: r["verdict"] = vb[r["skill"]]
    pa, pb = tmp_path / "a.csv", tmp_path / "b.csv"
    labelsheet.write_sheet(str(pa), a); labelsheet.write_sheet(str(pb), b)
    res = labelsheet.agree(str(pa), str(pb), n_boot=50)
    assert res["n"] == 3 and res["only_in_b"] == 1
    assert res["three verdicts"]["agreement"] == pytest.approx(2 / 3)
    assert res["lenient (wrong vs not)"]["agreement"] == 1.0
    assert res["confusion"]["right"]["borderline"] == 1
    assert [d["skill"] for d in res["disagreements"]] == ["skill 1"]
