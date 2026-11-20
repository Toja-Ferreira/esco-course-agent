"""The no-tools baseline's checks (no API calls)."""

import pandas as pd

from esco_agent.baseline import EscoLabels, bootstrap_courses, check, quote_found, summarise
from esco_agent.grounding import _norm

S = "http://data.europa.eu/esco/skill/"


def esco(tmp_path):
    pd.DataFrame({
        "conceptUri": [S + "prog", S + "prog", S + "os", S + "os", "http://data.europa.eu/esco/occupation/x"],
        "language": ["en"] * 5,
        "labelType": ["preferred", "alt", "preferred", "hidden", "preferred"],
        "label": ["computer programming", "coding", "operating systems", "Linux", "software developer"],
    }).to_parquet(tmp_path / "labels.parquet")
    return EscoLabels(tmp_path / "labels.parquet")


def test_labels_uris_and_quotes_are_checked(tmp_path):
    e = esco(tmp_path)
    hay = _norm("Students write programs in C. They use the Unix/Linux command line every week.")
    rows = check([
        {"label": "Computer Programming", "uri": S + "prog", "sentence": "Students write programs in C."},
        {"label": "Linux", "uri": "unknown", "sentence": "They use the Unix/Linux command line…"},
        {"label": "structured programming", "uri": S + "os", "sentence": "Students master pointers."},
        {"label": "software developer", "uri": "", "sentence": ""},   # an occupation, not a skill
    ], e, hay)
    assert [r["label_status"] for r in rows] == ["preferred", "alt", "none", "none"]
    assert [r["uri_given"] for r in rows] == [True, False, True, False]
    assert rows[0]["uri_matches_label"] and rows[2]["uri_real"] and not rows[2]["uri_matches_label"]
    assert [r["quote_found"] for r in rows] == [True, True, False, False]
    s = summarise(rows)
    assert s["label_preferred"] == 0.25 and s["label_any"] == 0.5 and s["uri_correct_of_given"] == 0.5


def test_quote_parts_and_bootstrap_over_courses():
    hay = _norm("Alpha beta gamma delta. Epsilon zeta eta theta.")
    assert quote_found("Alpha beta gamma ... zeta eta theta", hay)
    assert not quote_found("Alpha beta", hay)                      # too short to count
    rows = [{"course": c, "label_status": st, "quote_found": True, "uri_given": False,
             "uri_matches_label": False}
            for c, st in [("a", "preferred"), ("a", "none"), ("b", "preferred"), ("c", "none")]]
    lo, hi = bootstrap_courses(rows, "label_preferred", 300, 0)
    assert 0 <= lo <= 0.5 <= hi <= 1
