"""The occupation smoke test's own logic. The run itself is the evaluation; these
check the ranking, ties and intervals, and that its TF-IDF ranking is the one the
assistant uses. The last test needs data/ and is skipped without it."""

import numpy as np
import pytest

from esco_core.linker import data_dir
from esco_core.occupations import OccupationIndex
from esco_core.smoke_occupations import bootstrap, metrics, rank_of, report, run


def test_rank_counts_ties_as_half():
    s = np.array([0.9, 0.5, 0.5, 0.5, 0.1])
    assert rank_of(s, 0) == 1.0
    assert rank_of(s, 1) == 3.0            # one higher, two equal others: 1 + 1 + 1
    assert rank_of(s, 4) == 5.0


def test_metrics_and_bootstrap():
    ranks = np.array([1.0, 2.0, 4.0, 20.0])
    m = metrics(ranks)
    assert m == {"rank1": 0.25, "top3": 0.5, "top10": 0.75,
                 "mrr": pytest.approx((1 + 0.5 + 0.25 + 0.05) / 4)}
    lo, hi = bootstrap(ranks, lambda x: metrics(x)["mrr"], 500, np.random.default_rng(0))
    assert 0 < lo <= m["mrr"] <= hi <= 1


def test_tfidf_rank_matches_the_assistants_search():
    if not (data_dir() / "occupations.parquet").exists():
        pytest.skip(f"no data in {data_dir()}")
    index = OccupationIndex(data_dir())
    rows = run(index, seed=0)
    assert len(rows) > 100 and all(r["n_query"] == r["n_essential"] // 2 for r in rows)
    # same seed, same result
    assert [r["tfidf"] for r in run(index, seed=0)[:20]] == [r["tfidf"] for r in rows[:20]]
    # where the script says rank 1, the assistant's own search returns it first
    firsts = [r for r in rows if r["tfidf"] == 1.0][:10]
    for r in firsts:
        hits = index.search_occupations({u: 1.0 for u in r["query"].split(";")}, k=1)
        assert hits[0].uri == r["occupation"]
    res = report(rows[:200], n_boot=100, seed=0)
    assert set(res["scorers"]) == {"tfidf", "overlap", "coverage"}
