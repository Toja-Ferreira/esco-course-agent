"""Smoke test for occupation search: can an occupation be found from half of its
essential skills?

For every occupation with at least MIN_ESSENTIAL essential skills, a random half of
them (rounded down, at least 1) is used as the query, each with weight 1. We record
where the occupation itself ranks among all 3,039 under three scorers:

  tfidf     the assistant's search (TF-IDF cosine; essential 1.0, optional 0.5)
  overlap   how many query skills the occupation lists (essential or optional)
  coverage  overlap / number of skills the occupation lists

Ties count half (rank = 1 + higher + equal/2), so near-duplicate occupations with the
same skills don't make a scorer look better or worse than it is. Reported: rank 1,
top 3, top 10 and mean reciprocal rank (MRR), each with a 95% bootstrap interval over
occupations, and the paired MRR difference tfidf minus each baseline.

Data only: no LLM, no API. Run:
    python -m esco_core.smoke_occupations --data data --seed 0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import sparse

from esco_core.occupations import OccupationIndex

MIN_ESSENTIAL = 4          # a half of fewer than 2 skills says little
SCORERS = ("tfidf", "overlap", "coverage")


def rank_of(scores: np.ndarray, target: int) -> float:
    """1-based rank of `target`; ties with other occupations count half."""
    s = scores[target]
    higher = int(np.count_nonzero(scores > s))
    equal = int(np.count_nonzero(scores == s)) - 1
    return 1.0 + higher + equal / 2.0


def metrics(ranks: np.ndarray) -> dict:
    return {"rank1": float(np.mean(ranks <= 1.0)), "top3": float(np.mean(ranks <= 3.0)),
            "top10": float(np.mean(ranks <= 10.0)), "mrr": float(np.mean(1.0 / ranks))}


def bootstrap(values: np.ndarray, stat, n_boot: int, rng: np.random.Generator):
    """95% percentile interval of stat(values) over occupations resampled with replacement."""
    n = len(values)
    draws = [stat(values[rng.integers(0, n, n)]) for _ in range(n_boot)]
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def run(index: OccupationIndex, seed: int = 0) -> list[dict]:
    """One row per occupation: how many essentials, how many used, and the three ranks."""
    rng = np.random.default_rng(seed)
    rel = index._relations
    n_occ = len(index.occ_uris)
    occ_pos = {u: i for i, u in enumerate(index.occ_uris)}

    # binary occupation x skill matrix (essential or optional) for the baselines
    rows = rel["occupationUri"].map(occ_pos).to_numpy()
    cols = rel["skillUri"].map(index.skill_idx).to_numpy()
    binary = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)),
                               shape=(n_occ, len(index.skill_uris)))
    binary.data[:] = 1.0                                  # duplicates summed -> back to 1
    n_listed = np.asarray(binary.sum(axis=1)).ravel()
    n_listed[n_listed == 0] = 1.0

    essential = (rel[rel["relationType"] == "essential"]
                 .groupby("occupationUri")["skillUri"].apply(lambda s: sorted(set(s))))
    out = []
    for occ in index.occ_uris:                            # sorted: same order every run
        skills = essential.get(occ, [])
        if len(skills) < MIN_ESSENTIAL:
            continue
        half = rng.choice(len(skills), size=max(1, len(skills) // 2), replace=False)
        cols_q = np.asarray([index.skill_idx[skills[i]] for i in sorted(half)])

        q = np.zeros(len(index.skill_uris))
        q[cols_q] = index.idf[cols_q]
        q /= np.linalg.norm(q)
        tfidf = index.matrix @ q
        qb = np.zeros(len(index.skill_uris)); qb[cols_q] = 1.0
        overlap = binary @ qb
        coverage = overlap / n_listed

        t = occ_pos[occ]
        out.append({"occupation": occ, "label": index.occ_label.get(occ, occ),
                    "n_essential": len(skills), "n_query": len(cols_q),
                    "query": ";".join(index.skill_uris[cols_q]),
                    "tfidf": rank_of(tfidf, t), "overlap": rank_of(overlap, t),
                    "coverage": rank_of(coverage, t)})
    return out


def report(rows: list[dict], n_boot: int, seed: int) -> dict:
    rng = np.random.default_rng(seed + 1)
    res = {"n_occupations": len(rows), "min_essential": MIN_ESSENTIAL, "seed": seed,
           "bootstrap": n_boot, "scorers": {}, "mrr_difference": {}}
    ranks = {s: np.asarray([r[s] for r in rows]) for s in SCORERS}
    for s in SCORERS:
        point = metrics(ranks[s])
        res["scorers"][s] = {m: {"value": v, "ci95": bootstrap(ranks[s], lambda x, m=m: metrics(x)[m],
                                                               n_boot, rng)}
                             for m, v in point.items()}
    for base in ("overlap", "coverage"):
        diff = 1.0 / ranks["tfidf"] - 1.0 / ranks[base]       # paired, per occupation
        res["mrr_difference"][f"tfidf-{base}"] = {
            "value": float(diff.mean()), "ci95": bootstrap(diff, np.mean, n_boot, rng)}
    return res


def _print(res: dict) -> None:
    print(f"occupations: {res['n_occupations']} (>= {res['min_essential']} essential skills), "
          f"query = random half of the essential skills, seed {res['seed']}, "
          f"{res['bootstrap']} bootstrap resamples\n")
    print(f"{'scorer':<10} {'rank 1':>18} {'top 3':>18} {'top 10':>18} {'MRR':>18}")
    for s, m in res["scorers"].items():
        cells = [f"{m[k]['value']:.3f} [{m[k]['ci95'][0]:.3f}, {m[k]['ci95'][1]:.3f}]"
                 for k in ("rank1", "top3", "top10", "mrr")]
        print(f"{s:<10} " + " ".join(f"{c:>18}" for c in cells))
    print()
    for k, d in res["mrr_difference"].items():
        print(f"MRR {k}: {d['value']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]")


def _main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default="data")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", default=None, help="write ranks (.csv) and summary (.json) here")
    args = ap.parse_args()

    index = OccupationIndex(Path(args.data))
    rows = run(index, args.seed)
    res = report(rows, args.boot, args.seed)
    _print(res)
    if args.out:
        out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
        import csv
        with open(out.with_suffix(".csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        out.with_suffix(".json").write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(f"\nwritten: {out.with_suffix('.csv')}, {out.with_suffix('.json')}")


if __name__ == "__main__":
    _main()
