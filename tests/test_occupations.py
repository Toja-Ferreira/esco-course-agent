"""Occupation lookup and search on the real ESCO data.

Sanity checks, not the evaluation: the smoke test in the spec (half of the
essential skills) belongs to eval/. Skipped when data/ is missing.
Run:  pytest tests/test_occupations.py -v -s
"""

import random

import pandas as pd
import pytest

from esco_core.linker import data_dir
from esco_core.occupations import OccupationIndex

# First row of occupation_skill_relations.parquet, as printed on 12 Nov 2026.
KNOWN_OCC = "http://data.europa.eu/esco/occupation/00030d09-2b3a-4efd-87cc-c4ea39d27c34"
KNOWN_ESSENTIAL = "http://data.europa.eu/esco/skill/05bc7677-5a64-4e0c-ade3-0140348d4125"


@pytest.fixture(scope="module")
def index():
    if not (data_dir() / "occupations.parquet").exists():
        pytest.skip(f"no data in {data_dir()}")
    return OccupationIndex(data_dir())


@pytest.fixture(scope="module")
def relations():
    return pd.read_parquet(data_dir() / "occupation_skill_relations.parquet")


def test_lookup_known_occupation(index):
    occ = index.get_occupation_skills(KNOWN_OCC)
    assert KNOWN_ESSENTIAL in {s.uri for s in occ.essential}
    assert not occ.label.startswith("http")
    assert all(not s.label.startswith("http") for s in occ.essential + occ.optional)
    print(f"\n{occ.label} (ISCO {occ.isco_group}): "
          f"{len(occ.essential)} essential, {len(occ.optional)} optional")


def test_every_occupation_has_a_label(index):
    missing = [u for u in index.occ_uris if u not in index.occ_label]
    assert not missing, f"{len(missing)} occupations without an English label"


def test_contributions_sum_to_score(index, relations):
    skills = relations["skillUri"].drop_duplicates().sample(15, random_state=0)
    hits = index.search_occupations({s: 1.0 for s in skills}, k=10, max_matched=None)
    assert hits
    for h in hits:
        assert abs(sum(c.contribution for c in h.matched) - h.score) < 1e-9
        assert 0 < h.score <= 1 + 1e-9


def test_isco_filter(index, relations):
    skills = relations["skillUri"].drop_duplicates().sample(40, random_state=1)
    hits = index.search_occupations({s: 1.0 for s in skills}, k=20, isco_prefix="25")
    assert hits and all(h.isco_group.startswith("25") for h in hits)


def test_self_retrieval_full_essentials(index, relations):
    """An occupation queried with ALL its essential skills should come back near
    the top. ESCO has near-duplicate occupations, so rank 1 isn't guaranteed;
    the 90%-in-top-3 threshold is a guess, not a measured baseline."""
    ess = relations[relations["relationType"] == "essential"]
    occs = sorted(ess["occupationUri"].unique())
    random.Random(0).shuffle(occs)
    ranks = []
    for u in occs[:200]:
        q = {s: 1.0 for s in ess.loc[ess["occupationUri"] == u, "skillUri"]}
        uris = [h.uri for h in index.search_occupations(q, k=10, max_matched=0)]
        ranks.append(uris.index(u) + 1 if u in uris else None)
    at1 = sum(r == 1 for r in ranks) / len(ranks)
    top3 = sum(r is not None and r <= 3 for r in ranks) / len(ranks)
    print(f"\nall essentials -> rank 1: {at1:.0%}, top 3: {top3:.0%} (n={len(ranks)})")
    assert top3 >= 0.9


def test_end_to_end_database_course(index):
    """Linker -> occupation search on a database course text."""
    from esco_core.linker import SkillLinker

    linker = SkillLinker(device="cpu")
    hits = linker.link("Students design relational database schemas, apply "
                       "normalisation and write SQL queries with joins and "
                       "aggregation, and build small data pipelines in Python.", k=20)
    occs = index.search_occupations({h.uri: h.score for h in hits}, k=10)
    print("\nskills:", [h.label for h in hits[:8]])
    for o in occs:
        print(f"  {o.score:.3f}  {o.label}  (ISCO {o.isco_group})  <- "
              + ", ".join(c.skill_label for c in o.matched[:3]))
    assert any("database" in o.label.lower() for o in occs)
