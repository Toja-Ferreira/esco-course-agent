"""The ported linker must reproduce esco-skill-linker's output.

Reference: first_stage(clean(q)) from esco-skill-linker @ 3645d54f,
SkillLinker("en", device="cpu"), top 11 per query (tests/reference/).
On the linker machine, CPU and CUDA runs gave identical rankings with score
differences up to 5.4e-7, so: order must match exactly, scores within 1e-4.
Tightest margins: (a) ranks 10/11 differ by 0.0005, (c) ranks 8-10 by < 0.004.

Needs the real data (ESCO_DATA_DIR, default ./data) and the model; skipped
otherwise. Run:  pytest tests/test_parity.py -v
"""

import json
from pathlib import Path

import pytest

from esco_core.linker import EMBEDDINGS_FILE, SkillLinker, data_dir
from esco_core.retrieval import clean

REFERENCE = Path(__file__).parent / "reference" / "linker_3645d54f_en.json"
QUERIES = {
    "a": "design relational database schemas and write SQL queries",
    "b": "Students learn object-oriented programming in Java, unit testing "
         "and version control with git, working in small teams on a project.",
    "c": "manage a team and give live presentations to customers",
}
SCORE_TOL = 1e-4


@pytest.fixture(scope="module")
def linker():
    if not (data_dir() / EMBEDDINGS_FILE).exists():
        pytest.skip(f"no data in {data_dir()}")
    return SkillLinker(device="cpu")


@pytest.mark.parametrize("key", sorted(QUERIES))
def test_matches_linker(linker, key):
    expected = json.loads(REFERENCE.read_text())[key]
    got = linker.first_stage(clean(QUERIES[key]))[:len(expected)]

    got_uris = [u for u, _ in got]
    exp_uris = [u for u, _ in expected]
    assert got_uris == exp_uris, "ranking differs from the linker"

    for rank, ((u, s), (_, e)) in enumerate(zip(got, expected), start=1):
        assert abs(s - e) <= SCORE_TOL, f"rank {rank} {u}: {s:.6f} vs {e:.6f}"


def test_link_returns_labelled_hits(linker):
    hits = linker.link(QUERIES["a"], k=5)
    assert len(hits) == 5
    assert all(h.label and not h.label.startswith("http") for h in hits)
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)
