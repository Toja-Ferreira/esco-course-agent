"""SkillLinker: the linker's first stage (hybrid BM25 + dense), English only.

Copied from esco-skill-linker @ 3645d54f, src/linker.py (SkillLinker.__init__,
first_stage, link). Changes from the linker:
  - English only; the fusion weight is hardcoded (see FUSION_WEIGHT);
  - no reranker;
  - link() returns SkillHit objects with label, type and score, because the
    agent's tools need them (the linker returned bare URIs);
  - data files are read from one directory (ESCO_DATA_DIR, default ./data).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from esco_core.dense import EMBEDDINGS_FILE, DenseIndex
from esco_core.retrieval import (build_documents, clean, get_stemmer,
                                 get_stopwords, tokenize, top_n,
                                 weighted_score_fusion)

LANG = "en"
RUN_DEPTH = 200  # fusion input depth, as in the linker's experiment 13

# BM25's weight in the hybrid, selected on dev in the linker's experiment 13
# (results/fusion_dev.csv: 0.4 -> nDCG@10 0.379683 vs 0.3 -> 0.379445).
# The linker read this from the CSV at runtime; hardcoded here so a rerun of
# that experiment can't change it. Portuguese would be 0.3.
FUSION_WEIGHT = 0.4


def data_dir() -> Path:
    return Path(os.environ.get("ESCO_DATA_DIR", "data")).resolve()


@dataclass(frozen=True)
class SkillHit:
    uri: str
    label: str
    skill_type: str   # ESCO: "skill/competence" or "knowledge"
    score: float      # fused score, in [0, 1]
    description: str = ""   # ESCO's own definition of the skill


class SkillLinker:
    def __init__(self, data: Path | None = None, device: str | None = None):
        data = Path(data) if data is not None else data_dir()

        texts = pd.read_parquet(data / "texts.parquet")
        labels = pd.read_parquet(data / "labels.parquet")
        skills = pd.read_parquet(data / "skills.parquet")

        # Sorted URI order is the row order of the cached skill embeddings.
        self.uris = np.asarray(sorted(skills["conceptUri"]))
        t = texts[(texts["language"] == LANG) & (texts["conceptKind"] == "skill")]
        preferred = dict(zip(t["conceptUri"], t["preferredLabel"]))
        description = dict(zip(t["conceptUri"], t["description"]))
        a = labels[(labels["language"] == LANG) & (labels["labelType"] == "alt")]
        alt: dict[str, list[str]] = {}
        for uri, label in zip(a["conceptUri"], a["label"]):
            alt.setdefault(uri, []).append(label)

        self.label = preferred
        self.description = description
        self.skill_type = dict(zip(skills["conceptUri"], skills["skillType"]))

        docs = build_documents(list(self.uris), preferred, alt, description)
        self.stemmer = get_stemmer(LANG)
        self.stopwords = get_stopwords(LANG)
        self.bm25 = BM25Okapi([tokenize(d, self.stemmer, self.stopwords)
                               for d in docs])
        self.dense = DenseIndex(data / EMBEDDINGS_FILE, len(self.uris),
                                device=device)
        self.weight = FUSION_WEIGHT

        self.dense.encode_queries(["warm up"])  # load the model now, not on query 1

    def first_stage(self, text: str) -> list[tuple[str, float]]:
        """Every fused skill (~330), best first. Expects clean()ed text."""
        tokens = tokenize(text, self.stemmer, self.stopwords)
        # A query made only of stopwords has no lexical signal; leaving the
        # BM25 run empty lets the dense run decide alone.
        lexical = ({"q": top_n(self.bm25.get_scores(tokens), self.uris,
                               RUN_DEPTH)} if tokens else {})
        qvec = self.dense.encode_queries([text])[0]
        dense = {"q": top_n(self.dense.scores_for(qvec), self.uris, RUN_DEPTH)}
        return weighted_score_fusion([lexical, dense],
                                     [self.weight, 1.0 - self.weight])["q"]

    def link(self, text: str, k: int = 10) -> list[SkillHit]:
        ranked = self.first_stage(clean(text))[:k]
        return [SkillHit(u, self.label.get(u, u), self.skill_type.get(u, ""), s,
                         self.description.get(u) or "")
                for u, s in ranked]
