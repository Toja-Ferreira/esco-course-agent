"""ESCO occupations: look up an occupation's skills, and rank occupations by skills.

New code (not from the linker). Ranking is TF-IDF cosine over skills:

  occupation vector  o[s] = w(relation) * idf(s)     essential 1.0, optional 0.5
  query vector       q[s] = linker_score(s) * idf(s)
  idf(s)             = ln((1 + N) / (1 + df(s))) + 1  N occupations, df(s) =
                                                      occupations listing s
  score(o)           = cos(o, q) = sum_s o_hat[s] * q_hat[s]

IDF down-weights skills that many occupations list ("communication"), and the
cosine's length normalisation stops occupations with very long skill lists
from winning on raw overlap. Each term o_hat[s] * q_hat[s] is that skill's
contribution, so a score can be explained skill by skill.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

LANG = "en"
RELATION_WEIGHT = {"essential": 1.0, "optional": 0.5}


@dataclass(frozen=True)
class SkillRef:
    uri: str
    label: str
    skill_type: str
    reuse_level: str = ""


@dataclass(frozen=True)
class OccupationSkills:
    uri: str
    label: str
    isco_group: str
    description: str
    essential: list[SkillRef]
    optional: list[SkillRef]


@dataclass(frozen=True)
class Contribution:
    skill_uri: str
    skill_label: str
    relation: str          # "essential" or "optional" for this occupation
    contribution: float    # share of the occupation's score from this skill


@dataclass(frozen=True)
class OccupationHit:
    uri: str
    label: str
    isco_group: str
    score: float                       # cosine, in [0, 1]
    matched: list[Contribution]        # sorted, largest contribution first


class OccupationIndex:
    def __init__(self, data: Path):
        data = Path(data)
        occ = pd.read_parquet(data / "occupations.parquet")
        rel = pd.read_parquet(data / "occupation_skill_relations.parquet")
        texts = pd.read_parquet(data / "texts.parquet")

        t = texts[texts["language"] == LANG]
        o = t[t["conceptKind"] == "occupation"]
        s = t[t["conceptKind"] == "skill"]
        self.occ_label = dict(zip(o["conceptUri"], o["preferredLabel"]))
        self.occ_description = dict(zip(o["conceptUri"], o["description"]))
        self.skill_label = dict(zip(s["conceptUri"], s["preferredLabel"]))
        self.isco_group = dict(zip(occ["conceptUri"], occ["iscoGroup"]))
        skills_file = data / "skills.parquet"
        sk = pd.read_parquet(skills_file) if skills_file.exists() else pd.DataFrame()
        self.skill_reuse = (dict(zip(sk["conceptUri"], sk["reuseLevel"].fillna("")))
                            if "reuseLevel" in sk else {})

        rel = rel[rel["relationType"].isin(RELATION_WEIGHT)
                  & rel["occupationUri"].isin(self.isco_group)]
        self._relations = rel

        # Matrix rows: occupations; columns: skills that any occupation lists.
        self.occ_uris = np.asarray(sorted(occ["conceptUri"]))
        self.skill_uris = np.asarray(sorted(rel["skillUri"].unique()))
        occ_idx = {u: i for i, u in enumerate(self.occ_uris)}
        self.skill_idx = {u: j for j, u in enumerate(self.skill_uris)}

        # If a skill is listed both ways for one occupation, keep the stronger.
        w = (rel.assign(w=rel["relationType"].map(RELATION_WEIGHT))
                .groupby(["occupationUri", "skillUri"], as_index=False)["w"].max())
        rows = w["occupationUri"].map(occ_idx).to_numpy()
        cols = w["skillUri"].map(self.skill_idx).to_numpy()
        n_occ, n_skill = len(self.occ_uris), len(self.skill_uris)

        df = np.bincount(cols, minlength=n_skill)
        self.idf = np.log((1 + n_occ) / (1 + df)) + 1.0

        m = sparse.csr_matrix((w["w"].to_numpy() * self.idf[cols], (rows, cols)),
                              shape=(n_occ, n_skill))
        norms = np.sqrt(np.asarray(m.multiply(m).sum(axis=1)).ravel())
        norms[norms == 0] = 1.0
        self.matrix = sparse.csr_matrix(sparse.diags(1.0 / norms) @ m)

        # (occupation, skill) -> relation; essential wins if both are listed.
        self._relation_of: dict[tuple[str, str], str] = {}
        for kind in ("optional", "essential"):
            sub = rel[rel["relationType"] == kind]
            for o_, s_ in zip(sub["occupationUri"], sub["skillUri"]):
                self._relation_of[(o_, s_)] = kind

    # -- lookup ------------------------------------------------------------

    def get_occupation_skills(self, occupation_uri: str) -> OccupationSkills:
        if occupation_uri not in self.isco_group:
            raise KeyError(f"unknown occupation: {occupation_uri}")
        r = self._relations[self._relations["occupationUri"] == occupation_uri]

        def refs(kind: str) -> list[SkillRef]:
            sub = r[r["relationType"] == kind].sort_values("skillUri")
            return [SkillRef(u, self.skill_label.get(u, u), t, self.skill_reuse.get(u, ""))
                    for u, t in zip(sub["skillUri"], sub["skillType"])]

        return OccupationSkills(
            uri=occupation_uri,
            label=self.occ_label.get(occupation_uri, occupation_uri),
            isco_group=self.isco_group[occupation_uri],
            description=self.occ_description.get(occupation_uri) or "",
            essential=refs("essential"),
            optional=refs("optional"),
        )

    # -- search ------------------------------------------------------------

    def search_occupations(self, skill_scores: dict[str, float], k: int = 10,
                           isco_prefix: str | None = None,
                           max_matched: int | None = 5) -> list[OccupationHit]:
        """Rank occupations by TF-IDF cosine with the given skills.

        skill_scores: skill URI -> weight (the linker's fused score, or 1.0).
        isco_prefix: keep only occupations whose ISCO-08 code starts with it,
                     e.g. "25" (ICT professionals) or "2" (professionals).
        max_matched: how many contributing skills to return per occupation
                     (None = all; then the contributions sum to the score).
        Skills that no occupation lists are ignored.
        """
        q = np.zeros(len(self.skill_uris))
        for uri, score in skill_scores.items():
            j = self.skill_idx.get(uri)
            if j is not None and score > 0:
                q[j] = score * self.idf[j]
        qn = np.linalg.norm(q)
        if qn == 0:
            return []
        q /= qn

        scores = self.matrix @ q
        keep = scores > 0
        if isco_prefix:
            keep &= np.char.startswith(
                np.asarray([self.isco_group[u] for u in self.occ_uris], dtype=str),
                isco_prefix)
        idx = np.flatnonzero(keep)
        idx = idx[np.argsort(-scores[idx], kind="stable")][:k]

        hits = []
        for i in idx:
            o_uri = self.occ_uris[i]
            row = self.matrix.getrow(i)
            contrib = [
                Contribution(self.skill_uris[j],
                             self.skill_label.get(self.skill_uris[j], self.skill_uris[j]),
                             self._relation_of[(o_uri, self.skill_uris[j])],
                             float(v * q[j]))
                for j, v in zip(row.indices, row.data) if q[j] > 0
            ]
            contrib.sort(key=lambda c: -c.contribution)
            hits.append(OccupationHit(o_uri, self.occ_label.get(o_uri, o_uri),
                                      self.isco_group[o_uri], float(scores[i]),
                                      contrib if max_matched is None
                                      else contrib[:max_matched]))
        return hits
