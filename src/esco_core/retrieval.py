"""BM25, score fusion and helpers, copied from esco-skill-linker @ 3645d54f.

Function bodies are unchanged from the linker; only unused variants
(other document variants, RRF, z-score, search helpers) were dropped.
Sources: src/retrieval/bm25.py, src/retrieval/fusion.py, src/linker.py.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Mapping, Sequence
from importlib import resources

import numpy as np

# --- bm25.py -----------------------------------------------------------------

TOKEN_RE = re.compile(r"\w+", re.UNICODE)

# NLTK 'english' stopword list, vendored so no nltk.download() is needed.
# sha256 of the NLTK corpus file stopwords/english, as recorded on the
# linker machine; the file here must be a byte-identical copy.
STOPWORDS_EN_SHA256 = (
    "f6d005956f407dbc6ea32e5ff0c7e8e6f71488d3239b9023efdc7fc139d6375b"
)


def get_stemmer(lang: str):
    import snowballstemmer

    if lang != "en":
        raise ValueError("this port supports English only")
    return snowballstemmer.stemmer("english")


def get_stopwords(lang: str) -> frozenset[str]:
    if lang != "en":
        raise ValueError("this port supports English only")
    raw = resources.files("esco_core").joinpath("stopwords_en.txt").read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != STOPWORDS_EN_SHA256:
        raise RuntimeError(
            f"stopwords_en.txt sha256 {digest[:12]}... does not match NLTK's "
            f"english list ({STOPWORDS_EN_SHA256[:12]}...). Copy it again from "
            "~/nltk_data/corpora/stopwords/english."
        )
    # NLTK's WordListCorpusReader: one word per line, blank lines dropped.
    return frozenset(w for w in raw.decode("utf-8").splitlines() if w.strip())


def tokenize(text: str, stemmer=None, stopwords: frozenset[str] | None = None) -> list[str]:
    tokens = TOKEN_RE.findall(text.lower())
    if stopwords is not None:
        tokens = [t for t in tokens if t not in stopwords]
    if stemmer is not None:
        tokens = stemmer.stemWords(tokens)
    return tokens


def build_documents(
    uris: Sequence[str],
    preferred: dict[str, str],
    alt: dict[str, list[str]],
    description: dict[str, str],
) -> list[str]:
    """The linker's "labels_desc" variant: preferred + alt labels + description."""
    docs = []
    for u in uris:
        parts: list[str] = [preferred.get(u, "")]
        parts.extend(alt.get(u, []))
        parts.append(description.get(u, ""))
        docs.append(" ".join(p for p in parts if p))
    return docs


# --- fusion.py ---------------------------------------------------------------

Run = Mapping[str, Sequence[tuple[str, float]]]


def normalise(scores: np.ndarray) -> np.ndarray:
    """Min-max, per query (the linker's norm="minmax")."""
    if len(scores) == 0:
        return scores
    lo, hi = scores.min(), scores.max()
    return (scores - lo) / (hi - lo) if hi > lo else np.zeros_like(scores)


def weighted_score_fusion(runs: Sequence[Run], weights: Sequence[float]
                          ) -> dict[str, list[tuple[str, float]]]:
    if len(weights) != len(runs):
        raise ValueError("weights must match the number of runs")

    queries: set[str] = set()
    for run in runs:
        queries |= set(run)

    fused: dict[str, list[tuple[str, float]]] = {}
    for q in queries:
        acc: dict[str, float] = {}
        for run, weight in zip(runs, weights):
            items = run.get(q, [])
            if not items:
                continue
            docs = [d for d, _ in items]
            vals = normalise(np.array([s for _, s in items], dtype=float))
            for doc, val in zip(docs, vals):
                acc[doc] = acc.get(doc, 0.0) + weight * float(val)
        fused[q] = sorted(acc.items(), key=lambda kv: -kv[1])
    return fused


# --- linker.py ---------------------------------------------------------------

def top_n(scores: np.ndarray, uris: np.ndarray, n: int) -> list[tuple[str, float]]:
    n = min(n, len(scores))
    idx = np.argpartition(-scores, n - 1)[:n]
    idx = idx[np.argsort(-scores[idx], kind="stable")]
    return [(uris[i], float(scores[i])) for i in idx]


def clean(text: str) -> str:
    text = text.encode("utf-8", "surrogateescape").decode("utf-8", "replace")
    return unicodedata.normalize("NFC", text.replace("�", "")).strip()
