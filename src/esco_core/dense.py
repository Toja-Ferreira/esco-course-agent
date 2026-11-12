"""EmbeddingGemma query encoding + precomputed skill vectors.

Copied from esco-skill-linker @ 3645d54f, src/retrieval/dense.py (ModelSpec
"embeddinggemma", DenseIndex._encode / scores_for). Changes from the linker:
  - the model revision is pinned, so a model update cannot silently pair new
    query vectors with the old cached skill vectors;
  - the skill vectors are loaded from an explicit file and never rebuilt here
    (rebuilding on CPU would differ by ~1e-6 and take a long time).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

MODEL_ID = "google/embeddinggemma-300m"
MODEL_REVISION = "57c266a740f537b4dc058e1b0cda161fd15afa75"  # HF refs/main on the linker machine
QUERY_PREFIX = "task: search result | query: "
DOC_PREFIX = "title: none | text: "  # used only when the skill vectors were built
EMBEDDINGS_FILE = "embeddinggemma_en_labels_desc_e4d6efad9018.npy"


class DenseIndex:
    def __init__(self, embeddings_path: Path, n_docs: int,
                 device: str | None = None, batch_size: int = 64):
        self.embeddings = np.load(embeddings_path)
        if self.embeddings.shape[0] != n_docs:
            raise ValueError(
                f"{embeddings_path} has {self.embeddings.shape[0]} rows, "
                f"expected {n_docs} skills"
            )
        self.batch_size = batch_size
        self._device = device
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                MODEL_ID, revision=MODEL_REVISION, device=self._device)
        return self._model

    def _encode(self, texts: Sequence[str], prefix: str) -> np.ndarray:
        prefixed = [prefix + t for t in texts] if prefix else list(texts)
        vecs = self.model.encode(
            prefixed,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,   # cosine becomes a dot product
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype=np.float32)

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, QUERY_PREFIX)

    def scores_for(self, query_vec: np.ndarray) -> np.ndarray:
        return query_vec @ self.embeddings.T
