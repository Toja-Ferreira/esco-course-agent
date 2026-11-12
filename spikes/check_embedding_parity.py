"""Compare the Space's EmbeddingGemma vector with your local one for the same text.

PASS if cosine >= 0.999 (my threshold, not a standard: small float differences
between machines are normal; anything lower means different model, prompt or settings).

Run from the repo root, in an env with the same versions as esco-skill-linker:
    pip install gradio_client
    python spikes/check_embedding_parity.py <your-hf-username>/embeddinggemma-check
HF_TOKEN in .env is needed for a private Space.
"""
import os
import sys

import numpy as np
from dotenv import load_dotenv
from gradio_client import Client
from sentence_transformers import SentenceTransformer

load_dotenv()
MODEL_ID = "google/embeddinggemma-300m"


# Keep identical to embed() in embeddinggemma_space/app.py
# (copied from esco-skill-linker src/retrieval/dense.py).
QUERY_PREFIX = "task: search result | query: "


def embed(model, text: str) -> np.ndarray:
    vec = model.encode(
        [QUERY_PREFIX + text],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]
    return np.asarray(vec, dtype=np.float32)


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


if __name__ == "__main__":
    space_id = sys.argv[1]
    texts = [
        "design relational database schemas and write SQL queries",
        "Students learn object-oriented programming in Java, unit testing "
        "and version control with git, working in small teams on a project.",
    ]

    client = Client(space_id, token=os.environ.get("HF_TOKEN"))
    model = SentenceTransformer(MODEL_ID)

    ok = True
    for t in texts:
        remote = np.array(client.predict(t, api_name="/embed"))
        local = embed(model, t)
        c = cosine(remote, local)
        ok &= c >= 0.999
        print(f"cos={c:.6f}  dims={len(remote)}/{len(local)}  {t[:50]!r}")

    print("\nbenchmark:", client.predict(texts[1], api_name="/benchmark"))
    print("\nPARITY PASS" if ok else "\nPARITY FAIL")
    sys.exit(0 if ok else 1)