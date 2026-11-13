# esco-course-agent

An agent that tells a course designer which ESCO occupations a course prepares people for, and which essential skills it's missing.

**Status:** v1 runs end to end as a command-line chat. Evaluation, web demo and the full README come next.

## How it works

- `src/esco_core/`: skill linking (hybrid BM25 + EmbeddingGemma), ported from esco-skill-linker @ `3645d54f`; occupation lookup and TF-IDF occupation search.
- `src/esco_mcp/`: MCP server exposing `link_text`, `search_occupations`, `get_occupation_skills`.
- `src/esco_agent/`: LangGraph agent. The first turn is fixed (link the course, rank occupations); follow-ups run in a tool-calling loop with DeepSeek. "What's missing" is judged from the course text (`check_coverage`), not by comparing ESCO IDs.

## Run

Requires Python 3.10+, a DeepSeek API key, and a Hugging Face account that has accepted the [EmbeddingGemma](https://huggingface.co/google/embeddinggemma-300m) licence.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # plus torch for your machine
pip install -e .
echo "DEEPSEEK_API_KEY=sk-..." > .env

python -m esco_agent.cli --course course.txt --trace
```

**Data:** `data/` is not in the repo yet. It holds the ESCO v1.2.1 parquet files and the precomputed skill embeddings built by esco-skill-linker (`skills`, `texts`, `labels`, `occupations`, `occupation_skill_relations` `.parquet` + `embeddinggemma_en_labels_desc_e4d6efad9018.npy`). Hosting it is part of the deployment step.

`src/esco_core/stopwords_en.txt` is NLTK's English stopword list, included so no `nltk.download()` is needed; it is checked against a hash at load time.

## Tests

```bash
pytest -v
```

Tests that need `data/` are skipped without it. `tests/test_parity.py` checks the ported linker reproduces esco-skill-linker's rankings exactly.