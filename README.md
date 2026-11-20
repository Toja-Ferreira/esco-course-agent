# esco-course-agent

**A chat assistant that helps university teachers describe a course unit with [ESCO](https://esco.ec.europa.eu/) skills, and shows, sentence by sentence, why each skill fits.** The teacher pastes the course page; the assistant suggests ESCO skills, each with the course sentence that supports it; the teacher keeps or removes them in conversation, asks which occupations the course leads to and what it lacks for one, and gets proposed text for the course page, checked before it is shown.

- **Why it needs tools:** asked directly, the same LLM gets the exact ESCO name right for only **19%** of the skills it lists [95% CI 9–29%] and never gives an ESCO identifier. With tools, every suggestion is a real ESCO entry with a sentence from the text, and the judge's suggestions are right or partly right **98%** of the time (strictly right: 79%).
- **Stack:** LangGraph agent with function calling (DeepSeek), an MCP server over a hybrid BM25 + EmbeddingGemma skill linker, FastAPI with streamed replies, Docker.
- **Status:** deployed inside the University of Aveiro's course platform (POF), piloted and approved by the project supervisor; a study with teachers runs on 1 December 2026.

![demo](docs/demo.gif)

## How it works

```mermaid
flowchart LR
    T[Course page text] --> Q[LLM: short search phrases]
    Q --> L[Linker: BM25 + EmbeddingGemma<br/>6 candidates per phrase]
    T --> L
    L --> J[LLM judge: ESCO definition<br/>vs numbered course sentences]
    J --> S[Suggestions + supporting sentence]
    S --> C{Conversation:<br/>tool-calling loop}
    C -->|find_skills| L
    C -->|find_occupations| O[TF-IDF occupation search]
    C -->|check_occupation_gap| J
    C -->|suggest_text_change| D[Draft text, then re-judge<br/>the course with it added]
```

- **First answer is fixed, not agentic:** an LLM turns the course into short search phrases; the linker returns candidates for each; a judge reads each candidate's ESCO definition against the numbered course sentences and keeps only those it can cite a sentence for. A validator rejects answers that skip, repeat or invent skills or sentence numbers, with one retry.
- **Follow-ups are a tool-calling loop** (LangGraph) over: update the list, search for more skills, rank occupations, check an occupation's gap, draft a text change, decide on a text change. ESCO data is served by an MCP server (`link_text`, `search_occupations`, `get_occupation_skills`).
- **Text changes are verified:** a draft is added to its section and the judge runs again; it only counts if the judge now finds the skill and cites the new text. Failed drafts are explained ("what the course would need to cover"), not offered.
- **Grounding:** skills only from tool results, evidence only from the course text; every reply is checked for quotes not found in the text or tool results and for skill numbers that don't exist, and flagged to the teacher.

## Results

All numbers from `docs/findings.md`; raw outputs under `results/`.

| What | Result | How |
|---|---|---|
| **Occupation search** (half of an occupation's essential skills as the query) | rank 1: **0.942** [0.933, 0.949]; top 10: 0.987. Simpler scorers: overlap 0.780, coverage 0.744 | 3,038 ESCO occupations, 95% bootstrap CI; `python -m esco_core.smoke_occupations` |
| **No-tools baseline** (same LLM, asked directly) | exact ESCO label **0.190** [0.088, 0.291]; any ESCO label 0.206; URI given 0.000; sentence really in the text 0.948 | 6 real course pages × 3 runs, 248 items; `python -m esco_agent.baseline` |
| **Judge evidence precision** | strict **0.785** [0.678, 0.864]; lenient (incl. partly supported) **0.978** [0.954, 1.000]; wrong 2/93 | all 93 first-turn suggestions on 6 courses, labelled with a fixed rule by Claude (a different model family from the judge); human agreement on 19 rows: 0.84, κ = 0.31 |
| **Candidates per search phrase** 4 → 6 | e.g. "operating systems" now reaches the judge for a C course; adopted as the default | 7 courses × 3 runs; `python -m esco_agent.suggest --per-query-k` |

The labelling rule: for knowledge items, *right* if the cited sentence names or covers the topic; for activities, *right* only if students are shown doing it, *borderline* if only the topic is named; *wrong* if the words match in a different sense.

**Not evaluated:** the planned ~25 multi-turn conversations scoring tool choice, and an LLM judge of occupation correctness validated on human labels. Tool use is covered by unit tests with scripted LLMs and by real sessions read by hand, not by a measured rate.

## Limitations

- The linker was benchmarked on occupation descriptions, never on course text. Short or framed sentences ("become autonomous in the use of Linux") can rank the right skill low; 6 candidates per phrase was chosen on 7 courses.
- The judge sometimes accepts a skill on shared words despite its definition (2/93), and decides borderline candidates differently between runs.
- ESCO has gaps for course content: no skill for using Linux or the C language; programming languages and operating systems are folded into broad entries.
- English course text only (the linker retrieves better in English); the conversation is Portuguese or English.
- The smoke test uses exact ESCO skills of the target occupation, so it checks the ranking, not the occupations a real course gets.

## Run locally

Python 3.10+, a DeepSeek API key, and a Hugging Face read token for an account that has accepted the [EmbeddingGemma](https://huggingface.co/google/embeddinggemma-300m) licence.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # plus torch for your machine
pip install -e .
cp .env.example .env              # set DEEPSEEK_API_KEY and HF_TOKEN; remove ESCO_AUTH=pof

python -m esco_agent.web          # web assistant on http://localhost:8000 (no login: type any email)
python -m esco_agent.suggest --course course.json --repeat 3   # suggestions only, with stability
```

**Data:** `data/` is not in the repo. It holds ESCO v1.2.1 as parquet files and the precomputed skill embeddings, built by the ingest step of the skill-linker project: `skills`, `texts`, `labels`, `occupations`, `occupation_skill_relations` `.parquet` + `embeddinggemma_en_labels_desc_e4d6efad9018.npy`. ESCO is published by the European Commission and free to reuse.

**Course files** are JSON: `{"Learning outcomes": "...", "Contents": "..."}`.

**Deployment** behind another platform's login: `deploy/DEPLOY.md` (Docker, nginx).

## Repository

- `src/esco_core/`: skill linker (hybrid BM25 + EmbeddingGemma, ported from the linker project and checked for identical rankings), occupation search, smoke test.
- `src/esco_mcp/`: MCP server over the ESCO data.
- `src/esco_agent/`: the assistant (`assistant.py`, LangGraph), suggestion pipeline and judge (`suggest.py`, `coverage.py`), grounding check, baseline and labelling scripts, and the web app (`web/`).
- `tests/`: 78 tests; those needing `data/` are skipped without it. `pytest -q`.
- `docs/findings.md`: dated findings, decisions and every number above.
