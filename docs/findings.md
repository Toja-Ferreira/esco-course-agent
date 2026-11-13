# Findings log

Dated record of what was tried, what happened and what was decided, kept for the paper. One entry per finding. Each entry says how many courses it rests on; most so far rest on **one** course, so they are leads, not results.

Conventions: save raw outputs under `results/` (e.g. `python -m esco_agent.suggest --course course.txt > results/2026-11-13_c-course_llm.txt`) and name the file in the entry. Record the model actually served, not the alias.

---

## Baseline: the current POF matcher (as described on 13 Nov 2026)

`esco-skills-matcher-api`, `POST /analyze`. Matches page text against ESCO directly; no LLM extracts skill phrases.

1. KeyBERT keyphrases, fuzzy-matched (rapidfuzz `token_sort_ratio`, cutoff 90) against keywords pre-extracted from ~13.9k ESCO skills. Score = 0.8 × cosine + 0.2 × keyword strength.
2. Whole text embedded (`all-mpnet-base-v2`), cosine against every ESCO skill embedding, top 10.
3. Skills from the same KMeans clusters as the matches are added.
4. Lists merged, de-duplicated, scores under 50 dropped → `nlpResults`.
5. DeepSeek `deepseek-chat` re-ranks the shortlist and removes irrelevant skills → `hybridResults`. Returned names are checked for being valid ESCO skills, but **not** for being on the shortlist, so the LLM can add skills from memory.

No explanation or evidence is shown to the teacher; no back-and-forth.

## The test course used below

University of Aveiro "Programming" course unit (C language), English page: objectives, learning outcomes (same text as objectives), assessment, teaching methods, contents, bibliography. Saved as `course.txt`.

---

## 2026-11-12 · Whole-text linking returns the main topic's neighbours (n = 1 database course, 1 C course)

- Linker = hybrid BM25 + EmbeddingGemma, ported from esco-skill-linker @ `3645d54f` (parity-tested). Query = whole course text, top 20.
- Database course: sensible skills and occupations, but "data pipelines" linked to the oil-and-gas skill *analyse pipeline database information*, putting *pipeline engineer* (ISCO 2142) at rank 10. Word-sense error.
- C course: top skills were *Objective-C, C++, C#, SAS language, COBOL, Perl*. These drove the occupation ranking (*numerical tool and process control programmer* #1, *computer-aided design operator* #3).
- The linker was benchmarked on occupation descriptions, never on course text.

## 2026-11-12 · "Missing" cannot be decided by comparing ESCO IDs (n = 1 course–occupation pair)

- *Data warehouse designer* has 28 essential skills; the course's linked set is the linker's top 20. By ID nearly all 28 are "missing", including ones the course arguably covers (*information structure*, *create software design*).
- "Missing by ID" measures "not in the linker's top 20", not "absent from the course".
- **Decision:** an LLM judge reads the course as numbered sentences and returns covered/missing per skill with supporting sentence numbers; a validator rejects skipped, duplicated or invented skills and non-existent sentence numbers; one retry, then `unverified`.

## 2026-11-13 · First full agent run (n = 1, C course; DeepSeek alias served by `deepseek-flash`)

- Tool choice correct on 4/4 follow-ups ("why not the second one?", "only ict jobs please", "yes for number 10", "okay thanks").
- The judge marked *C++, C#, Objective-C* as missing although the linker had "found" them: text-based judging corrects the linker.
- The judge was too lenient elsewhere: for *computer science lecturer* it counted teaching skills (*assess students*, *apply teaching strategies*) as covered by a course that teaches C. The agent's own answer admitted they were "inferred".

## 2026-11-13 · Product change

Boss asked for a POF assistant that helps teachers pick ESCO skills with explanation and conversation; teacher test planned for the first half of December 2026 (5–10+ teachers, ≤3 at once); results to go in a paper. New spec in the Claude project (`skill-assistant-spec.md`). v1 (course → occupations → gaps) is tagged `v1` and is reused.

## 2026-11-13 · Suggesting skills: three ways of finding candidates, same judge (n = 1, C course)

Pipeline: candidates from the linker → LLM judge keeps those the text supports, with evidence sentences. Judge reply was valid on the first attempt in every run.

| Candidates from | Candidates judged | Suggested | What was suggested |
|---|---|---|---|
| Whole text, top 30 | 30 | 2 | *apply basic programming skills*, *computer programming* |
| Whole text top 15 + each sentence top 4 | 66 | 4 | the two above + *create solutions to problems*, *organise information, objects and resources* |
| Whole text top 15 + LLM-proposed queries (20), top 4 each | 76 | 11 | the first two + *think analytically*, *work independently*, *systems development life-cycle*, *create solutions to problems*, *address problems critically*, *types of strings*, *information structure*, *perform software unit testing*, *debug software* |

- The judge removed every near-miss language (*Objective-C, C++, C#, COBOL, Perl, …*) in all runs.
- Per-sentence queries added word-sense junk as candidates (*demolish structures*, *facilitate compositional structures in dance*, *mark lumber*, *file documents*) and nothing on Unix/Linux, the command line, memory or debugging. Candidate recall, not the judge, is the bottleneck.
- Per-sentence run (repeated with `found_by` shown): 6 suggested; the two extra were *assessment processes* and *assess students*, taken from the page's assessment section. The course is assessed, it does not teach assessment: a judge error of the "teaching vs learning" kind.
- LLM queries (served by `deepseek-flash`) read like the course: *use pointers*, *manage dynamic memory*, *compile and execute programmes via command line*, *organise programmes in multiple files using make*, *test code*, …
- LLM-query run, precision (our reading, not measured): most of the 11 are defensible, several are transversal skills (*think analytically*, *work independently*). Suspect: *types of strings* (found by the query "use strings" alongside *tune stringed musical instruments* and *thread wire*, so probably the musical sense; **check its ESCO description**) and *information structure* (from "use data structures"; may be a different sense). *debug software* rests on one sentence that only says "testing". The judge sees labels only, which is the likely cause.
- LLM-query run, recall: still nothing for Unix/Linux, the command line, memory management, file handling or `make`. Their queries returned wrong-sense candidates (*perform manual work autonomously*, *use geographic memory*, *use translation memory software*, *operate file for deburring*, *thread wire*). Two possible causes, not yet separated: long query phrases pulling BM25 off target, or ESCO having no skill for these topics (a taxonomy gap worth reporting if confirmed).
- Against POF on this course: POF's two errors (*C++*, *project management*) are absent here; POF's *ICT system programming* and *levels of software testing* were not among our candidates in this run.
- The judge may be slightly strict: it rejected *implement ICT coding conventions* and *provide software testing documentation* although the course mentions good practices, documentation and code testing. Not tuned; one course.

## 2026-11-13 · POF's current output on the same course (n = 1)

- `nlpResults` (10): *computer programming, C++, Solidity, MATLAB, R, teach computer science, Objective-C, C#, SAS language, develop data processing applications*. Same failure as our whole-text linking.
- `hybridResults` (8): *computer programming, apply basic programming skills, C++, analyse software specifications, conduct ICT code review, levels of software testing, ICT system programming, project management*.
- **Only 2 of the 8 final skills are on the shortlist.** The other 6 were added by the LLM from memory through the unchecked path in step 5. The more useful suggestions (*ICT system programming*, *levels of software testing*) are among them; so are errors (*project management* is not in the text; *C++* is wrong for a C course).
- Reading (ours, not measured): POF finds more but cannot say why and lets errors through; our judged pipeline is safe but finds too little.
- **Decision:** make the LLM's proposals explicit and checked. The LLM proposes short skill phrases as search queries only; the linker turns them into real ESCO skills; the judge keeps those the text supports, with evidence. Implemented as `--candidates llm` (default) in `esco_agent.suggest`; `sentences` and `whole` kept for comparison.

---

## Open questions and things not yet measured

- Everything above rests on one or two courses. Needed before tuning anything: a handful of courses, some outside computing, with expected skills written down **before** looking at any system's output.
- Whether the judge needs ESCO skill descriptions, not just labels. Leading candidate for the next change: it would address the word-sense acceptances (*types of strings*) and the borderline rejections.
- Candidate settings (15 whole-text, 4 per query, 30 per judge batch) are starting guesses.
- The model alias `deepseek-chat` moves; pin the model before any reported run.
- Teacher test design: what is logged; consent or ethics approval; whether each teacher uses both POF's current flow and the assistant.
