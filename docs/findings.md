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
- **Checked against ESCO's own descriptions (13 Nov):**
  - *types of strings* = "Vibrating elements that generate sounds in string instruments…". A confirmed wrong-sense acceptance by the judge.
  - *information structure* = "The type of infrastructure which defines the format of data: semi-structured, unstructured and structured." Not data structures in the programming sense; a second wrong-sense acceptance.
  - So 2 of the 11 suggestions are wrong for a reason the judge could not see: it is given labels only.
  - ESCO labels containing "Linux", "memory" or "operating system": *Kali Linux*, *operating systems*, *mobile operating systems*, *install operating system*, *Android (mobile operating systems)*, *use geographic memory*, *use translation memory software*, *develop translation memory software*. So the recall gap has both causes: ESCO has no skill for memory management or for Linux in general (a taxonomy gap), and it does have *operating systems*, which our queries failed to retrieve (a retrieval miss).
- Against POF on this course: POF's two errors (*C++*, *project management*) are absent here; POF's *ICT system programming* and *levels of software testing* were not among our candidates in this run.
- The judge may be slightly strict: it rejected *implement ICT coding conventions* and *provide software testing documentation* although the course mentions good practices, documentation and code testing. Not tuned; one course.

## 2026-11-13 · POF's current output on the same course (n = 1)

- `nlpResults` (10): *computer programming, C++, Solidity, MATLAB, R, teach computer science, Objective-C, C#, SAS language, develop data processing applications*. Same failure as our whole-text linking.
- `hybridResults` (8): *computer programming, apply basic programming skills, C++, analyse software specifications, conduct ICT code review, levels of software testing, ICT system programming, project management*.
- **Only 2 of the 8 final skills are on the shortlist.** The other 6 were added by the LLM from memory through the unchecked path in step 5. The more useful suggestions (*ICT system programming*, *levels of software testing*) are among them; so are errors (*project management* is not in the text; *C++* is wrong for a C course).
- Reading (ours, not measured): POF finds more but cannot say why and lets errors through; our judged pipeline is safe but finds too little.
- **Decision:** make the LLM's proposals explicit and checked. The LLM proposes short skill phrases as search queries only; the linker turns them into real ESCO skills; the judge keeps those the text supports, with evidence. Implemented as `--candidates llm` (default) in `esco_agent.suggest`; `sentences` and `whole` kept for comparison.

## 2026-11-13 · Which page sections to feed, and judge stability (n = 1, C course, LLM-query mode)

POF stores page sections as separate fields. Same pipeline on three inputs (`results/2026-11-13_c-course_lo.txt`, `..._lo-contents.txt`, and the full-page run above):

| Input | Candidates | Suggested |
|---|---|---|
| Full page | 76 | 11 |
| Learning outcomes + contents | 82 | 11 |
| Learning outcomes only | 83 | 9 |

- **Stable core:** 7 skills were suggested in all three runs: *apply basic programming skills, computer programming, think analytically, work independently, systems development life-cycle, create solutions to problems, address problems critically*. *perform software unit testing* also appeared in all three.
- **The judge is not consistent on borderline skills.** Same skill, same supporting sentences, different verdicts across runs: *types of strings* accepted in 2 runs and rejected in 1; *information structure* accepted in 2, rejected in 1; *debug software* accepted in 1, rejected in 1 (not a candidate in the third). Temperature is 0, but the surrounding candidates and batch composition differ.
- Consequence: a difference of 9 vs 11 suggestions between single runs is within this noise. **Counts from single runs cannot rank the three inputs.** Any reported comparison needs repeated runs or a per-skill agreement measure.
- With LLM queries, the assessment and bibliography sections produced no suggestions even on the full page (the query prompt excludes how the course is assessed or taught), so section choice matters less here than in per-sentence mode, where it produced *assess students*.
- Contents added one query ("apply code documentation and testing practices") that brought *implement ICT coding conventions* and *provide software testing documentation*, both rejected earlier in per-sentence mode with the same evidence sentence available. More sign of a judge near its decision boundary.
- **Decision:** take named sections as input (learning outcomes + contents by default; assessment, teaching methods, requisites and bibliography excluded), so evidence can cite its section. Reason is principled (those sections don't say what is taught) rather than shown by these counts.

## 2026-11-16 · Judge given ESCO definitions; named sections as input; stability report (results pending)

- Change: the judge now sees each candidate as `label: ESCO definition` (first 250 characters), and the prompt tells it to judge the skill as defined and to reject "the same word in a different sense". `--no-descriptions` restores label-only judging.
- Change: input is named sections (`.json`, as POF stores them); evidence cites section and sentence. Repeated sentences are kept once, where they first appear.
- New measurement: `--repeat N` runs the whole pipeline N times and reports, per skill, in how many runs it was suggested and in how many it was a candidate. "Same candidate judged differently" isolates the judge wavering from the query step producing different candidates.
- Result (C course, learning outcomes + contents, 5 full runs each, 16 Nov; `results/2026-11-16_c-course_labels_x5.txt`, `..._definitions_x5.txt`; served model not recorded in these files, record it next time):

| Judge sees | Suggested per run | In every run | In some runs only | Same candidate judged differently | *types of strings* | *information structure* |
|---|---|---|---|---|---|---|
| Label only | 6, 6, 10, 7, 8 | 6 | 5 | 4 | 1/5 | 0/5 |
| Label + definition | 5, 5, 6, 6, 6 | 4 | 4 | 2 | 0/5 | 0/5 |

- With definitions the two wrong-sense skills never appeared (10 runs is too few to call *types of strings* 1/5 → 0/5 a measured effect, but it is the direction expected), the run-to-run spread narrowed (5–6 vs 6–10) and judge flips halved (4 → 2).
- **The judge also became stricter.** Two skills suggested 5/5 from labels dropped to 0/5 with definitions: *organise information, objects and resources* (a stretch; a fair loss) and *apply basic programming skills*, which a first C programming course plausibly does teach. Probable cause, not checked: the judge reads the definition's wording literally. *address problems critically* also went (1/5 → 0/5).
- Stable core with definitions (5/5): *computer programming, create solutions to problems, perform software unit testing, think analytically*; *systems development life-cycle* 4/5.
- Part of the instability is the query step, not the judge: *work independently* and *implement ICT coding conventions* were suggested every time they were candidates, but were candidates in only 1–3 of 5 runs.
- **Recall is now the main problem.** 5–6 suggestions for a course page that names pointers, arrays, files, dynamic memory, the command line, Unix/Linux and `make`. Known causes: ESCO has no skill for several of these (taxonomy gap), and *operating systems* exists but is never retrieved.
- **Decision:** keep definitions on (fewer wrong-sense errors and steadier output matter more in front of teachers than one lost generic skill), and let the conversation recover what the judge drops: the teacher can ask "anything on X?" and can see rejected candidates. Not tuning the prompt further on one course.

## 2026-11-16 · First conversation with the skill assistant (n = 1, C course, Portuguese chat)

`esco_agent.assistant`: fixed first turn (6 suggestions with evidence), then DeepSeek with two tools (`update_skill_list`, `find_skills`). Transcript not yet saved under `results/`; save the next ones with their `skill_list.json`.

- **An open question recovered what the first turn missed.** Asked "que outras skills pensas que faltam?", the agent chose six searches itself (*operating systems, use command line interface, data structures, manage memory, debug software, write technical documentation*) and found three more skills the judge accepted with evidence: *operating systems*, *computer science*, *execute software tests*. *operating systems* is the skill the first-turn queries never retrieved.
- **Lead for the first turn:** the searches that worked here were short noun phrases ("operating systems"); the first-turn queries are longer verb phrases ("use Unix/Linux environments autonomously"). Shorter queries may fix part of the recall gap. Not tested.
- **The agent reported taxonomy gaps unprompted and correctly**, as far as checked on 13 Nov: nothing in ESCO for the command line, dynamic memory management or code documentation. It also said *debug software* exists but the text speaks of testing, not debugging.
- Tool use was correct on every turn: one `update_skill_list` call for "keep 1, 3, 4 … 7 and 23", an explanation of 39 from the state without a tool call, then a second call after "sim". It volunteered that *execute software tests* is "a reasonable, not perfect" match because the definition mentions customer requirements.
- Final list: *computer programming, systems development life-cycle, create solutions to problems, operating systems, computer science, execute software tests*. *computer science* for "data structures" is broad (our reading).
- Problems seen:
  - Skill numbers jump (7, 23, 39) because every search result gets a number, including the ones not in the text; after six searches about 48 skills were numbered, and all of them are resent to the LLM each turn.
  - One quote was shown with a leading "…" (a cut sentence). The automatic grounding check must allow for cut quotes or the agent must quote whole sentences.
  - Whether every skill named in the "ESCO has nothing" paragraph came from a tool result was not verified (the trace was truncated).

## 2026-11-17 · Labels vs definitions, rerun with verified files (n = 1, C course, 5 runs each)

On 16 Nov the source files had been installed by copy-pasting from a preview, which altered some characters (checksums differed from the originals, and a unit-test judge reply was rejected that is accepted with the original files). **Treat the 16 Nov numbers above, and the 16 Nov conversation, as provisional.** Rerun after installing byte-identical files (`results/2026-11-17_c-course_labels_x5.txt`, `..._definitions_x5.txt`):

| Judge sees | Suggested per run | In every run | In some runs only | Same candidate judged differently | *types of strings* | *systems development life-cycle* |
|---|---|---|---|---|---|---|
| Label only | 9, 7, 9, 9, 6 | 6 | 4 | 3 | 3/5 | 2/5 |
| Label + definition | 5, 5, 5, 6, 6 | 5 | 1 | 0 | 0/5 | 5/5 |

- Same direction as the first run, and clearer: with definitions the wrong-sense skill never appears (3/5 from labels), no candidate received different verdicts across runs (3 from labels), and the only unstable skill (*implement ICT coding conventions*, 2/5) was accepted every time it was a candidate, so that instability comes from the query step.
- The cost is unchanged: *apply basic programming skills* and *organise information, objects and resources* are 5/5 from labels and 0/5 with definitions. The first is plausibly a real loss.
- Recall is still low: five or six suggestions.
- Lesson recorded for the methods section: results are tied to exact prompt text; record a checksum or the commit hash with every reported run.

## 2026-11-16 · Built, not yet run with the real model: occupations and gaps, text suggestions, web page

- Occupations are ranked from the **accepted** list (TF-IDF search from v1), and a gap check sorts an occupation's essential skills into: on the list / shown by the text but not on the list (numbered, with evidence) / not in the text.
- Text suggestions: for one skill the text does not show, the model drafts one or two sentences for a named section; the draft is added to that section and the same judge is run on the revised course. It counts as verified only if the judge now finds the skill **and cites the new text**. One redraft on failure; a failed check is reported to the teacher as failed. The page is never changed; accepted changes are exported. A teacher's rewording is checked again.
- **Limitation to state in the paper:** drafter and checker are the same model, so "verified" means consistent with our own judge. The draft prompt forbids restating the skill's name, but a draft written to satisfy the judge can still pass without being a good course description. The teacher's accept / edit / reject decision is the real measure.
- Numbers the log now supports: share of drafts that pass the check (first try / after redraft), share accepted, share edited by the teacher, and whether edited text still passes.
- Web page (`esco_agent.webapp`): asks for the teacher's own list of expected skills **before** any suggestion and saves it with the session (never shown to the model); one MCP server and one graph for all visitors, one conversation thread each; session file rewritten after every turn.

## 2026-11-17 · Custom web page for the teacher test (built, not yet run with the real model)

Replaces the Gradio page. FastAPI + one static page in UA colours. Three screens: (1) email, stored only as a salted HMAC, and the teacher's expected skills, required, never sent to the assistant; (2) chat: the course name, learning outcomes and contents are collected with fixed messages (checks: empty, Portuguese, whole page, too short/long), then the first suggestions and the conversation, with suggestion buttons and a clickable skill list (definition, evidence, origin, ESCO link, keep/remove); (3) closing questions (SUS + 5 statements + 2 open) with the expected list, final list and conversation shown alongside.
- Logged in addition to the agent's events: whether a message started from a suggestion button, and whether each keep/remove came from the chat or a list button.
- SUS items are the standard English wording; a validated Portuguese version must be added before Portuguese-speaking teachers answer (do not use an ad-hoc translation for a reported score).

---

## Open questions and things not yet measured

- Everything above rests on one or two courses. Needed before tuning anything: a handful of courses, some outside computing, with expected skills written down **before** looking at any system's output.
- Whether the judge needs ESCO skill descriptions, not just labels. Leading candidate for the next change: it would address the word-sense acceptances (*types of strings*) and the borderline rejections.
- Judge stability: how often the same skill gets the same verdict across repeated runs, before and after adding descriptions.
- Every run file should record the commit hash and the served model; neither is written by `suggest` yet.
- Candidate settings (15 whole-text, 4 per query, 30 per judge batch) are starting guesses.
- The model alias `deepseek-chat` moves; pin the model before any reported run.
- Teacher test design: what is logged; consent or ethics approval; whether each teacher uses both POF's current flow and the assistant.
