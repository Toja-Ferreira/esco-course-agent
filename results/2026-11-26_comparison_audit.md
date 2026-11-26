# Audit: ESCO Open Access Mapping tool vs skill linker vs agent (26 Nov 2026)

Scope: `src/esco_agent/compare_tool.py`, `labelsheet.agree` (with the uncommitted fix that matches rows by course, row and skill), `results/esco-tool/*.json`, the 608-row line sheet (`2026-11-26_tool_vs_linker.csv` labelled by Gemini, `_claude.csv` labelled by Claude), `.key.json`, `_score*.json`, `2026-11-20_labels.csv` (93 agent rows, Gemini) and `_claude.csv`, `2026-11-26_tool_coverage.json`, the 25 Nov entry in `docs/findings.md`, and `README.md`.

All re-analyses are in `/tmp/claude-1000/-home-toja-esco-course-agent/a176e6b5-6f09-4239-ab33-1c8e34e2e5f9/scratchpad/audit/` (`a1.py`–`a4.py`). They are independent of `compare_tool.score` and `coverage`.

Conflict of interest: I (Claude) produced `_claude.csv` in an earlier turn of this session. Everything I say below about those labels is a self-report.

---

## 1. Verdict per claim

**Claim 1, "the tool's top 3 is more precise than the linker's; close at rank 1": supported with changes.** The top-3 part holds under both raters as a *paired* difference, not only as point estimates. Strict P@3: tool − linker = +0.061 [+0.023, +0.100] (Gemini) and +0.119 [+0.073, +0.166] (Claude). Lenient P@3: +0.133 / +0.165, both with intervals well above 0.

"Close at rank 1" holds only for Gemini's strict labels: +0.033 [−0.041, +0.106], exact McNemar p = 0.52. With Claude's labels the tool is better at rank 1 too: +0.106 [+0.024, +0.187], p = 0.02.

Part of the tool's advantage comes from 22 tool labels that resolve to no English ESCO skill, such as "travel, tourism and leisure", "programming" and "languages". The rater judged these by name alone and marked 21/22 right or borderline (Gemini) and 20/22 right (Claude). Counted as wrong, the strict P@3 difference under Gemini shrinks to +0.031 [−0.006, +0.067].

The claim must name the rater, give the paired difference, and say the strict top-3 advantage is partly carried by unresolvable labels.

**Claim 2, "neither retriever is precise alone (a third to a half fully right), while the agent's checked list is 0.79–0.84": supported with changes.** "A third to a half" fits rank 1 only: strict P@1 is 0.32–0.46 across systems and raters. Over the top 3 it is a fifth to a third (0.18–0.32).

The agent's 0.79–0.84 is a different unit. It is precision per suggestion after an LLM judge has filtered candidates, with evidence chosen by the agent (67 of 93 rows cite several sentences), labelled on a different sheet. The retriever figures are per ranked candidate, against one line.

The same concepts score better on the agent's sheet. For the 71 agent skills that also appear on the line sheet, Gemini marks 64/71 right on the agent sheet (0.90) but 54/71 (0.76) on the line sheet, using the best verdict over any line. For Claude the figures are 61/71 vs 51/71 (0.72). The gap is real (about 0.75 vs 0.24–0.46), but the 0.79–0.84 vs "a third" contrast overstates it. It should be stated as two different measurements, with the same-sheet figure alongside.

**Claim 3, "the agent needs 93 suggestions to review against the tool's 1,845 candidates": not supported as worded.**
- 1,845 = 123 lines × K=15. Only the top 5 per line were recorded, so 1,230 of these candidates were never seen.
- K=15 is a user setting; a teacher could choose K=3 (369 candidates, 206 distinct skills) or K=1 (123).
- "93 vs 1,845" compares a filtered output with an unfiltered list at the most generous setting.

A fair version compares review effort per fully right skill:

| system | reviews per fully right skill |
|---|---|
| agent | 93 / 78 = 1.2 |
| tool, K=1 | 123 / 29 = 4.2 (Gemini), 123 / 33 = 3.7 (Claude) |
| tool, K=3 | 369 / 59 = 6.3 (Gemini), 369 / 64 = 5.8 (Claude) |

The agent is roughly 3–5× more efficient, not 20×.

**Claim 4, "the agent's relative recall is 0.90 (78 / (78 + 9))": supported with changes.** The arithmetic reproduces exactly: `coverage()` gives only_tool_top3 right 9, borderline 30, wrong 111. But the denominator is narrow:
1. Only the tool's top 3 is in the pool. Of the 37 "agent-only" skills, 8 are at the tool's ranks 4–5, and about 110 distinct tool concepts at ranks 4–5 are unlabelled. Ranks 6–15 were not recorded.
2. The 22 unresolvable tool labels are excluded here (`compare_tool.py:166`, `if u`), yet they count in claim 1.
3. The 9 come from the line sheet, while the 78 come from the more generous agent sheet (see claim 2).
4. The 25 noise lines (8 with real content) are outside the pool.
5. The result depends on the rater: 73/(73+15) = 0.83 with Claude's labels, 0.81 if the linker's own right finds are added to the pool. Counting partly right skills, it is 0.61 (Gemini) or 0.48 (Claude).

If tool ranks 4–5 were right at the tool-only top-3 rate (9/150), about 7 more right skills would join the pool, giving 78/94 ≈ 0.83. "0.90" is defensible only as "relative to the tool's top 3, strict, Gemini labels"; the honest summary is a range of about 0.8–0.9.

**Claim 5, "rater agreement (Gemini vs Claude) supports the labels": supported with changes.** On the 608 rows, agreement is 0.78 [0.75, 0.81], κ 0.65 [0.60, 0.70], AC1 0.69 [0.64, 0.73]. Of the 132 disagreements, 130 are one step apart, and they are systematic: Claude rates topic headings and teaching skills higher.

That supports the *ordering* claims (P@3 tool > linker under both raters) but not exact values. Rater choice moves tool P@1 strict from 0.35 to 0.46 and changes the rank-1 conclusion.

Both raters are LLMs. Agreement between them shows a shared reading of the rule, not that the labels are right. The only human anchor is the 19 rows labelled on 20 Nov (`b067699`, described in findings as a human check): the current Gemini sheet matches 17 of them.

---

## 2. Issues ranked by severity

### Would change a conclusion

**C1. The labels are attributed to the wrong rater.** The 25 Nov findings entry says the 608 rows were "labelled blind by the author … in about 15 minutes, by one person", but the sheet was labelled by Gemini. The timestamps fit an LLM (sheet written 22:31:02, labelled file 22:46:58: 1.5 s per row), and there is a `…_labelled_complete.csv:Zone.Identifier` browser-download marker in `results/`. Commit `d91ab1f` calls the 93-row sheet "full human labels"; the working-tree diff of README and findings already corrects that sheet to Gemini. No claim here rests on human labels.
*Fix:* say "labelled by an LLM (Gemini), second LLM rater Claude, no human labels" in findings 25 Nov and in any README row; drop "by one person".

**C2. Claim 2 compares two different units on two different sheets** (see §1). Evidence: same-concept verdicts disagree between the sheets. On Gemini's labels, 7 concepts are right on the agent sheet but borderline on the line sheet, and only 1 goes the other way (overlap of 56). Of the 93 agent rows, 67 cite more than one sentence.
*Fix:* report the agent's precision on its own sheet *and* on the line sheet (0.72–0.76 for the 71 shared skills), and name the units.

**C3. Claim 4's pool is the tool's top 3 only, and asymmetric** (see §1, items 1–5).
*Fix:* report "relative to the tool's top 3", give the range 0.81–0.90, and list the exclusions. Better still, label the tool's ranks 4–5 that are already in the JSON (about 110 distinct concepts not in the agent's list) and the 37 agent-only skills on their best line, then recompute.

**C4. "Close at rank 1" depends on the rater, and the strict top-3 advantage depends on unresolvable labels.** Both are shown by the paired analyses in §3.
*Fix:* report paired differences for both raters and the unknown-labels-as-wrong sensitivity.

**C5. Claim 3 uses the tool's most generous setting and counts unseen candidates.**
*Fix:* use the per-right-skill review load at K=1 and K=3 (§3.4).

### Would change a number

**N1. De-duplication is by label string, not by concept, and it shifts ranks** (`compare_tool.py:90-98`, `course_top`).
- On 4 lines a repeated label is removed and the tool's 4th or 5th candidate moves into the top 3: the tool gets an extra candidate.
- On 6 lines two *different* labels in the top 3 resolve to the same concept, for example "mathematical subjects"/"mathematics", "confidence"/"confidence showing" and "work in groups"/"working in teams". `build_rows` keeps only the first rank (`:67-68`), so the tool has 2 concepts in its top 3 and P@3 is averaged over 2.
- Scored "as displayed" (each of the raw first 3 slots gets its concept's verdict), tool P@3 strict rises from 0.242 to 0.255 (Gemini) and from 0.322 to 0.333 (Claude). The direction is unchanged.

*Fix:* dedupe by resolved URI and fill from rank 4–5, or score as displayed. Use one rule and state it.

**N2. Ambiguous alternative labels resolve to the lexicographically smallest URI** (`compare_tool.py:50`, `sorted(uris)[0]`), not "the first in order" as the docstring says. That is effectively arbitrary. 4 rows are affected:
- "confidence" → *show confidence* (wrong)
- "speak foreign languages" → *speak different languages* (wrong)
- "electrical current" → *electricity principles*
- "Cascading Style Sheets" → *style sheet languages*

*Fix:* pick among the candidate concepts using the tool's `type` field (knowledge vs skill), or keep ambiguous rows out of the scores and report them.

**N3. Unresolvable tool labels are handled inconsistently.** They are scored in P@k (`build_rows:65`, as `unknown:<label>`) and judged by name. They are dropped from `coverage` (`:166-167`) and from `best_verdicts` (`:149`). Effect on claim 1: see C4. Effect on claim 4: counting the 6 distinct right unknown labels (Gemini) as tool finds gives 78/(78+9+6) = 0.84.
*Fix:* either treat them as non-ESCO (exclude or count them wrong in both analyses) or include them in both. All 22 are typed "skill/competence" by the tool, yet many look like broad field names ("law", "biochemistry", "travel, tourism and leisure").

**N4. The reported intervals are unpaired.** `compare_tool.score` (`:126-133`) bootstraps each metric separately. The published tool and linker intervals overlap, e.g. P@3 strict [0.196, 0.289] vs [0.141, 0.225], which hides that the paired difference excludes 0.
*Fix:* add a paired bootstrap of tool − linker (resample lines once per draw and compute both) and report it as the result.

**N5. Clustering.** Lines are resampled i.i.d. while courses differ. Per-course differences in P@1 strict flip sign: molecular-cell-biology −0.11 and plurilingual −0.07 (Gemini). A course-cluster bootstrap gives intervals of similar width (§3.1), but with only 6 clusters it is unreliable. Agreement is bootstrapped over rows while rows sit within lines; a line-cluster bootstrap gives [0.747, 0.816], close to the reported [0.75, 0.81].
*Fix:* report per-course differences, and call the line-level interval approximate.

**N6. Multiple comparisons.** Each rater has 8 precision values and 4 tool − linker differences, and there are 2 raters, 2 sheets and the recall and load figures. With Holm over the 4 differences per rater, the Gemini P@1 lenient difference (p ≈ 0.028) no longer reaches 0.05 (0.056); the P@3 differences survive. Under Claude all 4 survive.
*Fix:* name P@3 strict as the primary metric in advance and treat the others as secondary.

**N7. The old `agree` bug** matched by (course, skill) and kept 387 of 608 rows. It is fixed in the uncommitted `labelsheet.py`, and `_agreement.json` (n = 608) uses the fix. The 93-row agreement (n = 93, no repeated skills) was not affected. §1 of `2026-11-26_tool_vs_linker_review.md` still shows the 387-row output.
*Fix:* commit the fix with a test, and mark the 387-row block in the review as superseded.

### Wording only

**W1. The wrong-sense examples are one-sided.** Findings lists only the tool's errors. The linker makes as many at rank 1 and 2:
- "Its association in series and parallel." → *parallel park vehicles* (linker rank 1)
- "Polynomial and rational functions." → *feed livestock* (rank 1), *theology* (rank 2)
- "4 - Node and mesh laws." → *property law* (rank 1)
- "2 - Ohm's Law;" → *teach principles of law* (rank 2)

"Ohm's Law → laws of obligation" was ranked by *both* systems (tool 1, linker 3). Rows only the linker proposed are wrong 181/245 (74%), rows only the tool proposed 129/239 (54%) (Gemini).
*Fix:* give examples from both, or the rates.

**W2. "Linker" here means the raw retriever on single lines** (`link(line, k=3)`, `:69`). The agent uses it differently: LLM search phrases, 6 per phrase, then a judge. Say so, or readers will read claim 1 as "the agent's retrieval is worse".

**W3. "22 of its names are not English ESCO v1.2.1 labels"** should read "could not be matched to an English ESCO v1.2.1 skill label (several look like broad field names)".

**W4. Noise lines.** Both retrievers lose the 8 content lines the tool marks as noise, so claim 1 is unaffected. The agent reads them, so claims 3 and 4 favour the agent slightly. State this.

**W5. Fragments.** The tool's split produces fragments, e.g. "C8. Conventional PCR and RT-PCR:", "diode and the transistor." and "Its association in series and parallel.". Both systems face the same fragments, but the agent cites sentences with their section header ("[Learning outcomes] …"). This is one more reason the agent's precision is not comparable with the retrievers' (claim 2).

**W6. The tool output was transcribed by hand from one run, with no screenshot or HTML saved.** Keep the raw page or screenshots so the JSON can be checked.

### Blinding (item B)

- **The 22 unknown rows reveal their source.** The definition "(not an English ESCO skill label)" appears only on tool rows, and the rater instruction tells raters to judge those rows "by the skill name". Every other row shows the preferred label and the ESCO definition, so those are blind.
- **Rater choice mainly moves the tool's numbers.** Claude and Gemini agree on 0.83 of linker-only rows but 0.75 of tool-only rows. Claude marks 66 tool-only rows right against Gemini's 44, but on linker-only rows they are nearly equal (24 vs 23). This is mostly item type (the tool proposes more broad knowledge labels on topic headings, which the two raters read differently), but it means the size of claim 1 depends on the rater.
- **Claude's independence.** In this session's transcript, Claude's labels were written before Gemini's verdict column was read: the first prints showed no verdicts, and `agree` ran after the file was filled. But the rule Claude was given was written after the sheet existed and quotes a row from it ("Ohm's Law → canon law"). Gemini's prompt is not saved, so it cannot be confirmed that both raters got the same rule.

### Labels (item G)

- Two LLM raters and no human labels. Gemini took about 1.5 s per row.
- The rule's two ambiguous cases were never resolved, and the sheet was not relabelled:
  - teaching skills (17 disagreements)
  - topic headings for knowledge skills (29 disagreements)

Effect on each claim:

| claim | effect |
|---|---|
| 1 | direction at top 3 stable; the rank-1 result flips with the rater |
| 2 | retriever numbers move by ~0.1 between raters; the agent's figure was labelled on another sheet and day (20/23 Nov) |
| 3 | per-right-skill loads move by ~10% |
| 4 | 0.83–0.90 |
| 5 | agreement is moderate and the disagreement is systematic, so it supports orderings, not values |

---

## 3. Re-analyses (scratch scripts, all on the committed data)

### 3.1 Recomputed scores and paired differences (`a1.py`, `a4.py`)

The recomputed values match `_score.json` and `_score_claude.json` exactly. All 123 lines have a tool rank 1 and a linker rank 1.

Paired tool − linker differences (4,000 draws). Columns: line bootstrap; course-cluster bootstrap; two-sided bootstrap p.

| rater | metric | tool | linker | diff | line 95% | course 95% | p |
|---|---|---|---|---|---|---|---|
| Gemini | P@1 strict | 0.350 | 0.317 | +0.033 | [−0.041, +0.106] | [−0.042, +0.090] | 0.44 |
| Gemini | P@1 lenient | 0.650 | 0.545 | +0.106 | [+0.016, +0.195] | [+0.023, +0.188] | 0.03 |
| Gemini | P@3 strict | 0.242 | 0.182 | +0.061 | [+0.020, +0.099] | [+0.021, +0.098] | 0.002 |
| Gemini | P@3 lenient | 0.526 | 0.393 | +0.133 | [+0.077, +0.185] | [+0.065, +0.225] | <0.001 |
| Claude | P@1 strict | 0.455 | 0.350 | +0.106 | [+0.024, +0.187] | [+0.068, +0.146] | 0.015 |
| Claude | P@1 lenient | 0.732 | 0.634 | +0.098 | [+0.016, +0.179] | [+0.016, +0.178] | 0.02 |
| Claude | P@3 strict | 0.322 | 0.203 | +0.119 | [+0.071, +0.165] | [+0.064, +0.158] | <0.001 |
| Claude | P@3 lenient | 0.661 | 0.496 | +0.165 | [+0.114, +0.220] | [+0.108, +0.227] | <0.001 |

Discordant rank-1 pairs, strict (lines where exactly one system's rank 1 is right):

| rater | tool only | linker only | exact McNemar p |
|---|---|---|---|
| Gemini | 13 | 9 | 0.52 |
| Claude | 21 | 8 | 0.02 |

Per-course P@1 strict difference (Gemini): industrial +0.08, maths +0.14, molecular −0.11, plurilingual −0.07, programming +0.04, tourism +0.05.

### 3.2 Sensitivity to the scoring rules (`a4.py`)

Paired tool − linker difference, with 95% interval:

| variant | Gemini P@1 strict | Gemini P@3 strict | Claude P@1 strict | Claude P@3 strict |
|---|---|---|---|---|
| as computed | +0.033 [−0.04, +0.11] | +0.061 [+0.02, +0.10] | +0.106 [+0.02, +0.19] | +0.119 [+0.07, +0.17] |
| as displayed (raw top-3 slots) | +0.033 | +0.073 [+0.03, +0.12] | +0.106 | +0.130 [+0.08, +0.18] |
| unknown labels counted wrong | +0.016 [−0.06, +0.09] | **+0.031 [−0.006, +0.067]** | +0.049 [−0.02, +0.12] | +0.064 [+0.02, +0.11] |
| unknown labels dropped (next rank moves up) | +0.016 | +0.044 [+0.01, +0.08] | +0.049 | +0.081 [+0.04, +0.12] |

Lenient P@3 stays positive with an interval above 0 in every variant. With unknown labels counted wrong, Claude's rank-1 advantage also loses significance (+0.049 [−0.02, +0.12]).

### 3.3 Recall variants (`a3.py`)

| variant | Gemini | Claude |
|---|---|---|
| claimed: agent right / (agent right + tool-only top-3 right) | 78/87 = **0.897** | 73/88 = 0.830 |
| + linker-only top-3 right | 0.897 (0 linker-only right) | 73/90 = 0.811 |
| + the tool's unresolvable labels that are right (6 distinct) | 78/93 ≈ 0.84 | — |
| + estimate for tool ranks 4–5 (~110 unlabelled, at 9/150) | ≈ 0.83 | — |
| lenient (right + borderline) | 93/152 = 0.61 | 91/190 = 0.48 |
| tool top 3 on the same pool (symmetric) | 59/87 = 0.68 | 64/90 = 0.71 |

8 of the 37 agent-only skills are at the tool's ranks 4–5. Ranks 6–15 are unknown.

Same concepts on both sheets (56 in both the agent's list and the tool's top 3, plus 15 agent-only skills in the linker's top 3; 71 in total): fully right on the agent sheet 64/71 (Gemini) and 61/71 (Claude); best verdict on the line sheet 54/71 (0.76) and 51/71 (0.72).

### 3.4 Review load (`a3.py`)

| setting | candidates shown | distinct skills | distinct fully right (Gemini / Claude) | reviews per right skill |
|---|---|---|---|---|
| tool K=15 (as run) | 1,845 (1,230 never recorded) | unknown | unknown | — |
| tool K=5 | 615 | 324 | not labelled beyond rank 3 | — |
| tool K=3 | 369 | 206 | 59 / 64 | 6.3 / 5.8 |
| tool K=1 | 123 | — | 29 / 33 | 4.2 / 3.7 |
| agent | 93 | 93 | 78 / 73 | 1.2 / 1.3 |

### 3.5 Agreement checks (`a4.py`)

| measure | value |
|---|---|
| raw agreement, all 608 rows | 0.783 |
| raw agreement without the 22 unknown rows | 0.790 |
| line-cluster 95% interval for raw agreement | [0.747, 0.816] |
| agreement on tool-only rows | 0.753 |
| agreement on linker-only rows | 0.833 |
| agreement on rows proposed by both | 0.742 |

The 19 early labels on the 93-row sheet (`b067699`) match the current sheet on 17 of 19.

---

## 4. Exact wording

**Claim 1 (README row and findings):**
> **Retriever vs retriever** (ESCO Open Access Mapping tool vs this project's linker on raw lines): same 123 lines (the tool's split, without its noise lines), top 3 of each, 608 rows labelled blind by an LLM (Gemini) and re-labelled by a second LLM (Claude). Over the top 3 the tool is more precise: strict P@3 0.24 vs 0.18 (paired difference +0.06 [+0.02, +0.10]; Claude's labels 0.32 vs 0.20, +0.12 [+0.07, +0.17]). At rank 1 the difference depends on the rater: +0.03 [−0.04, +0.11] (Gemini), +0.11 [+0.02, +0.19] (Claude). 22 of the tool's names match no English ESCO skill and were judged by name; counted as wrong, the strict top-3 difference is +0.03 [−0.01, +0.07] (Gemini). The linker here is the bare retriever, not the agent's pipeline.

**Claim 2:**
> Neither retriever is precise on single lines: 32–46% of rank-1 candidates and 18–32% of top-3 candidates are fully right (two LLM raters). The agent's first list is a different measurement: suggestions already filtered by an LLM judge, each with the sentences it cites. 79–84% of them are fully right on their own sheet, and 72–76% for the 71 of them that also appear on the retriever sheet when judged there against single lines.

**Claim 3:**
> Review effort: the agent shows 93 suggestions, 78 of them fully right (about 1.2 reviews per right skill). The tool shows K candidates on each of 123 lines: 1,845 at K=15 (the setting used), 369 at K=3 (206 distinct skills, 59–64 fully right; about 6 reviews per right skill) and 123 at K=1 (29–33 fully right; about 4). The agent's own LLM filtering is not counted as review effort.

**Claim 4:**
> Relative recall against the tool's top 3: of the fully right skills found by the agent or by the tool's top 3, the agent has 78/87 = 0.90 (Gemini labels; 0.83 with Claude's). This is an upper bound: the tool's ranks 4–15 were not labelled (8 of the 37 skills only the agent found are at the tool's ranks 4–5), lines the tool marks as noise are excluded, and the agent's skills were judged with its own evidence. Counting partly right skills, it is 0.48–0.61.

**Claim 5:**
> Labels: two LLM raters (Gemini, Claude) and no human labels. On the 608 retriever rows they agree on 0.78 of rows (κ 0.65 [0.60, 0.70], AC1 0.69). Almost all disagreements are one step apart, and they are systematic: Claude rates topic headings and teaching skills higher. Orderings that hold under both raters are reported as findings; values are given for both. On the 93 agent rows: agreement 0.92, κ 0.76.

**Findings, 25 Nov caveat line** (replaces "labelled in about 15 minutes, by one person"):
> Caveat: both raters are LLMs; Gemini labelled the 608 rows in about 15 minutes; the labelling rule leaves teaching skills and topic headings ambiguous, and the sheet was not relabelled after the rule was clarified.
