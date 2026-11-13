"""Decide which of an occupation's essential skills a course covers, from its text.

Design (see the spec, "Decision: how missing is decided"):
  - the course is split into numbered sentences;
  - the LLM gets the sentences and the numbered essential skills, and returns,
    per skill number, "covered" with supporting sentence numbers, or "missing";
  - a validator checks every skill is judged exactly once, verdicts are valid,
    and every cited sentence number exists. Numbers instead of quotes make a
    fabricated quote impossible;
  - skills that fail validation get one retry; still failing -> "unverified".
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

# Sentence ends, or line breaks (syllabus items are often one per line).
_SPLIT = re.compile(r"(?<=[.!?;])\s+|\n+")
_BULLET = re.compile(r"^\s*(?:[-*•·]|\d+[.)])\s*")


def split_sentences(text: str) -> list[str]:
    parts = (_BULLET.sub("", p).strip() for p in _SPLIT.split(text))
    return [p for p in parts if len(p) > 2]


PROMPT = """You are checking which skills a university course covers.

Course sentences:
{sentences}

Skills required by the occupation "{occupation}":
{skills}

For EVERY skill above, decide from the course sentences alone:
- "covered": the course explicitly teaches or practises this skill, or something \
that clearly includes it. Cite every sentence number that shows it.
- "missing": the sentences don't show it. Don't infer from the course topic.

Reply with JSON only, one entry per skill number, in this form:
{{"judgements": [{{"skill": 1, "verdict": "covered", "sentences": [2, 5]}}, \
{{"skill": 2, "verdict": "missing", "sentences": []}}]}}"""

RETRY = """Your previous reply had problems:
{problems}
Reply again with JSON only, judging ONLY these skill numbers: {numbers}."""


@dataclass
class Judgement:
    skill: int
    verdict: str                  # covered | missing | unverified
    sentences: list[int] = field(default_factory=list)


def parse_and_validate(raw: str, skill_numbers: set[int], n_sentences: int
                       ) -> tuple[dict[int, Judgement], list[str]]:
    """Return the valid judgements for the expected skills, and the problems.

    Invalid or missing skills are left out of the result, so they can be retried.
    """
    try:
        data = json.loads(raw)
        items = data["judgements"]
        if not isinstance(items, list):
            raise TypeError
    except (json.JSONDecodeError, KeyError, TypeError):
        return {}, ["reply is not JSON of the form {\"judgements\": [...]}"]

    good: dict[int, Judgement] = {}
    seen: set[int] = set()
    problems: list[str] = []
    for it in items:
        if not isinstance(it, dict):
            problems.append(f"not an object: {it!r}")
            continue
        num, verdict, sents = it.get("skill"), it.get("verdict"), it.get("sentences", [])
        if not isinstance(num, int) or num not in skill_numbers:
            problems.append(f"unknown skill number {num!r}")
            continue
        if num in seen:
            problems.append(f"skill {num} judged more than once")
            good.pop(num, None)
            continue
        seen.add(num)
        if verdict not in ("covered", "missing"):
            problems.append(f"skill {num}: verdict must be covered or missing, got {verdict!r}")
            continue
        if not isinstance(sents, list) or not all(isinstance(s, int) for s in sents):
            problems.append(f"skill {num}: sentences must be a list of numbers")
            continue
        bad = [s for s in sents if not 1 <= s <= n_sentences]
        if bad:
            problems.append(f"skill {num}: sentence numbers {bad} don't exist "
                            f"(valid: 1-{n_sentences})")
            continue
        if verdict == "covered" and not sents:
            problems.append(f"skill {num}: covered but no sentence cited")
            continue
        good[num] = Judgement(num, verdict, sorted(set(sents)) if verdict == "covered" else [])

    absent = sorted(skill_numbers - seen)
    if absent:
        problems.append(f"skills not judged: {absent}")
    return good, problems


@dataclass
class CoverageResult:
    judgements: dict[int, Judgement]   # by skill number, 1-based
    first_attempt_valid: bool          # the validity-rate metric
    problems: list[str]                # from the first attempt


async def judge_coverage(llm, sentences: list[str], occupation: str,
                         skill_labels: list[str],
                         prompt_template: str = PROMPT) -> CoverageResult:
    """llm: a chat model already bound to JSON output (async .ainvoke).
    prompt_template: PROMPT (an occupation's requirements) or another template
    with the same {sentences} / {skills} fields and the same reply format."""
    numbers = set(range(1, len(skill_labels) + 1))
    prompt = prompt_template.format(
        sentences="\n".join(f"{i}. {s}" for i, s in enumerate(sentences, 1)),
        occupation=occupation,
        skills="\n".join(f"{i}. {s}" for i, s in enumerate(skill_labels, 1)),
    )
    messages = [("user", prompt)]
    reply = await llm.ainvoke(messages)
    good, problems = parse_and_validate(reply.content, numbers, len(sentences))
    first_ok = not problems

    todo = numbers - set(good)
    if todo:
        messages += [("assistant", reply.content),
                     ("user", RETRY.format(problems="\n".join(f"- {p}" for p in problems),
                                           numbers=sorted(todo)))]
        reply = await llm.ainvoke(messages)
        again, _ = parse_and_validate(reply.content, todo, len(sentences))
        good.update(again)
        for n in todo - set(again):
            good[n] = Judgement(n, "unverified")

    return CoverageResult(good, first_ok, problems)
