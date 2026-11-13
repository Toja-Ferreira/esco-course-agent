"""Coverage check: sentence splitting, validation and retry. No data or API needed."""

import asyncio
import json

from esco_agent.coverage import judge_coverage, parse_and_validate, split_sentences


def test_split_sentences():
    text = ("Objectives: design schemas. Write SQL queries!\n"
            "Syllabus:\n- the relational model\n2) normal forms; transactions\n\n")
    assert split_sentences(text) == ["Objectives: design schemas.", "Write SQL queries!",
                                     "Syllabus:", "the relational model",
                                     "normal forms;", "transactions"]


def j(*items):
    return json.dumps({"judgements": [dict(skill=s, verdict=v, sentences=n)
                                      for s, v, n in items]})


def test_valid_reply():
    good, problems = parse_and_validate(
        j((1, "covered", [2, 1, 2]), (2, "missing", [])), {1, 2}, n_sentences=3)
    assert not problems
    assert good[1].sentences == [1, 2] and good[2].verdict == "missing"


def test_each_kind_of_error_is_caught():
    cases = {
        "not json": "Sure! Skill 1 is covered.",
        "skipped skill": j((1, "covered", [1])),
        "invented skill": j((1, "covered", [1]), (2, "missing", []), (9, "missing", [])),
        "duplicate": j((1, "covered", [1]), (1, "missing", []), (2, "missing", [])),
        "bad sentence": j((1, "covered", [7]), (2, "missing", [])),
        "covered without evidence": j((1, "covered", []), (2, "missing", [])),
        "bad verdict": j((1, "partly", [1]), (2, "missing", [])),
    }
    for name, raw in cases.items():
        good, problems = parse_and_validate(raw, {1, 2}, n_sentences=3)
        assert problems, name
        assert set(good) != {1, 2} or name == "invented skill", name


class ScriptedLLM:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    async def ainvoke(self, messages):
        self.calls.append(messages)
        return type("R", (), {"content": self.replies.pop(0)})()


def test_retry_fixes_only_the_bad_skills():
    llm = ScriptedLLM([j((1, "covered", [2]), (2, "covered", [99])),   # skill 2 bad, 3 skipped
                       j((2, "missing", []), (3, "covered", [1]))])
    res = asyncio.run(judge_coverage(llm, ["a.", "b."], "x", ["s1", "s2", "s3"]))
    assert not res.first_attempt_valid and len(llm.calls) == 2
    assert "[2, 3]" in llm.calls[1][-1][1]          # retry asks only for 2 and 3
    assert [res.judgements[n].verdict for n in (1, 2, 3)] == ["covered", "missing", "covered"]


def test_still_invalid_after_retry_is_unverified():
    llm = ScriptedLLM(["nonsense", "still nonsense"])
    res = asyncio.run(judge_coverage(llm, ["a."], "x", ["s1", "s2"]))
    assert {j.verdict for j in res.judgements.values()} == {"unverified"}


def test_valid_first_time_makes_one_call():
    llm = ScriptedLLM([j((1, "missing", []))])
    res = asyncio.run(judge_coverage(llm, ["a."], "x", ["s1"]))
    assert res.first_attempt_valid and len(llm.calls) == 1
