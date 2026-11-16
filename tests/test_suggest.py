"""Suggestions: the judge filters linker candidates and attaches evidence. No data or API."""

import asyncio
import json

from esco_agent.suggest import (QUERY_PROMPT, SUGGEST_PROMPT, gather_candidates, judge_line,
                                load_sections, numbered_sentences, propose_queries,
                                sentence_queries, stability, suggest_skills,
                                unique_sentences)

COURSE = "Students write programs in C. They debug code with gdb.\n- pointers and memory"
CANDS = [{"id": "a", "label": "Objective-C", "type": "knowledge", "score": 0.83},
         {"id": "b", "label": "debug software", "type": "skill/competence", "score": 0.61},
         {"id": "c", "label": "computer programming", "type": "knowledge", "score": 0.50}]


class Scripted:
    def __init__(self, *replies):
        self.replies, self.prompts = list(replies), []

    async def ainvoke(self, messages):
        self.prompts.append(messages[0][1])
        return type("R", (), {"content": self.replies.pop(0)})()


def reply(*items):
    return json.dumps({"judgements": [dict(skill=s, verdict=v, sentences=n) for s, v, n in items]})


def test_judge_filters_candidates_and_attaches_evidence():
    llm = Scripted(reply((1, "missing", []), (2, "covered", [2]), (3, "covered", [1, 3])))
    r = asyncio.run(suggest_skills(CANDS, llm, COURSE))
    assert [s["label"] for s in r["suggested"]] == ["debug software", "computer programming"]
    assert [s["label"] for s in r["rejected"]] == ["Objective-C"]
    assert r["suggested"][0]["evidence"] == [
        {"sentence": 2, "section": "Course", "text": "They debug code with gdb."}]
    assert all(e["text"] in COURSE for s in r["suggested"] for e in s["evidence"])
    assert r["first_attempt_valid"] and r["suggested"][0]["score"] == 0.61
    assert "1. Objective-C" in llm.prompts[0] and "3. pointers and memory" in llm.prompts[0]


def test_invalid_judgement_is_unverified_not_suggested():
    bad = reply((1, "missing", []), (2, "covered", [9]), (3, "covered", [1]))
    r = asyncio.run(suggest_skills(CANDS, Scripted(bad, "nonsense"), COURSE))
    assert [s["label"] for s in r["unverified"]] == ["debug software"]
    assert [s["label"] for s in r["suggested"]] == ["computer programming"]
    assert not r["first_attempt_valid"]


def test_prompt_formats():
    SUGGEST_PROMPT.format(sentences="1. x", skills="1. y", occupation="")


def test_repeated_paragraphs_are_judged_once():
    assert unique_sentences("Write C programs. Use pointers.\nWrite C programs.") == \
        ["Write C programs.", "Use pointers."]


def test_candidates_come_from_whole_text_and_each_query():
    calls = []

    async def link(text, k):
        calls.append((text, k))
        if "\n" in text:                       # whole course
            return [{"id": "c", "label": "C++", "type": "knowledge", "score": 0.8}]
        if "memory" in text:
            return [{"id": "m", "label": "manage memory", "type": "skill", "score": 1.0},
                    {"id": "c", "label": "C++", "type": "knowledge", "score": 0.9}]
        return []

    course = "Objectives\nStudents write programs in the C language.\nThey manipulate files and dynamic memory."
    cands = asyncio.run(gather_candidates(link, course, ["manage dynamic memory", "C"]))
    assert [c["id"] for c in cands] == ["c", "m"]          # deduplicated, whole text first
    assert cands[0]["score"] == 0.9                        # best score kept
    assert cands[0]["found_by"] == ["whole text", "manage dynamic memory"]
    assert [t for t, _ in calls][1:] == ["manage dynamic memory", "C"]
    assert sentence_queries(course) == ["Students write programs in the C language.",
                                        "They manipulate files and dynamic memory."]  # heading skipped


def test_propose_queries_parses_and_cleans():
    ok = Scripted(json.dumps({"queries": ["debug software", " Linux ", "debug software", "", 7]}))
    assert asyncio.run(propose_queries(ok, "c")) == ["debug software", "Linux"]
    assert "c" in ok.prompts[0] and QUERY_PROMPT.format(course="x")
    for bad in ["not json", json.dumps({"queries": "debug"}), json.dumps(["a"])]:
        assert asyncio.run(propose_queries(Scripted(bad), "c")) == []


def test_many_candidates_are_judged_in_batches():
    cands = [{"id": str(i), "label": f"skill {i}", "type": "skill", "score": 0.5} for i in range(70)]

    class EchoJudge:
        calls = 0

        async def ainvoke(self, messages):
            EchoJudge.calls += 1
            n = messages[0][1].split("Candidate skills")[1].split("For EVERY")[0].count("\n") - 1
            return type("R", (), {"content": reply(*[(i, "covered", [1]) for i in range(1, n + 1)])})()

    r = asyncio.run(suggest_skills(cands, EchoJudge(), "Students write programs in C."))
    assert EchoJudge.calls == 3 and r["judge_calls"] == 3
    assert len(r["suggested"]) == 70 and [s["id"] for s in r["suggested"]] == [str(i) for i in range(70)]


SECTIONS = {"Learning outcomes": "Students write programs in C. They use strings and pointers.",
            "Contents": "Students write programs in C.\n- dynamic memory\n- text files"}


def test_sections_are_numbered_across_fields_and_deduplicated():
    sents = numbered_sentences(SECTIONS)
    assert [(s["section"], s["text"]) for s in sents] == [
        ("Learning outcomes", "Students write programs in C."),
        ("Learning outcomes", "They use strings and pointers."),
        ("Contents", "dynamic memory"), ("Contents", "text files")]


def test_evidence_cites_its_section_and_judge_sees_definitions():
    cands = [{"id": "s", "label": "types of strings", "type": "knowledge", "score": 0.9,
              "description": "Vibrating elements that generate sounds in string instruments."},
             {"id": "m", "label": "memory", "type": "knowledge", "score": 0.5}]
    llm = Scripted(reply((1, "missing", []), (2, "covered", [3])))
    r = asyncio.run(suggest_skills(cands, llm, SECTIONS))
    prompt = llm.prompts[0]
    assert "1. types of strings: Vibrating elements that generate sounds" in prompt
    assert "2. memory\n" in prompt                      # no definition -> label only
    assert "3. [Contents] dynamic memory" in prompt
    assert r["suggested"][0]["evidence"] == [
        {"sentence": 3, "section": "Contents", "text": "dynamic memory"}]

    llm = Scripted(reply((1, "missing", []), (2, "missing", [])))
    asyncio.run(suggest_skills(cands, llm, SECTIONS, use_descriptions=False))
    assert "Vibrating" not in llm.prompts[0]


def test_long_definitions_are_cut_at_a_word():
    line = judge_line({"label": "x", "description": "word " * 200})
    assert len(line) < 270 and line.endswith("word…")


def test_load_sections(tmp_path):
    j = tmp_path / "c.json"
    j.write_text(json.dumps({"Learning outcomes": " a ", "Contents": "b", "Empty": " "}))
    assert load_sections(str(j)) == {"Learning outcomes": "a", "Contents": "b"}
    t = tmp_path / "c.txt"
    t.write_text("plain text\n")
    assert load_sections(str(t)) == {"Course": "plain text"}


def test_stability_counts_flips():
    def run(sug, rej):
        mk = lambda ids: [{"id": i, "label": i.upper()} for i in ids]
        return {"suggested": mk(sug), "rejected": mk(rej), "unverified": []}
    st = stability([run("ab", "c"), run("ac", "b"), run("a", "")])
    assert st["runs"] == 3 and st["always"] == 1 and st["unstable"] == 2
    assert [(r["label"], r["suggested"], r["candidate"]) for r in st["rows"]] == [
        ("A", 3, 3), ("B", 1, 2), ("C", 1, 2)]
    assert st["judge_flips"] == 2
