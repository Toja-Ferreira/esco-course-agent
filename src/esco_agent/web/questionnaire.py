"""The closing questions. Shown after "Finish", saved in the session file.

SUS: the System Usability Scale (Brooke, 1996), 10 items, 1-5, scored 0-100.
The items are the standard English wording. BEFORE THE TEST: if teachers answer
in Portuguese, SUS_PT is the published, validated European Portuguese version
(never an ad-hoc translation, for a reported score).
"""

from __future__ import annotations

SUS_EN = [
    "I think that I would like to use this system frequently.",
    "I found the system unnecessarily complex.",
    "I thought the system was easy to use.",
    "I think that I would need the support of a technical person to be able to use this system.",
    "I found the various functions in this system were well integrated.",
    "I thought there was too much inconsistency in this system.",
    "I would imagine that most people would learn to use this system very quickly.",
    "I found the system very cumbersome to use.",
    "I felt very confident using the system.",
    "I needed to learn a lot of things before I could get going with this system.",
]
# European Portuguese SUS, validated: Martins, Rosa, Queirós, Silva & Rocha (2015),
# "European Portuguese Validation of the System Usability Scale (SUS)",
# Procedia Computer Science 67, 293-300, doi:10.1016/j.procs.2015.09.273.
# Wording kept exactly as published ("produto"), so the validation applies.
SUS_PT: list[str] | None = [
    "Acho que gostaria de utilizar este produto com frequência.",
    "Considerei o produto mais complexo do que necessário.",
    "Achei o produto fácil de utilizar.",
    "Acho que necessitaria de ajuda de um técnico para conseguir utilizar este produto.",
    "Considerei que as várias funcionalidades deste produto estavam bem integradas.",
    "Achei que este produto tinha muitas inconsistências.",
    "Suponho que a maioria das pessoas aprenderia a utilizar rapidamente este produto.",
    "Considerei o produto muito complicado de utilizar.",
    "Senti-me muito confiante a utilizar este produto.",
    "Tive que aprender muito antes de conseguir lidar com este produto.",
]

STATEMENTS = {
    "evidence": {
        "en": "The course sentence shown with each skill helped me decide whether to keep it.",
        "pt": "A frase da unidade curricular mostrada com cada competência ajudou-me a decidir se a mantinha.",
        "na": None},
    "conversation": {
        "en": "Being able to ask questions and ask for more skills made the list better than a fixed list would.",
        "pt": "Poder fazer perguntas e pedir mais competências tornou a lista melhor do que uma lista fixa.",
        "na": None},
    "added_value": {
        "en": "The assistant suggested relevant skills I hadn't thought of.",
        "pt": "O assistente sugeriu competências relevantes em que eu não tinha pensado.",
        "na": None},
    "text_suggestions": {
        "en": "The suggested text changes were useful for improving the course page.",
        "pt": "As alterações de texto sugeridas foram úteis para melhorar a página da unidade curricular.",
        "na": {"en": "I didn't use them", "pt": "Não usei"}},
    "vs_pof": {
        "en": "Compared with POF's current skill picker, I would rather use this assistant.",
        "pt": "Comparando com o atual seletor de competências do POF, prefiro usar este assistente.",
        "na": {"en": "I have never used it", "pt": "Nunca usei"}},
}

OPEN = {
    "most_useful": {"en": "What was the most useful part, and what got in the way?",
                    "pt": "O que foi mais útil, e o que atrapalhou?"},
    "distrust": {"en": "Was any suggestion wrong in a way that would make you distrust the tool?",
                 "pt": "Alguma sugestão estava errada de uma forma que o faria desconfiar da ferramenta?"},
    "comments": {"en": "Anything else you would like to tell us? (optional)",
                 "pt": "Mais alguma coisa que nos queira dizer? (opcional)"},
}

SCALE = {"en": ["Strongly disagree", "Strongly agree"], "pt": ["Discordo totalmente", "Concordo totalmente"]}


def questions(language: str) -> dict:
    lang = language if language in ("en", "pt") else "en"
    sus = SUS_PT if (lang == "pt" and SUS_PT) else SUS_EN
    return {
        "scale": SCALE[lang],
        "sus": sus,
        "sus_language": "pt" if sus is SUS_PT else "en",
        "statements": [{"key": k, "text": v[lang], "na": v["na"][lang] if v["na"] else None}
                       for k, v in STATEMENTS.items()],
        "open": [{"key": k, "text": v[lang]} for k, v in OPEN.items()],
    }


def sus_score(answers: list[int]) -> float:
    """Standard SUS scoring: odd items contribute (a - 1), even items (5 - a); sum x 2.5."""
    if len(answers) != 10 or not all(isinstance(a, int) and 1 <= a <= 5 for a in answers):
        raise ValueError("SUS needs 10 answers from 1 to 5")
    return 2.5 * sum((a - 1) if i % 2 == 0 else (5 - a) for i, a in enumerate(answers))


def check_answers(raw: dict) -> dict:
    """Validate a submission; returns the cleaned answers with the SUS score and the
    language the questions were shown in (the page follows POF's PT/EN switch)."""
    sus = raw.get("sus")
    score = sus_score(sus)
    statements = raw.get("statements") or {}
    clean = {}
    for key, spec in STATEMENTS.items():
        v = statements.get(key)
        if v == "na" and spec["na"]:
            clean[key] = "na"
        elif isinstance(v, int) and 1 <= v <= 5:
            clean[key] = v
        else:
            raise ValueError(f"missing or invalid answer for '{key}'")
    opens = raw.get("open") or {}
    lang = raw.get("language") if raw.get("language") in ("en", "pt") else None
    return {"sus": sus, "sus_score": score, "statements": clean,
            "questions_language": lang,
            "sus_language": "pt" if (lang == "pt" and SUS_PT) else "en",
            "open": {k: str(opens.get(k, "")).strip()[:3000] for k in OPEN}}
