"""The closing questions. Shown after "Finish", saved in the session file.

SUS: the System Usability Scale (Brooke, 1996), 10 items, 1-5, scored 0-100.
The items are the standard English wording. BEFORE THE TEST: if teachers answer
in Portuguese, replace SUS_PT with a published, validated Portuguese version
(do not use an ad-hoc translation for a reported score). Until then the English
items are shown in both languages.
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
SUS_PT: list[str] | None = None   # validated Portuguese SUS goes here

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
    """Validate a submission; returns the cleaned answers with the SUS score."""
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
    return {"sus": sus, "sus_score": score, "statements": clean,
            "open": {k: str(opens.get(k, "")).strip()[:3000] for k in OPEN}}
