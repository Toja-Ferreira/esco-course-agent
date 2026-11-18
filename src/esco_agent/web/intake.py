"""The chat's opening: collecting the course text, with fixed messages (no LLM).

Same principle as the assistant's fixed first turn: this part must not go wrong
in unpredictable ways, and it costs no API calls. Stages:
    name -> outcomes -> contents -> ready
Each answer is checked before moving on.
"""

from __future__ import annotations

import re

MAX_SECTION_CHARS = 6000
MIN_SECTION_WORDS = 8

MESSAGES = {
    "pt": {
        "hello": ("Olá! Vou ajudar a escolher as competências ESCO que descrevem a sua unidade "
                  "curricular. Primeiro, como se chama a unidade curricular?"),
        "hello_named": ("Olá! Vamos escolher as competências ESCO de **{name}**. Para começar, cole "
                        "os **resultados de aprendizagem** da página da unidade curricular, **em "
                        "inglês** (a versão inglesa da página)."),
        "ask_outcomes": ("Obrigado. Agora cole os **resultados de aprendizagem** da página da "
                         "unidade curricular, **em inglês** (a versão inglesa da página)."),
        "ask_contents": ("Agora cole os **conteúdos** (programa), também em inglês. Se a página "
                         "não tiver conteúdos, escreva \"não tenho\"."),
        "empty": "Não recebi texto. Pode colar novamente?",
        "name_long": "Isso parece mais do que um nome. Qual é só o nome da unidade curricular?",
        "too_short": ("Este texto é muito curto para encontrar competências. Pode colar a secção "
                      "completa?"),
        "portuguese": ("Este texto parece estar em português. Preciso da versão **inglesa** da "
                       "página da unidade curricular. Pode colá-la?"),
        "too_long": (f"Este texto tem mais de {MAX_SECTION_CHARS} caracteres. Pode colar só "
                     "esta secção?"),
        "whole_page": ("Isto parece ser a página inteira (vejo títulos como {headings}). Pode "
                       "colar só a secção pedida?"),
        "working": "A ler a unidade curricular e a procurar competências. Pode demorar um minuto…",
    },
    "en": {
        "hello": ("Hello! I'll help you choose the ESCO skills that describe your course. First, "
                  "what is the course called?"),
        "hello_named": ("Hello! Let's choose the ESCO skills for **{name}**. To begin, paste the "
                        "course's **learning outcomes** from its page, **in English**."),
        "ask_outcomes": ("Thank you. Now paste the course's **learning outcomes** from its page, "
                         "**in English**."),
        "ask_contents": ("Now paste the **contents** (syllabus), also in English. If the page has "
                         "no contents section, write \"none\"."),
        "empty": "I didn't receive any text. Could you paste it again?",
        "name_long": "That looks like more than a name. What is just the course's name?",
        "too_short": "This text is too short to find skills in. Could you paste the whole section?",
        "portuguese": ("This text seems to be in Portuguese. I need the **English** version of the "
                       "course page. Could you paste it?"),
        "too_long": (f"This text is over {MAX_SECTION_CHARS} characters. Could you paste just "
                     "this section?"),
        "whole_page": ("This looks like the whole page (I can see headings such as {headings}). "
                       "Could you paste just the section I asked for?"),
        "working": "Reading the course and looking for skills. This can take a minute…",
    },
}

NONE_ANSWERS = {"não tenho", "nao tenho", "não tem", "nao tem", "não", "nao", "none", "no",
                "n/a", "-", "nenhum", "nenhuns", "sem conteúdos", "sem conteudos"}

_PT = {"de", "que", "para", "os", "das", "dos", "uma", "um", "com", "não", "são", "pelo",
       "pela", "estudantes", "alunos", "aprendizagem", "conteúdos", "ao", "às", "na", "no"}
_EN = {"the", "and", "of", "to", "students", "will", "be", "with", "for", "is", "are",
       "learning", "this", "on", "in", "their"}
_OTHER_SECTIONS = ["assessment", "evaluation", "bibliography", "teaching methods",
                   "requisites", "prerequisites", "avaliação", "bibliografia",
                   "métodos de ensino", "metodologias de ensino", "requisitos"]


def looks_portuguese(text: str) -> bool:
    words = re.findall(r"[a-zà-ú]+", text.lower())
    pt = sum(w in _PT for w in words) + 3 * len(re.findall(r"ção|ções", text.lower()))
    en = sum(w in _EN for w in words)
    return pt > en and pt >= 3


def other_headings(text: str) -> list[str]:
    lines = [ln.strip().rstrip(":").lower() for ln in text.splitlines() if ln.strip()]
    return [h for h in _OTHER_SECTIONS if h in lines]


def check_section(text: str) -> str | None:
    """None if the text is usable, else the key of the message to send back."""
    if not text.strip():
        return "empty"
    if len(text) > MAX_SECTION_CHARS:
        return "too_long"
    if len(other_headings(text)) >= 2:
        return "whole_page"
    if looks_portuguese(text):
        return "portuguese"
    if len(text.split()) < MIN_SECTION_WORDS:
        return "too_short"
    return None


def check_name(text: str) -> bool:
    """A course name: not empty, one line, not a pasted section."""
    text = (text or "").strip()
    return bool(text) and len(text) <= 150 and text.count("\n") <= 1


def step(stage: str, text: str, language: str) -> dict:
    """One intake answer. Returns {"reply", "stage", and "name"/"outcomes"/"contents"
    when accepted}. "stage" == "ready" means the course is complete."""
    m = MESSAGES[language]
    text = text.strip()

    def say(key, **kw):
        return m[key].format(**kw)

    if stage == "name":
        if not text:
            return {"reply": say("empty"), "stage": "name"}
        if not check_name(text):
            return {"reply": say("name_long"), "stage": "name"}
        return {"reply": say("ask_outcomes"), "stage": "outcomes", "name": text}

    if stage in ("outcomes", "contents"):
        if stage == "contents" and text.lower().strip(" .!\"'") in NONE_ANSWERS:
            return {"reply": say("working"), "stage": "ready", "contents": ""}
        problem = check_section(text)
        if problem == "whole_page":
            return {"reply": say(problem, headings=", ".join(
                f"\"{h.title()}\"" for h in other_headings(text)[:3])), "stage": stage}
        if problem:
            return {"reply": say(problem), "stage": stage}
        if stage == "outcomes":
            return {"reply": say("ask_contents"), "stage": "contents", "outcomes": text}
        return {"reply": say("working"), "stage": "ready", "contents": text}

    raise ValueError(f"no intake step for stage {stage!r}")
