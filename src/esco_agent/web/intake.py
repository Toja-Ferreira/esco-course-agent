"""The chat's opening: collecting the course text, with fixed messages (no LLM).

Same principle as the assistant's fixed first turn: this part must not go wrong
in unpredictable ways, and it costs no API calls. Stages:
    name -> outcomes -> contents -> ready
Each answer is checked before moving on. If the first paste is the whole course
page (recognised by its section headings), it is split by those headings: the
objectives, learning outcomes and contents are kept, the rest (assessment,
teaching methods, requisites, bibliography) is left out, and the intake is done.
"""

from __future__ import annotations

import re

MAX_SECTION_CHARS = 6000
MAX_PAGE_CHARS = 20000
MIN_SECTION_WORDS = 8

MESSAGES = {
    "pt": {
        "hello": ("Olá! Vou ajudar a escolher as competências ESCO que descrevem a sua unidade "
                  "curricular. Primeiro, como se chama a unidade curricular?"),
        "hello_named": ("Olá! Vamos escolher as competências ESCO de **{name}**. Para começar, cole "
                        "o texto da página da unidade curricular **em inglês**: a página inteira, "
                        "ou só os **resultados de aprendizagem**."),
        "ask_outcomes": ("Obrigado. Agora cole o texto da página da unidade curricular **em "
                         "inglês**: a página inteira, ou só os **resultados de aprendizagem**."),
        "page_used": ("Usei {used} da página{dropped}. A ler a unidade curricular e a procurar "
                      "competências…"),
        "page_dropped": " (deixei de fora {dropped}, que não descrevem competências)",
        "page_none": ("Vejo títulos da página, mas não encontrei objetivos, resultados de "
                      "aprendizagem nem conteúdos. Pode colar os **resultados de aprendizagem**?"),
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
                        "course page's text **in English**: the whole page, or just the "
                        "**learning outcomes**."),
        "ask_outcomes": ("Thank you. Now paste the course page's text **in English**: the whole "
                         "page, or just the **learning outcomes**."),
        "page_used": "I used {used} from the page{dropped}. Reading the course and looking for skills…",
        "page_dropped": " (I left out {dropped}, which don't describe skills)",
        "page_none": ("I can see the page's headings, but no objectives, learning outcomes or "
                      "contents. Could you paste the **learning outcomes**?"),
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


# Section headings of a course page (POF, PT and EN), matched as whole lines.
KEEP = {"objectives": "Objectives", "objetivos": "Objectives",
        "learning outcomes": "Learning outcomes", "intended learning outcomes": "Learning outcomes",
        "resultados de aprendizagem": "Learning outcomes",
        "contents": "Contents", "syllabus": "Contents", "conteúdos": "Contents",
        "conteúdos programáticos": "Contents", "programa": "Contents"}
DROP = {"requisites": "requisites", "prerequisites": "requisites", "requisitos": "requisites",
        "assessment": "assessment", "evaluation": "assessment", "avaliação": "assessment",
        "teaching methods": "teaching methods", "teaching methodology": "teaching methods",
        "métodos de ensino": "teaching methods", "metodologias de ensino": "teaching methods",
        "bibliography": "bibliography", "recommended bibliography": "bibliography",
        "bibliografia": "bibliography", "bibliografia recomendada": "bibliography"}
DROP_PT = {"requisites": "requisitos", "assessment": "avaliação",
           "teaching methods": "métodos de ensino", "bibliography": "bibliografia"}
KEEP_PT = {"Objectives": "objetivos", "Learning outcomes": "resultados de aprendizagem",
           "Contents": "conteúdos"}


def _heading(line: str) -> str | None:
    key = " ".join(line.strip().rstrip(":").strip().lower().split())
    if key in KEEP or key in DROP:
        return key
    # a qualified heading of a section that is left out ("Abbreviated bibliography",
    # "Main bibliography"), so it doesn't end up inside the section above it
    words = key.split()
    if 1 < len(words) <= 4:
        for term in DROP:
            if len(term.split()) == 1 and words[-1] == term:
                return term
    return None


def split_page(text: str) -> tuple[dict, list[str]] | None:
    """A whole course page split by its headings: ({section: text} kept, [dropped
    kinds]). None if the text doesn't look like a page (fewer than two headings)."""
    lines = text.splitlines()
    if sum(_heading(ln) is not None for ln in lines) < 2:
        return None
    kept, dropped, current, buf = {}, [], None, []

    def flush():
        if current is None:
            return                            # text before the first heading (the title)
        body = "\n".join(buf).strip()
        if current in KEEP and body:
            name = KEEP[current]
            kept[name] = (kept[name] + "\n" + body) if name in kept else body
        elif current in DROP and DROP[current] not in dropped:
            dropped.append(DROP[current])

    for ln in lines:
        h = _heading(ln)
        if h is not None:
            flush(); current, buf = h, []
        else:
            buf.append(ln)
    flush()
    order = ["Objectives", "Learning outcomes", "Contents"]
    return {k: kept[k] for k in order if k in kept}, dropped


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

    if stage == "outcomes":
        page = split_page(text) if len(text) <= MAX_PAGE_CHARS else None
        if page is not None:
            kept, dropped = page
            if not kept:
                return {"reply": say("page_none"), "stage": "outcomes"}
            for body in kept.values():
                problem = check_section(body)
                if problem in ("portuguese", "too_long"):
                    return {"reply": say(problem), "stage": "outcomes"}
            if len(" ".join(kept.values()).split()) < MIN_SECTION_WORDS:
                return {"reply": say("too_short"), "stage": "outcomes"}
            pt = language == "pt"
            names = [KEEP_PT[k] if pt else k.lower() for k in kept]
            gone = [DROP_PT[d] if pt else d for d in dropped]
            join = (lambda xs: ", ".join(xs[:-1]) + (" e " if pt else " and ") + xs[-1]
                    if len(xs) > 1 else xs[0])
            return {"reply": say("page_used", used=join(names),
                                 dropped=say("page_dropped", dropped=join(gone)) if gone else ""),
                    "stage": "ready", "sections": kept}

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
