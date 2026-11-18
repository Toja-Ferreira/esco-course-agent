"""Live grounding check on the assistant's replies.

The skill list is grounded by construction (numbers, exact names, evidence by
sentence number). The model's prose is not: it could quote a sentence that
isn't in the course, or refer to a skill number that doesn't exist. This checks
each reply against what the session actually contains:

  - every quotation of four or more words must appear (ignoring case, spacing
    and punctuation at the ends) in the course text, a tool result, the
    session's skills and text changes, the teacher's own messages, or the
    assistant's earlier replies (with their own quotations removed, so an
    unsupported quote cannot become supported by being repeated); a quote
    shortened with "…" is checked piece by piece;
  - a quoted question ("Ask 'what is missing for 1?'") is a suggestion to the
    teacher, not a citation, and is not checked;
  - every "number N" / "número N" must be a skill that exists.

A reply that fails is still shown, marked, and the issue is logged: the
share of flagged replies is a reported number.
"""

from __future__ import annotations

import re

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

QUOTE = re.compile(r'"([^"\n]{12,})"|“([^”\n]{12,})”|«([^»\n]{12,})»')
NUMBER = re.compile(r"\b(?:n[úu]mero|number|n\.?º)\s*(\d{1,3})\b", re.I)
MIN_WORDS = 4


def _norm(text: str) -> str:
    text = text.lower().replace("’", "'").replace("‘", "'")
    return " ".join(re.sub(r"[^\w'\s-]", " ", text).split())


def haystack(state: dict, earlier: int | None = None) -> str:
    """Everything a reply may legitimately quote. `earlier`: how many messages came
    before this turn; the assistant's replies among them may be quoted back."""
    parts = list((state.get("sections") or {}).values())
    for i, m in enumerate(state.get("messages") or []):
        if not isinstance(m.content, str):
            continue
        if isinstance(m, (ToolMessage, HumanMessage)):
            parts.append(m.content)
        elif isinstance(m, AIMessage) and earlier is not None and i < earlier:
            parts.append(QUOTE.sub(" ", m.content))
    for s in list(state.get("known") or []) + list((state.get("offered") or {}).values()):
        parts += [s.get("label", ""), s.get("description", "")]
        parts += [e.get("text", "") for e in s.get("evidence", [])]
    for c in state.get("text_changes") or []:
        parts += [c.get("text", ""), c.get("reason", "")] + list(c.get("add_to_course") or [])
    return _norm("\n".join(parts))


def check_reply(reply: str, state: dict, earlier: int | None = None) -> list[dict]:
    """Problems found in one reply, as [{"kind", "text"}]; empty if none."""
    hay = haystack(state, earlier)
    issues = []
    for m in QUOTE.finditer(reply):
        quote = next(g for g in m.groups() if g)
        if quote.rstrip(" .'\"’”").endswith("?"):
            continue                         # a question the teacher could ask
        pieces = [p for p in re.split(r"…|\.\.\.", quote) if len(p.split()) >= MIN_WORDS]
        missing = [p for p in pieces if _norm(p) not in hay]
        if missing:
            issues.append({"kind": "quote_not_found", "text": quote[:200]})
    n_known = len(state.get("known") or [])
    for m in NUMBER.finditer(reply):
        n = int(m.group(1))
        if not 1 <= n <= n_known:
            issues.append({"kind": "unknown_skill_number", "text": m.group(0)})
    return issues
