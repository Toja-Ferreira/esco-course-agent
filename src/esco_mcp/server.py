"""MCP server exposing ESCO skill linking and occupation lookup as three tools.

    link_text              course text -> ESCO skills (hybrid BM25 + EmbeddingGemma)
    search_occupations     ESCO skills -> best-matching ESCO occupations (TF-IDF cosine)
    get_occupation_skills  one occupation -> its essential and optional skills

Run over stdio (what the agent and Claude Desktop use):
    python -m esco_mcp.server
Data comes from ESCO_DATA_DIR (default ./data). The models load in a
background thread at startup, so the MCP handshake isn't held up; the first
tool call waits for loading to finish.

IDs: tools return the UUID part of ESCO URIs ("6c08403c-...") to keep
responses short; they accept either that or the full URI.
"""

from __future__ import annotations

import threading

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from esco_core.linker import SkillLinker, data_dir
from esco_core.occupations import OccupationIndex

SKILL_PREFIX = "http://data.europa.eu/esco/skill/"
OCC_PREFIX = "http://data.europa.eu/esco/occupation/"
MAX_LINK_K = 50
MAX_SEARCH_K = 25

mcp = FastMCP(
    "esco",
    instructions=(
        "Tools over the EU's ESCO taxonomy (v1.2.1, English). Use link_text to "
        "find the ESCO skills in a text, search_occupations to rank occupations "
        "by those skills, and get_occupation_skills to see what one occupation "
        "requires. Only state skills and occupations that a tool returned."
    ),
    log_level="WARNING",  # stdio: stderr only, but INFO per request is noise
)

# --- loading ------------------------------------------------------------------

_ready = threading.Event()
_state: dict = {}


def _load() -> None:
    try:
        _state["linker"] = SkillLinker(data_dir())
        _state["occupations"] = OccupationIndex(data_dir())
    except Exception as exc:  # surfaced on the first tool call
        _state["error"] = exc
    finally:
        _ready.set()


def _get(name: str):
    _ready.wait()
    if "error" in _state:
        raise ToolError(f"ESCO data failed to load: {_state['error']}")
    return _state[name]


# --- ids ----------------------------------------------------------------------

def _short(uri: str) -> str:
    return uri.rsplit("/", 1)[-1]


def _full(id_or_uri: str, prefix: str) -> str:
    s = id_or_uri.strip()
    return s if s.startswith("http") else prefix + s


# --- tools --------------------------------------------------------------------

@mcp.tool()
def link_text(text: str, k: int = 20, include_descriptions: bool = False) -> dict:
    """Find the ESCO skills a text describes, best match first.

    Use on a course description (or any passage of it). Returns up to k skills
    with id, label, type ("skill/competence" or "knowledge") and a relevance
    score in [0, 1]. Scores rank skills within one call; they are not
    probabilities and are not comparable across calls.
    include_descriptions adds ESCO's definition of each skill; use it when a
    label alone is ambiguous ("types of strings" is about musical instruments).
    """
    if not text.strip():
        raise ToolError("text is empty")
    k = max(1, min(int(k), MAX_LINK_K))
    hits = _get("linker").link(text, k=k)
    skills = []
    for h in hits:
        skill = {"id": _short(h.uri), "label": h.label,
                 "type": h.skill_type, "reuse": h.reuse_level, "score": round(h.score, 3)}
        if include_descriptions:
            skill["description"] = h.description
        skills.append(skill)
    return {"skills": skills}


@mcp.tool()
def search_occupations(skill_ids: list[str], weights: list[float] | None = None,
                       isco_prefix: str | None = None, k: int = 10) -> dict:
    """Rank ESCO occupations by how well they match a set of ESCO skills.

    skill_ids: ids from link_text or get_occupation_skills.
    weights:   optional, one per skill id (e.g. link_text's scores); default 1.0.
               Pass link_text's scores to reproduce an earlier ranking.
    isco_prefix: optional ISCO-08 code prefix to restrict the sector, e.g.
               "25" ICT professionals, "35" ICT technicians, "2" all
               professionals, "3" technicians and associate professionals,
               "1" managers, "7" craft and related trades workers.
    k:         how many occupations to return.

    Each occupation comes with a score in [0, 1] and the skills that contributed
    most to it, with whether the occupation lists that skill as essential or
    optional. Use those to explain why an occupation matched.
    """
    if not skill_ids:
        raise ToolError("skill_ids is empty")
    if weights is not None and len(weights) != len(skill_ids):
        raise ToolError(f"got {len(weights)} weights for {len(skill_ids)} skill_ids")
    k = max(1, min(int(k), MAX_SEARCH_K))
    occ = _get("occupations")

    weights = weights if weights is not None else [1.0] * len(skill_ids)
    scores: dict[str, float] = {}
    unknown = []
    for sid, w in zip(skill_ids, weights):
        uri = _full(sid, SKILL_PREFIX)
        if uri in occ.skill_idx:
            scores[uri] = max(scores.get(uri, 0.0), float(w))
        elif uri in occ.skill_label:
            pass  # a real ESCO skill that no occupation lists: no signal
        else:
            unknown.append(sid)
    if unknown and not scores:
        raise ToolError(f"none of these are ESCO skill ids: {unknown}")

    hits = occ.search_occupations(scores, k=k, isco_prefix=isco_prefix,
                                  max_matched=3)
    out = {"occupations": [
        {"id": _short(h.uri), "label": h.label, "isco_group": h.isco_group,
         "score": round(h.score, 3),
         "top_matching_skills": [{"id": _short(c.skill_uri), "label": c.skill_label,
                                  "relation": c.relation} for c in h.matched]}
        for h in hits]}
    if unknown:
        out["unknown_skill_ids"] = unknown
    if not hits:
        out["note"] = ("no occupation matched"
                       + (f" within ISCO prefix {isco_prefix!r}" if isco_prefix else ""))
    return out


@mcp.tool()
def get_occupation_skills(occupation_id: str) -> dict:
    """List what one ESCO occupation requires: its essential and optional skills.

    occupation_id: an id from search_occupations. To find which essential
    skills a course is missing, compare the essential list here with the
    skills link_text found in the course.
    """
    try:
        o = _get("occupations").get_occupation_skills(_full(occupation_id, OCC_PREFIX))
    except KeyError:
        raise ToolError(f"not an ESCO occupation id: {occupation_id!r}") from None

    def refs(skills):
        return [{"id": _short(s.uri), "label": s.label, "type": s.skill_type,
                 "reuse": s.reuse_level} for s in skills]

    return {"id": _short(o.uri), "label": o.label, "isco_group": o.isco_group,
            "description": o.description,
            "essential": refs(o.essential), "optional": refs(o.optional)}


def main() -> None:
    threading.Thread(target=_load, daemon=True).start()
    mcp.run()  # stdio


if __name__ == "__main__":
    main()
