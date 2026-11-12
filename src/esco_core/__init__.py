"""Minimal inference path of esco-skill-linker, ported for the agent.

Source: esco-skill-linker at commit 3645d54f296c9dfc51f7a883071c8d98d22326ba
(src/linker.py, src/retrieval/{bm25,dense,fusion}.py). Only the deployed
first stage is kept: BM25 + EmbeddingGemma, weighted score fusion, English.
No reranking. tests/test_parity.py checks this reproduces the linker's output.
"""

from esco_core.linker import SkillHit, SkillLinker

LINKER_COMMIT = "3645d54f296c9dfc51f7a883071c8d98d22326ba"

__all__ = ["SkillHit", "SkillLinker", "LINKER_COMMIT"]
