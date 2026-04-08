"""
OpenSkill Core — Pure Business Logic, Zero I/O
==============================================

Cada módulo contém apenas lógica de domínio, SEM dependência de:
  - Sistema de arquivos
  - APIs HTTP externas
  - Armazenamento

A injeção de dependência (storage, LLM provider) é feita pelo caller
(OpenSkillClient ou testes).

Módulos:
  crafter.py   — MemCollab: dual-agent contrastive distillation
  evolver.py   — Trace2Skill: fleet-based skill evolution
  vector.py    — TurboQuant: geometric vector quantization
  graph.py     — S-Path-RAG: semantic graph retrieval
"""

from openskill.core.crafter import SkillCrafter
from openskill.core.evolver import SkillEvolver
from openskill.core.vector import TurboQuantizer
from openskill.core.graph import SkillGraph

__all__ = [
    "SkillCrafter",
    "SkillEvolver",
    "TurboQuantizer",
    "SkillGraph",
]
