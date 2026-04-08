"""
OpenSkill — Geometric Skill Memory for LLM Agents
=================================================

Um sistema de ciclo de vida completo de skills para agentes LLM:

  MemCollab  (arXiv:2603.23234)
      Dual-agent contrastive trajectory distillation.
      Cria skills agent-agnostic a partir de pares fraco/forte.

  Trace2Skill (arXiv:2603.25158)
      Fleet-based parallel skill evolution.
      Evolução via frota de sub-agentes + consolidação hierárquica.

  TurboQuant (arXiv:2504.19874)
      Near-optimal vector quantization (4 bits/channel).
      Memória geométrica compressa com correção QJL.

  S-Path-RAG (arXiv:2603.23512)
      Semantic-aware shortest-path retrieval over skill graph.
      Neural-Socratic loop + soft latent injection.

Exemplo rápido:
    from openskill import OpenSkillClient, LocalDiskStore

    client = OpenSkillClient(store=LocalDiskStore("./skills"))
    skill = await client.craft(
        task="Implement Raft consensus over high-latency network",
        weak_model="openai/gpt-4o-mini",
        strong_model="anthropic/claude-3-5-sonnet",
    )
    guidance = await client.retrieve("How to handle network partitions in Raft?")
"""

from openskill.core.crafter import SkillCrafter
from openskill.core.evolver import SkillEvolver
from openskill.core.vector import TurboQuantizer
from openskill.core.graph import SkillGraph

from openskill.storage.base import BaseSkillStore
from openskill.storage.local import LocalDiskStore
from openskill.storage.cloud import CloudSaaSStore

from openskill.llm.base import BaseLLMProvider
from openskill.llm.openrouter import OpenRouterProvider
from openskill.llm.ollama import OllamaProvider

from openskill.retrieval.retriever import OpenSkillRetriever
from openskill.injection.soft import SkillProjector

# ── High-level client ─────────────────────────────────────────────────────────

from openskill.client import OpenSkillClient

__all__ = [
    # Core
    "SkillCrafter",
    "SkillEvolver",
    "TurboQuantizer",
    "SkillGraph",
    # Storage
    "BaseSkillStore",
    "LocalDiskStore",
    "CloudSaaSStore",
    # LLM providers
    "BaseLLMProvider",
    "OpenRouterProvider",
    "OllamaProvider",
    # Retrieval
    "OpenSkillRetriever",
    # Injection
    "SkillProjector",
    # High-level
    "OpenSkillClient",
    # Version
    "__version__",
]

__version__ = "0.1.0"
