"""
Storage Abstraction Layer — Adapter Pattern
==========================================
TODOS os adaptadores de armazenamento implementam esta interface.

Isto é o que torna o OpenSkill um produto Open Core:
  - LocalDiskStore  → Gratuito, 100% local, roda no laptop do dev
  - CloudSaaSStore  → Pago,     infraestrutura gerenciada na nuvem
"""
# --- openskill/storage/base.py ---

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field, fields
import json
from enum import Enum

class SkillType(str, Enum):
    ACTIVE = "active"     # Requer invocação de código (EvoSkills Python Scripts)
    PASSIVE = "passive"   # Requer apenas injeção no espaço latente (S-Path RAG Cross-Attention)
    HYBRID = "hybrid"     # Possui regras semânticas fortes E scripts utilitários

@dataclass
class SkillVectorProfile:
    model: str = "unknown"
    dimension: int = 0
    provider: str = "unknown"
    qvector: dict = field(default_factory=dict)
    embedding: Optional[list[float]] = None

    def to_dict(self) -> dict:
        return {
            "model": self.model, "dimension": self.dimension,
            "provider": self.provider, "qvector": self.qvector,
            "embedding": self.embedding
        }


@dataclass
class SkillMetadata:
    # Campos obrigatórios com fallback default para não quebrar no 'from_dict'
    id: str = ""
    title: str = "Untitled"
    skill_type: SkillType = SkillType.PASSIVE
    domain: str = "General"
    category: str = "General"
    subcategory: str = "General"

    # --- (Game Mechanics) ---
    level: int = 1  # Sobe a cada co-evolução (EvoSkills)
    xp: float = 0.0  # Acumula baseado na taxa de sucesso em produção
    last_success_rate: Optional[float] = None
    evolution_count: int = 0  # Quantas vezes o Trace2Skill fez "respec" ou patch
    mana_cost: int = 0  # Estimativa de tokens (Custo para a Hotbar)

    task: str = ""
    filename: str = ""
    created_at: str = ""
    weak_model: str = ""
    strong_model: str = ""

    trajectory_count: int = 0
    last_evolved_at: Optional[str] = None

    vectors: dict[str, SkillVectorProfile] = field(default_factory=dict)
    embedding: Optional[list[float]] = None
    qvector: Optional[dict] = None

    def to_dict(self) -> dict:
        d = dataclass_asdict_filter_none(self)
        d['skill_type'] = self.skill_type.value  # Serializa o Enum
        if self.vectors:
            d["vectors"] = {k: v.to_dict() if hasattr(v, 'to_dict') else v for k, v in self.vectors.items()}
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "SkillMetadata":
        import dataclasses
        valid_fields = {f.name for f in dataclasses.fields(cls)}

        # Converte o dicionário de vetores com segurança
        vectors_raw = d.get("vectors", {})
        vectors = {}
        if isinstance(vectors_raw, dict):
            for k, v in vectors_raw.items():
                if isinstance(v, dict):
                    # Filtra campos do perfil também
                    p_fields = {f.name for f in dataclasses.fields(SkillVectorProfile)}
                    p_data = {pk: pv for pk, pv in v.items() if pk in p_fields}
                    vectors[k] = SkillVectorProfile(**p_data)

        if 'skill_type' in d:
            d['skill_type'] = SkillType(d['skill_type'])

        # Filtra campos da skill
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        if "vectors" in filtered: del filtered["vectors"]

        return cls(**filtered, vectors=vectors)


@dataclass
class SkillGraphData:
    """Grafo de skills (nós + arestas)."""
    nodes: dict[str, dict] = field(default_factory=dict)
    edges: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"nodes": self.nodes, "edges": self.edges}

    @classmethod
    def from_dict(cls, d: dict) -> "SkillGraphData":
        return cls(nodes=d.get("nodes", {}), edges=d.get("edges", []))


def dataclass_asdict_filter_none(obj) -> dict:
    import dataclasses
    d = {}
    for f in dataclasses.fields(obj):
        val = getattr(obj, f.name)
        # Se for Ellipsis ou None, ignoramos.
        if val is not None and val is not Ellipsis:
            d[f.name] = val
    return d


class BaseSkillStore(ABC):
    """
    Interface abstrata para armazenamento de skills.

    Implemente esta interface para criar um novo adapter:
      1. LocalDiskStore    — salva em arquivos .md/.json no disco
      2. CloudSaaSStore    — chama API REST do SaaS
      3. RedisStore         — vetores no Redis (exemplo futuro)
      4. PgVectorStore      — vetores no PostgreSQL + pgvector
    """

    @abstractmethod
    async def save_skill(
        self,
        skill_id: str,
        markdown: str,
        metadata: SkillMetadata,
    ) -> None:
        """Salva uma skill (markdown + metadados)."""
        ...

    @abstractmethod
    async def get_skill_md(self, skill_id: str) -> Optional[str]:
        """Retorna o conteúdo Markdown de uma skill."""
        ...

    @abstractmethod
    async def get_skill_meta(self, skill_id: str) -> Optional[SkillMetadata]:
        """Retorna os metadados de uma skill."""
        ...

    @property
    def workspace_path(self) -> Optional[Path]:
        """
        Retorna o caminho local do workspace, se aplicável.
        Retorna None para armazenamentos em nuvem pura que não suportam cache local de pesos neurais.
        """
        return None

    @abstractmethod
    async def list_skills(self) -> list[SkillMetadata]:
        """Lista todas as skills no store."""
        ...

    @abstractmethod
    async def delete_skill(self, skill_id: str) -> None:
        """Remove uma skill do store."""
        ...

    @abstractmethod
    def get_graph(self) -> SkillGraphData:
        """Retorna o grafo de skills."""
        ...

    @abstractmethod
    async def update_graph(self, graph: SkillGraphData) -> None:
        """Atualiza o grafo de skills."""
        ...

    @abstractmethod
    async def save_embedding(
            self,
            skill_id: str,
            embedding: list[float],
            qvector: dict,
            model_name: str,
            dimension: int,
            provider: str
    ) -> None:
        """Salva o vetor quantizado (TurboQuant) de uma skill."""
        ...

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_uri(cls, uri: str, **kwargs) -> "BaseSkillStore":
        """
        Factory que retorna o adapter correto baseado na URI.

        Exemplos:
          LocalDiskStore.from_uri("local:./skills")
          CloudSaaSStore.from_uri("cloud://osk_live_xxx?workspace=acme")
          RedisStore.from_uri("redis://localhost:6379/0")
        """
        scheme = uri.split("://")[0] if "://" in uri else "local"

        if scheme == "local":
            path = uri.replace("local://", "").strip() or "./skills"
            return cls(path=path, **kwargs)  # type: ignore[call-arg]
        elif scheme == "cloud":
            api_key = kwargs.get("api_key", "")
            workspace = kwargs.get("workspace", "default")
            return cls(api_key=api_key, workspace=workspace, **kwargs)  # type: ignore[call-arg]
        else:
            raise ValueError(f"Unknown storage scheme: {scheme}")
