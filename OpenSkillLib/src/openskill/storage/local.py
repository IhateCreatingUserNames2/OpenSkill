"""
LocalDiskStore — Armazenamento 100% Local e Gratuito
======================================================
Salva skills como arquivos .md/.json no disco do desenvolvedor.
Não requer API key, não manda dados para nenhum servidor.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, TYPE_CHECKING

from openskill.storage.base import (
    BaseSkillStore,
    SkillMetadata,
    SkillGraphData, SkillVectorProfile, SkillType,
)

if TYPE_CHECKING:
    pass


class LocalDiskStore(BaseSkillStore):
    """
    Adapter de armazenamento que salva tudo no disco local.

    Estrutura de diretórios:
        <path>/
        ├── skills/
        │   ├── raft_consensus_a493a4b2.md
        │   ├── paxos_01a3f8cd.md
        │   └── ...
        │   ├── raft_consensus_a493a4b2.json
        │   ├── paxos_01a3f8cd.json
        │   └── ...
        └── skill_graph.json
    """

    def __init__(
        self,
        path: str | Path = "./skills_output",
        create: bool = True,
    ):
        self.root = Path(path).resolve()
        self.skills_dir = self.root / "skills"

        if create:
            self.skills_dir.mkdir(parents=True, exist_ok=True)
            self._init_graph()

    def _init_graph(self) -> None:
        graph_path = self.root / "skill_graph.json"
        if not graph_path.exists():
            graph_path.write_text(
                json.dumps({"nodes": {}, "edges": []}),
                encoding="utf-8",
            )

    # ── Graph ────────────────────────────────────────────────────────────────

    def get_graph(self) -> SkillGraphData:
        graph_path = self.root / "skill_graph.json"
        try:
            data = json.loads(graph_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            data = {"nodes": {}, "edges": []}
        return SkillGraphData.from_dict(data)

    async def update_graph(self, graph: SkillGraphData) -> None:
        graph_path = self.root / "skill_graph.json"
        graph_path.write_text(
            json.dumps(graph.to_dict(), indent=2),
            encoding="utf-8",
        )

    # ── Skills CRUD ──────────────────────────────────────────────────────────

    async def save_skill(
            self,
            skill_id: str,
            markdown: str,
            metadata: SkillMetadata,
    ) -> None:
        """
        Implementação obrigatória da interface BaseSkillStore.
        Redireciona para o novo sistema de pacotes (Bundles) do EvoSkills
        passando código executável vazio.
        """
        await self.save_skill_bundle(skill_id, markdown, metadata, executable_code="")

    async def save_skill_bundle(
        self,
        skill_id: str,
        markdown: str,
        metadata: SkillMetadata,
        executable_code: str = ""
    ) -> None:
        """
        Nova Estrutura (EvoSkills Bundle):
        skills_output/
        └── <skill_id>/
            ├── SKILL.md            # Conhecimento Declarativo (MemCollab)
            ├── meta.json           # Vetores S-Path e Level (Ficha RPG)
            └── scripts/
                └── utils.py        # Código Ativo (EvoSkills)
        """
        bundle_dir = self.skills_dir / skill_id
        bundle_dir.mkdir(parents=True, exist_ok=True)

        # Atualiza a rota do arquivo MD nos metadados
        metadata.filename = f"{skill_id}/SKILL.md"

        # 1. Salva a "Aura" (Regras Passivas em Markdown)
        md_path = bundle_dir / "SKILL.md"
        md_path.write_text(markdown, encoding="utf-8")

        # 2. Salva os "Status" (Meta.json)
        meta_path = bundle_dir / "meta.json"
        meta_path.write_text(
            json.dumps(metadata.to_dict(), indent=2),
            encoding="utf-8",
        )

        # 3. Salva a "Magia Ativa" (Python Script)
        if executable_code or metadata.skill_type in [SkillType.ACTIVE, SkillType.HYBRID]:
            scripts_dir = bundle_dir / "scripts"
            scripts_dir.mkdir(exist_ok=True)
            code_path = scripts_dir / "utils.py"
            code_path.write_text(executable_code, encoding="utf-8")

    async def get_skill_meta(self, skill_id: str) -> Optional[SkillMetadata]:
        sid = skill_id.strip()
        # 1. Novo formato: skills/{skill_id}/meta.json (save_skill_bundle)
        bundle_meta = self.skills_dir / sid / "meta.json"
        if bundle_meta.exists():
            try:
                return SkillMetadata.from_dict(json.loads(bundle_meta.read_text(encoding="utf-8")))
            except Exception:
                pass
        # 2. Legado: {skill_id}.json na raiz de skills/
        for f in self.skills_dir.glob(f"{sid}.json"):
            try:
                return SkillMetadata.from_dict(json.loads(f.read_text(encoding="utf-8")))
            except Exception:
                pass
        # 3. Fallback amplo (qualquer lugar)
        for f in self.root.rglob("meta.json"):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                if data.get("id") == sid:
                    return SkillMetadata.from_dict(data)
            except Exception:
                pass
        return None

    async def get_skill_md(self, skill_id: str) -> Optional[str]:
        meta = await self.get_skill_meta(skill_id)
        if meta is None: return None

        # Procura o Markdown onde quer que ele esteja
        paths = [self.skills_dir / meta.filename, self.root / meta.filename]
        for p in paths:
            if p.exists():
                return p.read_text(encoding="utf-8")
        return None

    async def list_skills(self) -> list[SkillMetadata]:
        metas: list[SkillMetadata] = []
        # rglob procura em TUDO (arquivos meta.json dentro das pastas e .json na raiz)
        files = list(self.root.rglob("meta.json")) + list(self.root.glob("*.json"))

        seen_ids = set()
        for f in files:
            if f.name == "skill_graph.json": continue
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                if "id" not in data: continue

                meta = SkillMetadata.from_dict(data)
                if meta.id not in seen_ids:
                    metas.append(meta)
                    seen_ids.add(meta.id)
            except Exception:
                continue
        return sorted(metas, key=lambda m: m.created_at, reverse=True)

    async def delete_skill(self, skill_id: str) -> None:
        meta = await self.get_skill_meta(skill_id)
        if meta:
            (self.skills_dir / meta.filename).un_exists(missing_ok=True)
        (self.skills_dir / f"{skill_id}.json").unlink(missing_ok=True)

    @property
    def workspace_path(self) -> Optional[Path]:
        return self.root

    async def save_embedding(
            self,
            skill_id: str,
            embedding: list[float],
            qvector: dict,
            model_name: str,
            dimension: int,
            provider: str
    ) -> None:
        meta = await self.get_skill_meta(skill_id)
        if meta is None: return

        # Cria ou atualiza o perfil específico
        profile_key = f"{model_name}_{dimension}".replace("/", "_")
        meta.vectors[profile_key] = SkillVectorProfile(
            model=model_name,
            dimension=dimension,
            provider=provider,
            qvector=qvector,
            embedding=embedding
        )

        # Mantém compatibilidade com campos antigos
        meta.qvector = qvector
        meta.embedding = embedding

        # CORRIGIDO: salva no bundle (meta.json) se existir, senão no legado (.json)
        # Antes sempre salvava em {skill_id}.json, ignorando o bundle — o bootstrap
        # não encontrava os vetores porque lia de skills/{id}/meta.json.
        bundle_meta = self.skills_dir / skill_id / "meta.json"
        if bundle_meta.exists():
            meta_path = bundle_meta
        else:
            # Legado: skill criada antes do sistema de bundles
            meta_path = self.skills_dir / f"{skill_id}.json"

        meta_path.write_text(json.dumps(meta.to_dict(), indent=2), encoding="utf-8")