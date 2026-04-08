"""
SkillEvolver — Trace2Skill: Fleet-Based Parallel Evolution
==========================================================
Implementação completa do pipeline Trace2Skill (arXiv:2603.25158).

Diferente do RAG tradicional, o Evolver não apenas recupera, ele REESCREVE
as diretrizes de raciocínio baseado em evidência empírica de falhas e sucessos.

Mecânica:
  1. Batching: Divide N trajetórias entre uma frota de sub-agentes.
  2. Patch Proposal: Cada sub-agent propõe um 'Skill Patch' (JSON Diff).
  3. Hierarchical Merge: Um operador de merge consolida os patches, mantendo
     apenas o que for prevalente (sinal > ruído).
  4. Application: Aplica as mudanças estruturadas no Markdown original.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Any

import structlog
from openskill.llm.base import BaseLLMProvider, LLMMessage

log = structlog.get_logger()

# ── Configurações do Trace2Skill ─────────────────────────────────────────────

FLEET_BATCH_SIZE = 4  # Trajetórias por sub-agente
MAX_FLEET_SIZE = 10  # Limite de paralelismo para evitar rate limit
MIN_PREVALENCE = 0.3  # Só aceita patches vistos em >30% das trajetórias do batch


# ── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class SkillPatch:
    """Representa uma mudança sugerida em uma seção da Skill."""
    section: str  # 'Invariants', 'Violations', 'Constraints', etc.
    op: str  # 'append', 'replace', 'insert', 'remove'
    content: str  # O novo texto
    target: str = ""  # Texto âncora para replace/insert
    justification: str = ""  # Por que essa mudança é necessária?
    prevalence: float = 0.5  # Quão comum foi o padrão observado (0.0 a 1.0)


@dataclass
class EvolutionResult:
    """Resultado final do processo de evolução."""
    evolved_md: str
    patches_applied: list[SkillPatch]
    success_rate: float
    fleet_size: int
    patch_count: int


# ── Prompts ──────────────────────────────────────────────────────────────────

EVOLVER_SYSTEM_PROMPT = (
    "You are a Skill Evolution Sub-Agent (Trace2Skill framework).\n"
    "Analyze execution trajectories and propose targeted patches to improve a skill document.\n\n"
    "RULES:\n"
    "1. Only propose patches backed by OBSERVABLE patterns in the trajectories.\n"
    "2. Focus on patterns that repeat across MULTIPLE trajectories.\n"
    "3. Use the format 'avoid X; enforce Y' for constraints.\n"
    "4. Prevalence: 1.0 if seen in all trajectories, 0.5 if in half, etc.\n"
    "Output ONLY a JSON array of patches."
)

MERGE_SYSTEM_PROMPT = (
    "You are a Skill Merge Coordinator.\n"
    "Consolidate multiple patch sets into one coherent, non-redundant set.\n\n"
    "STRATEGY:\n"
    "1. Deduplicate: Merge similar suggestions into the best-worded one.\n"
    "2. Conflict Resolution: If patches contradict, keep the one with higher prevalence.\n"
    "3. Pattern Elevation: If multiple sub-agents found the same issue, increase its prevalence.\n"
    "Output ONLY a JSON array of merged patches."
)


# ── Classe Principal ─────────────────────────────────────────────────────────

class SkillEvolver:
    def __init__(self, llm: BaseLLMProvider):
        self.llm = llm

    async def evolve(
            self,
            skill_md: str,
            trajectories: list[dict]
    ) -> EvolutionResult:
        """
        Executa o pipeline completo de evolução em frota.

        trajectories: list de {"task": str, "trajectory": str, "success": bool}
        """
        if not trajectories:
            log.warning("evolver.no_trajectories")
            return EvolutionResult(skill_md, [], 0.0, 0, 0)

        success_rate = sum(1 for t in trajectories if t.get("success")) / len(trajectories)

        # 1. Stage 2: Proposta de Patches em Paralelo
        batches = self._create_batches(trajectories)
        log.info("evolver.dispatch_fleet", num_batches=len(batches))

        patch_groups = await asyncio.gather(*[
            self._analyze_batch(skill_md, batch) for batch in batches
        ])
        patch_groups = [g for g in patch_groups if g]  # Remove falhas

        # 2. Stage 3: Consolidação Hierárquica (Merge)
        consolidated_patches = await self._hierarchical_merge(skill_md, patch_groups)

        # 3. Stage 4: Aplicação dos Patches
        evolved_md = self._apply_patches(skill_md, consolidated_patches)

        return EvolutionResult(
            evolved_md=evolved_md,
            patches_applied=consolidated_patches,
            success_rate=success_rate,
            fleet_size=len(batches),
            patch_count=len(consolidated_patches)
        )

    async def generate_trajectories(
            self,
            model: str,
            skill_md: str,
            tasks: list[str]
    ) -> list[dict]:
        """Usa o LLM para rodar a skill contra tarefas e gerar trajetórias de teste."""

        async def _run_task(task: str):
            prompt = f"Using the following SKILL GUIDE, solve the task.\n\nSKILL:\n{skill_md}\n\nTASK:\n{task}"
            # Nota: O agente deve reportar se teve sucesso no final
            res = await self.llm.generate([LLMMessage(role="user", content=prompt)])
            success = "RESULT: SUCCESS" in res.content.upper()
            return {"task": task, "trajectory": res.content, "success": success}

        return await asyncio.gather(*[_run_task(t) for t in tasks[:MAX_FLEET_SIZE]])

    # ── Helpers Privados ──────────────────────────────────────────────────────

    def _create_batches(self, trajectories: list[dict]) -> list[list[dict]]:
        """Divide as trajetórias em lotes para a frota."""
        size = FLEET_BATCH_SIZE
        return [trajectories[i:i + size] for i in range(0, len(trajectories), size)][:MAX_FLEET_SIZE]

    async def _analyze_batch(self, skill_md: str, batch: list[dict]) -> list[SkillPatch]:
        """Sub-agente analisa um lote específico."""
        traj_str = ""
        for i, t in enumerate(batch):
            status = "SUCCESS" if t['success'] else "FAILURE"
            traj_str += f"\n--- Trajectory {i} [{status}] ---\nTask: {t['task']}\nTrace: {t['trajectory'][:1000]}...\n"

        user_msg = f"CURRENT SKILL:\n{skill_md}\n\nBATCH TO ANALYZE:\n{traj_str}\n\nPropose JSON patches."

        try:
            raw = await self.llm.generate([
                LLMMessage(role="system", content=EVOLVER_SYSTEM_PROMPT),
                LLMMessage(role="user", content=user_msg)
            ], max_tokens=2000, temperature=0.2)

            data = self._extract_json(raw.content)
            if isinstance(data, list):
                return [SkillPatch(**p) for p in data if self._is_valid_patch(p)]
            return []
        except Exception as e:
            log.error("evolver.subagent_error", error=str(e))
            return []

    async def _hierarchical_merge(self, skill_md: str, groups: list[list[SkillPatch]]) -> list[SkillPatch]:
        """Consolida os patches de todos os sub-agentes (Recursivo)."""
        if not groups: return []
        if len(groups) == 1: return groups[0]

        # Para simplificar, fazemos um merge global. 
        # Em larga escala (>100 patches), faríamos merge em pares (árvore).
        all_patches_json = json.dumps([p.__dict__ for group in groups for p in group], indent=2)

        user_msg = f"SKILL CONTEXT:\n{skill_md[:500]}...\n\nPATCHES TO MERGE:\n{all_patches_json}"

        raw = await self.llm.generate([
            LLMMessage(role="system", content=MERGE_SYSTEM_PROMPT),
            LLMMessage(role="user", content=user_msg)
        ], max_tokens=3000, temperature=0.1)

        data = self._extract_json(raw.content)
        if isinstance(data, list):
            # Filtra por prevalência mínima para garantir qualidade
            return [SkillPatch(**p) for p in data if p.get('prevalence', 0) >= MIN_PREVALENCE]
        return groups[0]  # Fallback para o primeiro grupo se falhar

    def _apply_patches(self, skill_md: str, patches: list[SkillPatch]) -> str:
        """Aplica as mudanças no Markdown (Baseado em Regex/Heurística)."""
        lines = skill_md.split("\n")

        for patch in patches:
            # Tenta achar a seção (ex: ## Normative Constraints)
            section_header = f"## {patch.section}"

            # 1. Append (mais simples e seguro)
            if patch.op == "append":
                found = False
                for i, line in enumerate(lines):
                    if section_header.lower() in line.lower():
                        # Insere após o cabeçalho ou no fim da seção
                        lines.insert(i + 1, f"- {patch.content}")
                        found = True
                        break
                if not found:
                    lines.append(f"\n{section_header}\n- {patch.content}")

            # 2. Replace/Remove (requer target)
            elif patch.op in ["replace", "remove"] and patch.target:
                for i, line in enumerate(lines):
                    if patch.target.lower() in line.lower():
                        if patch.op == "replace":
                            lines[i] = f"- {patch.content}"
                        else:
                            lines.pop(i)
                        break

        # Adiciona log de evolução no final do documento
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
        lines.append(f"\n---\n*Evolved on {now} via Trace2Skill ({len(patches)} prevalent patterns identified).*")

        return "\n".join(lines)

    def _extract_json(self, text: str) -> Any:
        """Helper robusto para extrair JSON."""
        try:
            m = re.search(r'\[.*\]', text, re.DOTALL)
            if m: return json.loads(m.group(0))
            return json.loads(text)
        except:
            return None

    def _is_valid_patch(self, p: dict) -> bool:
        """Valida se o dicionário tem os campos mínimos de um SkillPatch."""
        return all(k in p for k in ["section", "op", "content"])