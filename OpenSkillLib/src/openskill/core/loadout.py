"""
loadout.py — The Hotbar System
================================
Gerencia quais skills estão ativas na RAM (Contexto do Agente) vs
quais estão "equipadas" como auras passivas (Espaço Latente / S-Path RAG).
Resolve o problema de "Context Bloat" e define limites rígidos.
"""

from typing import List, Dict, Any
from openskill.storage.base import SkillMetadata, SkillType
import structlog

log = structlog.get_logger()


class SkillLoadout:
    def __init__(self, max_active_slots=3, max_passive_slots=5):
        self.max_active = max_active_slots
        self.max_passive = max_passive_slots

        self.equipped_active: List[Dict[str, Any]] = []  # Vão para o Prompt (Tokens)
        self.equipped_passive: List[Dict[str, Any]] = []  # Vão para Injeção Latente (Geometria)

    def equip(self, meta: SkillMetadata, content: str, code_path: str = "") -> bool:
        """Tenta equipar uma skill na Hotbar. Retorna False se não houver Slot."""

        skill_payload = {
            "id": meta.id,
            "title": meta.title,
            "content": content,
            "code_path": code_path,
            "vectors": meta.vectors  # Necessário para o S-Path RAG
        }

        # Equipa baseada na taxonomia de Game Design
        if meta.skill_type == SkillType.PASSIVE:
            if len(self.equipped_passive) < self.max_passive:
                self.equipped_passive.append(skill_payload)
                log.info("loadout.equipped_passive", skill=meta.title)
                return True
            else:
                log.warning("loadout.passive_slots_full", skill=meta.title)
                return False

        elif meta.skill_type in [SkillType.ACTIVE, SkillType.HYBRID]:
            if len(self.equipped_active) < self.max_active:
                self.equipped_active.append(skill_payload)
                # Se for híbrida, equipa passivamente também (Sinergia)
                if meta.skill_type == SkillType.HYBRID and len(self.equipped_passive) < self.max_passive:
                    self.equipped_passive.append(skill_payload)
                log.info("loadout.equipped_active", skill=meta.title)
                return True
            else:
                log.warning("loadout.active_slots_full", skill=meta.title)
                return False

    def generate_system_prompt_appendage(self) -> str:
        """Gera o texto que será injetado no prompt do Agente para as Active Skills."""
        import os

        if not self.equipped_active:
            return ""

        prompt = "\n\n[EQUIPPED ACTIVE SKILLS]\n"
        prompt += "You have the following tested and executable skills equipped in your Hotbar. "
        prompt += "DO NOT write these functions from scratch. Import and exact functions provided below.\n\n"

        for skill in self.equipped_active:
            prompt += f"### {skill['title']}\n"
            prompt += f"Description/Constraints: {skill['content'][:200]}...\n"  # Resumo

            if skill['code_path']:
                prompt += f"Usage: Add `import sys; sys.path.append(r'{skill['code_path']}')` to your environment.\n"

                # NOVO: Tenta ler o utils.py e extrair o nome das funções para evitar alucinações
                utils_file = os.path.join(skill['code_path'], "utils.py")
                if os.path.exists(utils_file):
                    try:
                        with open(utils_file, 'r', encoding='utf-8') as f:
                            code_content = f.read()
                            # Injeta as primeiras linhas de código ou as definições de função
                            import ast
                            tree = ast.parse(code_content)
                            functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
                            prompt += f"Available functions in `utils.py`: {functions}\n"
                            prompt += f"Example Import: `from utils import {functions[0] if functions else '*'}`\n"
                    except Exception as e:
                        pass
                prompt += "\n"

        return prompt