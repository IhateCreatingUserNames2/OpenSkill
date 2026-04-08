"""
SkillCrafter — MemCollab: Contrastive Trajectory Distillation
=============================================================
Implementação completa do pipeline MemCollab (arXiv:2603.23234).

Princípio central:
  Memórias de agentes são agent-specific (vieses, estilo, heurísticas).
  Para criar memórias transferíveis entre modelos DIFERENTES,
  contrastamos trajetórias de um modelo FORTE vs FRACO no MESMO problema.
  O que é INVARIANTE entre ambos = princípio transferível.
  O que é específico do fraco = viés a eliminar.

Pipeline de 5 estágios:
  1. Dual Trajectory Generation   → Gera τ_fraco e τ_forte para a mesma task
  2. Contrastive Analysis          → Extrai constraints (invariants + violations)
  3. Task Classification           → Category/Subcategory (p/ retrieval task-aware)
  4. Skill Synthesis              → Monta JSON estruturado da skill
  5. Markdown Render               → Produz SKILL.md final

Não faz I/O — recebe LLMProvider e retorna dados puros.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import structlog

from openskill.core.verifier import SurrogateVerifier
from openskill.llm.base import BaseLLMProvider, LLMMessage

log = structlog.get_logger()


def _slugify(text: str) -> str:
    """Converte 'Optimal Fibonacci' para 'optimal-fibonacci' (Padrão Agent Skills)."""
    import re
    # Remove caracteres especiais, troca espaços por hífens e deixa minúsculo
    text = re.sub(r'[^a-zA-Z0-9\s-]', '', text).strip().lower()
    text = re.sub(r'[\s-]+', '-', text)
    return text[:64] # O padrão exige max 64 chars

# ── Data Classes de Output ───────────────────────────────────────────────────

@dataclass
class DualTrajectories:
    """Par de trajetórias para o mesmo problema."""
    task: str
    weak_model: str
    strong_model: str
    weak_trajectory: str
    strong_trajectory: str
    weak_success: bool = False
    strong_success: bool = False

    @property
    def preferred(self) -> tuple[str, bool]:
        """Retorna (trajetória preferida, se veio do forte)."""
        if self.strong_success:
            return self.strong_trajectory, True
        if self.weak_success:
            return self.weak_trajectory, False
        # Nenhum成功 — usa o forte por ser mais capaz
        return self.strong_trajectory, True

    @property
    def unpreferred(self) -> tuple[str, bool]:
        """Retorna (trajetória não-preferida, se veio do fraco)."""
        if self.strong_success:
            return self.weak_trajectory, False
        if self.weak_success:
            return self.strong_trajectory, True
        return self.weak_trajectory, False


@dataclass
class TaskClassification:
    """Resultado da classificação de tarefa para retrieval task-aware."""
    category: str
    subcategory: str


@dataclass
class ExtractedConstraints:
    """
    Constraints extraídas da análise contrastiva.
    Formato: lista de strings "When X, enforce Y; avoid Z"
    """
    items: list[str]
    reasoning: str = ""  # Trace do raciocínio do LLM

    def __len__(self) -> int:
        return len(self.items)

    def __bool__(self) -> bool:
        return len(self.items) > 0

    def to_list(self) -> list[str]:
        return self.items


@dataclass
class SkillData:
    """Dados estruturados de uma skill (antes de renderizar)."""
    title: str
    domain: str
    description: str
    category: str
    subcategory: str
    invariants: list[str] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    when_to_apply: str = ""
    example_pattern: str = ""
    source_constraints: list[str] = field(default_factory=list)


# ── Prompts do Sistema ───────────────────────────────────────────────────────

CRAFTER_SYSTEM_PROMPT = (
    "You are an expert skill architect for AI reasoning agents.\n"
    "Generate structured, reusable skill documents in JSON format.\n"
    "Skills encode NORMATIVE CONSTRAINTS — rules that agents MUST follow "
    "to produce correct reasoning.\n\n"
    "Output ONLY valid JSON. No markdown fences. No preamble."
)

TRAJECTORY_SYSTEM_PROMPT = (
    "You are a reasoning agent solving the given task step-by-step.\n"
    "Show your FULL reasoning, intermediate steps, code/formulas, and final answer.\n"
    "Be precise. Every step must be logged."
)

TRAJECTORY_USER_TEMPLATE = "Task:\n{task}\n\nSolve this problem completely."


CONSTRASTIVE_SYSTEM_PROMPT = (
    "You are an expert analyst extracting reusable REASONING MEMORY from\n"
    "contrastive multi-step reasoning trajectories.\n\n"
    "Extract:\n"
    "1) Reusable failure-aware reasoning constraints\n"
    "2) High-level reasoning strategies that characterize CORRECT reasoning\n\n"
    "Each strategy MUST:\n"
    "- Be written as ONE sentence\n"
    "- Follow this FORMAT: 'When ... , enforce ... ; avoid ...'\n"
    "- Be ABSTRACT and REUSABLE across different problems\n"
    "- NOT reference specific numbers, constants, or task details\n\n"
    "Output ONLY a numbered list of strategies. No explanations. No preamble. "
    "No markdown."
)

CONSTRASTIVE_USER_TEMPLATE = (
    "TASK:\n{task}\n\n"
    "PREFERRED TRAJECTORY (correct reasoning):\n{preferred}\n\n"
    "UNPREFERRED TRAJECTORY (incorrect or suboptimal reasoning):\n{unpreferred}\n\n"
    "Extract 3-8 reusable reasoning constraints in the format:\n"
    "'When [condition], enforce [principle]; avoid [failure pattern]'"
)

SKILL_BUNDLE_USER_TEMPLATE = (
    "TASK DOMAIN: {task}\n"
    "CONSTRAINTS:\n{constraints}\n"
    "STRONG AGENT APPROACH:\n{strong_trajectory}\n\n"
    "You are creating a Multi-file EvoSkill Bundle. You must provide TWO outputs:\n"
    "1. A JSON object with the SKILL.md metadata (title, invariants, etc).\n"
    "2. Executable Python code containing utility functions to solve this task.\n\n"
    "Format your response exactly like this:\n"
    "```json\n{{ ... }}\n```\n"
    "```python\n# Your executable code here\n```"
)

REFINE_CODE_TEMPLATE = (
    "Your previous code attempt for task '{task}' failed the verification.\n\n"
    "CURRENT CODE:\n```python\n{current_code}\n```\n\n"
    "DIAGNOSTIC FEEDBACK:\n{diagnostic}\n\n"
    "Fix the errors and provide the updated executable Python code wrapped in ```python ... ```."
)


CLASSIFY_SYSTEM_PROMPT = (
    "You are an expert Task Classifier for an AI Memory System.\n"
    "Classify the task into 'category' and 'subcategory'.\n\n"
    "Categories and subcategories:\n"
    "  Mathematics: Algebra, Geometry, Combinatorics, Number Theory, Calculus, Probability\n"
    "  Programming: Algorithms, Data Structures, Debugging, System Design, Web Dev\n"
    "  Logic/Reasoning: Puzzles, Planning, Fallacy Detection\n"
    "  General: General\n\n"
    "Respond ONLY with valid JSON: {{\"category\": \"...\", \"subcategory\": \"...\"}}"
)

CLASSIFY_USER_TEMPLATE = "TASK:\n{task}"


SYNTHESIZE_USER_TEMPLATE = (
    "TASK DOMAIN: {task}\n\n"
    "EXTRACTED CONSTRAINTS:\n{constraints}\n\n"
    "STRONG AGENT APPROACH (reference):\n{strong_trajectory}\n\n" 
    "Generate a SKILL with this exact JSON structure:\n"
    "{{"
    '  "title": "short skill name",\n'
    '  "domain": "Mathematics | Programming | Logic/Reasoning | General",\n'
    '  "description": "what this skill teaches in one sentence",\n'
    '  "invariants": ["essential principle 1", ...],\n'
    '  "violations": ["forbidden pattern 1", ...],\n'
    '  "constraints": ["enforce X; avoid Y", ...],\n'
    '  "when_to_apply": "trigger description",\n'
    '  "example_pattern": "brief abstract example"\n'
    "}}"
)

# ── LLM Helpers (privados) ──────────────────────────────────────────────────

async def _call_llm(
    llm: BaseLLMProvider,
    messages: list[LLMMessage],
    max_tokens: int,
    temperature: float = 0.7,
) -> str:
    """Chamada básica ao LLM com error handling."""
    try:
        resp = await llm.generate(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.content.strip()
    except Exception as e:
        log.error("crafter.llm.error", error=str(e))
        raise
def _strip_think(text: str) -> str:
    """
    Remove blocos de raciocínio <think>...</think> comuns em modelos de CoT (Chain of Thought).
    Isso evita que o log de pensamento do modelo interfira no parsing de JSON ou listas.
    """
    if not text:
        return ""
    # Remove as tags <think> e tudo que estiver dentro delas (non-greedy)
    # flags=re.DOTALL permite que o '.' capture quebras de linha
    # flags=re.IGNORECASE lida com variações como <THINK>
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()

class SkillCrafter:
    """
    Orquestrador do pipeline MemCollab.
    Transforma uma tarefa bruta em uma Skill.md refinada.
    """

    def __init__(self, llm: BaseLLMProvider):
        self.llm = llm

    async def generate_trajectories(
            self, task: str, weak_model: str, strong_model: str
    ) -> tuple[str, str]:
        """Gera as trajetórias tau_w e tau_s em paralelo."""
        import asyncio

        async def _run(model_id: str):
            messages = [
                LLMMessage(role="system", content=TRAJECTORY_SYSTEM_PROMPT),
                LLMMessage(role="user", content=TRAJECTORY_USER_TEMPLATE.format(task=task)),
            ]
            # Nota: Aqui o BaseLLMProvider deve suportar a troca de modelo se for OpenRouter,
            # ou ignorar se for um modelo local fixo.
            resp = await self.llm.generate(messages, max_tokens=3000)
            return resp.content

        # Execução paralela para performance
        return await asyncio.gather(_run(weak_model), _run(strong_model))

    async def co_evolve_skill_bundle(
            self,
            task: str,
            constraints: list[str],
            strong_trajectory: str,
            verifier: 'SurrogateVerifier',
            max_iters: int = 5
    ) -> tuple[dict, str]:
        """
        Algoritmo 1 (EvoSkills): Loop de Co-Evolução.
        Gera a lógica, escreve o código, testa e refina iterativamente.

        CORREÇÃO: passa skill_code ao generate_tests para inferência correta
                  do nome da função. diagnostic nunca mais vazio.
        """
        constraints_str = "\n".join([f"- {c}" for c in constraints])

        messages = [
            LLMMessage(role="system", content=CRAFTER_SYSTEM_PROMPT),
            LLMMessage(role="user", content=SKILL_BUNDLE_USER_TEMPLATE.format(
                task=task,
                constraints=constraints_str,
                strong_trajectory=strong_trajectory[:800]
            )),
        ]

        # 1ª Geração (One-Shot)
        raw_response = await _call_llm(self.llm, messages, max_tokens=3000)
        skill_json = self._extract_json(raw_response) or {"title": "Unnamed", "domain": "General"}
        skill_code = self._extract_python_code(raw_response)

        # Passa skill_code para que o verifier infira o nome da função principal
        log.info("evoskills.generating_surrogate_tests", task=task[:30])
        test_code = await verifier.generate_tests(task, skill_code=skill_code)

        # Loop de Evolução Iterativa (co-evolutionary loop)
        for i in range(max_iters):
            log.info("evoskills.verifying_iteration", iteration=i + 1, max=max_iters)

            success, diagnostic = verifier.evaluate_in_sandbox(skill_code, test_code)

            if success:
                log.info("evoskills.verification_passed", iteration=i + 1)
                break

            # diagnostic agora SEMPRE contém informação útil
            log.warning(
                "evoskills.verification_failed",
                iteration=i + 1,
                diagnostic=diagnostic[:200],
            )

            if not skill_code.strip():
                # Código vazio — gera do zero em vez de tentar refinar
                log.warning("evoskills.empty_code_regenerating", iteration=i + 1)
                raw_response = await _call_llm(self.llm, messages, max_tokens=3000)
                skill_json = self._extract_json(raw_response) or skill_json
                skill_code = self._extract_python_code(raw_response)
                # Atualiza testes para o novo código
                test_code = await verifier.generate_tests(task, skill_code=skill_code)
                continue

            # Refinamento com feedback real do verifier (Eq 5, 7 do EvoSkills)
            refine_msgs = [
                LLMMessage(
                    role="system",
                    content=(
                        "You are a code refinement agent. Fix the Python code based on "
                        "the test diagnostics below. Return ONLY the corrected Python code "
                        "wrapped in ```python ... ``` blocks. Do not add explanations."
                    ),
                ),
                LLMMessage(role="user", content=REFINE_CODE_TEMPLATE.format(
                    task=task,
                    current_code=skill_code,
                    diagnostic=diagnostic,
                )),
            ]
            refine_resp = await _call_llm(self.llm, refine_msgs, max_tokens=2500)
            new_code = self._extract_python_code(refine_resp)

            if new_code.strip():
                skill_code = new_code
                # Atualiza testes se o nome da função mudou
                test_code = await verifier.generate_tests(task, skill_code=skill_code)

        return skill_json, skill_code

    def _extract_python_code(self, text: str) -> str:
        import re
        m = re.search(r'```python\s*(.*?)\s*```', text, re.DOTALL)
        return m.group(1).strip() if m else ""

    async def contrastive_analysis(
            self, task: str, preferred: str, unpreferred: str
    ) -> list[str]:
        """Extrai as lições invariantes entre as duas tentativas."""
        messages = [
            LLMMessage(role="system", content=CONSTRASTIVE_SYSTEM_PROMPT),
            LLMMessage(role="user", content=CONSTRASTIVE_USER_TEMPLATE.format(
                task=task, preferred=preferred, unpreferred=unpreferred
            )),
        ]
        raw = await _call_llm(self.llm, messages, max_tokens=2000, temperature=0.3)
        cleaned = _strip_think(raw)

        # Parse simples de lista numerada
        items = re.findall(r'^\d+[\.\)]\s*(.*)', cleaned, re.MULTILINE)
        return items if items else [cleaned]

    async def classify_task(self, task: str) -> dict:
        """Determina a categoria para o retrieval geométrico futuro."""
        messages = [
            LLMMessage(role="system", content=CLASSIFY_SYSTEM_PROMPT),
            LLMMessage(role="user", content=CLASSIFY_USER_TEMPLATE.format(task=task)),
        ]
        raw = await _call_llm(self.llm, messages, max_tokens=200, temperature=0.0)
        return self._extract_json(raw) or {"category": "General", "subcategory": "General"}

    async def synthesize_skill(
            self,
            task: str,
            constraints: list[str],
            weak_trajectory: str,
            strong_trajectory: str
    ) -> dict:
        """Funde as lições em um objeto Skill estruturado."""
        constraints_str = "\n".join([f"- {c}" for c in constraints])

        messages = [
            LLMMessage(role="system", content=CRAFTER_SYSTEM_PROMPT),
            LLMMessage(role="user", content=SYNTHESIZE_USER_TEMPLATE.format(
                task=task,
                constraints=constraints_str,
                strong_trajectory=strong_trajectory[:800]  # Slice feito aqui!
            )),
        ]
        raw = await _call_llm(self.llm, messages, max_tokens=2500, temperature=0.5)
        return self._extract_json(raw) or {"title": "Extraction Error"}

    def render_markdown(
            self,
            skill: dict,
            task: str,
            weak_model: str,
            strong_model: str,
            weak_traj: str,
            strong_traj: str,
            constraints: list[str]
    ) -> str:
        """Renderiza o arquivo SKILL.md 100% compatível com o padrão Agent Skills."""

        title = skill.get('title', 'Unnamed Skill')
        slug_name = _slugify(title)

        # Evita quebra de YAML e limita tamanho
        description = skill.get('description', f"Workflow and utilities for {task}").strip()
        when_to_apply = skill.get('when_to_apply', description).strip()
        yaml_desc = (description + " " + when_to_apply)[:1000]

        inv_md = "\n".join(f"- {i}" for i in skill.get("invariants", []))
        viol_md = "\n".join(f"- ⚠️ {v}" for v in skill.get("violations", []))
        con_md = "\n".join(f"- {c}" for c in skill.get("constraints", []))

        # IMPORTANTE: sem indentação no início
        return f"""---
    name: {slug_name}
    description: "{yaml_desc}"
    metadata:
      domain: {skill.get('domain', 'General')}
      generated_by: OpenSkill EvoSkills Framework
      weak_agent: {weak_model}
      strong_agent: {strong_model}
    ---

    # {title}

    > {description}

    ## Available Scripts
    - `scripts/utils.py` - Core utility functions for this skill.

    ## Reasoning Invariants
    {inv_md}

    ## Violation Patterns
    {viol_md}

    ## Normative Constraints
    {con_md}

    ---
    
    ## Example Pattern
    {skill.get('example_pattern', 'No example provided.')}
    Or
    ```python
    import sys
    # Always use relative paths when executing inside the skill directory
    sys.path.insert(0, 'scripts')
    from utils import * ```
    Code
    ---
    
    ## Source: MemCollab Analysis
    *Analysis performed by contrasting {strong_model} against {weak_model}.*
    """

    def _extract_json(self, text: str) -> Optional[dict]:
        """Extrai JSON de forma ultra-robusta, limpando blocos de raciocínio."""
        text = _strip_think(text)
        try:
            # 1. Tenta achar bloco ```json ... ```
            import re
            m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
            if m:
                return json.loads(m.group(1))

            # 2. Tenta achar o primeiro { e o último } no texto todo
            m = re.search(r'(\{.*\})', text, re.DOTALL)
            if m:
                return json.loads(m.group(1))

            return json.loads(text)
        except Exception:
            # Se falhar, tenta extrair o título via Regex simples como último recurso
            title_match = re.search(r'"title":\s*"(.*?)"', text)
            if title_match:
                return {"title": title_match.group(1)}
            return None