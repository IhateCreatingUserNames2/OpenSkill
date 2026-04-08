"""
verifier.py — EvoSkills: Surrogate Verifier & Sandbox Execution
================================================================
CORREÇÕES:
  1. diagnostic nunca mais vazio — captura stdout + stderr + returncode
  2. Sandbox isolado com sys.path limpo para evitar import side-effects
  3. Timeout com mensagem clara
  4. Wrapping robusto: código do skill + testes em namespace separado
  5. Detecção de função principal automática para os asserts
"""

from __future__ import annotations

import sys
import os
import textwrap
import tempfile
import subprocess
import structlog
from typing import Tuple

from openskill.llm.base import BaseLLMProvider, LLMMessage

log = structlog.get_logger()

VERIFIER_SYSTEM_PROMPT = (
    "You are an independent, isolated Surrogate Verifier for an AI system.\n"
    "Your ONLY job is to write deterministic Python `assert` tests to verify if a "
    "solution correctly solves a given task.\n"
    "Do NOT write the solution. ONLY write the test cases.\n"
    "IMPORTANT: Assume the solution function is already defined in the namespace.\n"
    "           Do NOT import it. Do NOT use 'from X import Y'.\n"
    "           Write ONLY assert statements and helper variables.\n"
    "Output ONLY valid Python code wrapped in ```python ... ``` blocks.\n"
    "Example output:\n"
    "```python\n"
    "assert fibonacci(0) == 0\n"
    "assert fibonacci(1) == 1\n"
    "assert fibonacci(10) == 55\n"
    "```"
)

VERIFIER_TEST_TEMPLATE = (
    "TASK:\n{task}\n\n"
    "Write Python assert statements to test the solution.\n"
    "RULES:\n"
    "- Do NOT import anything\n"
    "- Do NOT define functions\n"
    "- ONLY write assert statements\n"
    "- Test at least 5 cases including edge cases (0, 1, small, medium values)\n"
    "- The main function name is likely: {func_hint}\n"
    "Example:\n"
    "assert {func_hint}(0) == 0\n"
    "assert {func_hint}(1) == 1\n"
)

# Template do script completo que roda no subprocess
SANDBOX_TEMPLATE = '''\
import sys
import os

# Isola o sandbox do projeto principal
_project_paths = [p for p in sys.path if "OpenSkill" in p or "openskill" in p.lower()]
for _p in _project_paths:
    try:
        sys.path.remove(_p)
    except ValueError:
        pass

# ── SKILL CODE ────────────────────────────────────────────────
{skill_code}

# ── TEST CODE ─────────────────────────────────────────────────
_test_passed = 0
_test_failed = 0
_errors = []

{indented_tests}

if _test_failed == 0:
    print(f"SANDBOX_OK: {{_test_passed}} tests passed")
else:
    print(f"SANDBOX_FAIL: {{_test_failed}} failed, {{_test_passed}} passed")
    for e in _errors:
        print(f"  ERROR: {{e}}")
    sys.exit(1)
'''

# Wrapper para cada assert — captura falhas individualmente
ASSERT_WRAPPER = '''\
try:
    {assert_line}
    _test_passed += 1
except Exception as _e:
    _test_failed += 1
    _errors.append(f"{assert_line!r} → {{_e}}")
'''


def _extract_func_hint(skill_code: str) -> str:
    """Tenta adivinhar o nome da função principal no código."""
    import re
    # Procura 'def nome(' no código
    matches = re.findall(r'def\s+(\w+)\s*\(', skill_code)
    if not matches:
        return "solution"
    # Prefere nomes que não são helpers (não começam com _)
    public = [m for m in matches if not m.startswith("_")]
    if public:
        # Prefere nomes relacionados à tarefa
        for name in public:
            if any(kw in name.lower() for kw in ["fibonacci", "fib", "solution", "calc", "compute"]):
                return name
        return public[0]
    return matches[0]


def _wrap_asserts(test_code: str) -> str:
    """Envolve cada linha de assert em try/except para diagnóstico individual."""
    lines = []
    for line in test_code.strip().split("\n"):
        stripped = line.strip()
        if stripped.startswith("assert ") or stripped.startswith("assert("):
            wrapped = ASSERT_WRAPPER.format(assert_line=stripped)
            lines.append(textwrap.indent(wrapped, ""))
        elif stripped and not stripped.startswith("#"):
            # Linhas que não são assert (ex: variáveis auxiliares) — mantém direto
            lines.append(line)
    return "\n".join(lines) if lines else test_code


class SurrogateVerifier:
    def __init__(self, llm: BaseLLMProvider):
        self.llm = llm

    async def generate_tests(self, task: str, skill_code: str = "") -> str:
        """
        Gera o script de teste (verifier test suite V).
        Usa o código da skill para inferir o nome da função principal.
        """
        func_hint = _extract_func_hint(skill_code) if skill_code else "solution"

        messages = [
            LLMMessage(role="system", content=VERIFIER_SYSTEM_PROMPT),
            LLMMessage(role="user", content=VERIFIER_TEST_TEMPLATE.format(
                task=task,
                func_hint=func_hint,
            )),
        ]

        resp = await self.llm.generate(messages, max_tokens=800, temperature=0.1)
        test_code = self._extract_python_code(resp.content)

        log.debug("verifier.tests_generated", func_hint=func_hint, lines=len(test_code.split("\n")))
        return test_code

    def evaluate_in_sandbox(self, skill_code: str, test_code: str) -> Tuple[bool, str]:
        """
        Executa skill_code + test_code em subprocess isolado.

        Retorna:
            (success: bool, diagnostic: str)
            diagnostic é SEMPRE não-vazio em caso de falha.
        """
        if not skill_code or not skill_code.strip():
            return False, "DIAGNOSTIC: Skill code está vazio — nenhum código foi gerado."

        if not test_code or not test_code.strip():
            return False, "DIAGNOSTIC: Test code está vazio — nenhum teste foi gerado."

        # Envolve asserts em try/except para diagnóstico granular
        wrapped_tests = _wrap_asserts(test_code)

        # Monta o script completo
        script = SANDBOX_TEMPLATE.format(
            skill_code=skill_code,
            indented_tests=wrapped_tests,
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(script)
            tmp_path = f.name

        try:
            result = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True,
                text=True,
                timeout=15,
                # Roda no diretório temp para evitar imports acidentais do projeto
                cwd=tempfile.gettempdir(),
                env={**os.environ, "PYTHONPATH": ""},  # limpa PYTHONPATH
            )

            stdout = (result.stdout or "").strip()
            stderr = (result.stderr or "").strip()

            log.debug(
                "verifier.sandbox_result",
                returncode=result.returncode,
                stdout=stdout[:200],
                stderr=stderr[:200],
            )

            if result.returncode == 0 and "SANDBOX_OK" in stdout:
                return True, "Passed"

            # Monta diagnóstico detalhado — NUNCA vazio
            parts = []
            if stdout:
                parts.append(f"STDOUT:\n{stdout}")
            if stderr:
                parts.append(f"STDERR:\n{stderr}")
            if not parts:
                parts.append(
                    f"DIAGNOSTIC: Processo retornou código {result.returncode} sem output.\n"
                    "Possíveis causas:\n"
                    "  1. SyntaxError no código gerado\n"
                    "  2. Import de módulo não instalado\n"
                    "  3. Erro de indentação\n"
                    f"Script executado:\n{script[:500]}..."
                )

            diagnostic = "\n".join(parts)
            return False, diagnostic

        except subprocess.TimeoutExpired:
            return False, (
                "DIAGNOSTIC: Timeout (15s) — código provavelmente tem loop infinito.\n"
                "Verifique se a condição de parada da recursão está correta."
            )
        except Exception as e:
            return False, f"DIAGNOSTIC: Erro ao executar sandbox: {type(e).__name__}: {e}"
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    def _extract_python_code(self, text: str) -> str:
        """Extrai apenas o bloco de código do markdown."""
        import re
        m = re.search(r'```python\s*(.*?)\s*```', text, re.DOTALL)
        if m:
            return m.group(1).strip()
        # Fallback: retorna o texto todo se não tiver bloco markdown
        return text.strip()