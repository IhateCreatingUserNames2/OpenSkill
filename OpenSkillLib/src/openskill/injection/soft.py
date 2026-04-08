"""
soft.py — SkillProjector + Soft Latent Injection (S-Path-RAG Gap 2)
====================================================================
Alterações em relação à versão anterior:

  - SkillProjector: adicionado save() / load() para persistir pesos treinados
  - capture_attn_mass(): captura attnMass(p) via output_attentions=True
    sem precisar de forward hooks — usa o model() diretamente
  - inject_skills_to_embeds(): usa alphas reais (guidance.skill_alphas) em vez
    de escala fixa 0.1 quando disponíveis
  - _alpha_scale(): converte alpha normalizado para escala de energia compatível

Mecânica do L_align (Eq 7 + 8 do paper):
  attnMass(p) = (1/T_tok) * Σ_t Σ_{k ∈ idx(p)} A_{t,k}
  onde A_{t,k} é o peso de atenção do token t para a posição k da skill p.
  L_align = (1/|P_sel|) * Σ_p (alpha_p - attnMass_p)²
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

import structlog

log = structlog.get_logger()

# Camadas do Qwen usadas para medir attnMass
# Usamos as últimas N camadas — onde a atenção é mais semântica
ATTN_LAYERS_FRAC = 0.5   # últimos 50% das camadas
MIN_ATTN_LAYERS  = 4     # mínimo absoluto


class SkillProjector(nn.Module):
    """
    Projeta vetores de skill (384d) para o espaço oculto do LLM.

    Linear(embed_dim → llm_hidden_size) + LayerNorm.
    Treinado com L_align para que o LLM efetivamente atenda ao que é injetado.
    """

    def __init__(self, embed_dim: int, llm_hidden_size: int):
        super().__init__()
        self.embed_dim      = embed_dim
        self.llm_hidden_size = llm_hidden_size
        self.proj = nn.Linear(embed_dim, llm_hidden_size)
        self.ln   = nn.LayerNorm(llm_hidden_size)

    def forward(self, skill_vectors: torch.Tensor) -> torch.Tensor:
        """[N, D_skill] → [N, D_llm]  ou  [B, N, D_skill] → [B, N, D_llm]"""
        return self.ln(self.proj(skill_vectors))

    # ── Persistência ──────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Salva apenas os pesos do projector (não o scorer)."""
        from safetensors.torch import save_file
        save_file(self.state_dict(), str(path))
        log.info("projector.saved", path=str(path))

    @classmethod
    def load(
        cls,
        path: str | Path,
        embed_dim: int,
        llm_hidden_size: int,
        device: str = "cpu",
    ) -> "SkillProjector":
        """Carrega projector treinado do disco."""
        from safetensors.torch import load_file
        proj = cls(embed_dim, llm_hidden_size)
        proj.load_state_dict(load_file(str(path)))
        proj.to(device)
        proj.eval()
        log.info("projector.loaded", path=str(path))
        return proj

    @property
    def is_trained(self) -> bool:
        """Heurística: se os pesos divergiram de eye_, o projector foi treinado."""
        w = self.proj.weight.data
        if w.shape[0] != w.shape[1]:
            return True   # Não-quadrada — nunca foi eye_, portanto foi inicializada aleatória
        eye = torch.eye(w.shape[0], device=w.device, dtype=w.dtype)
        # Se a distância de Frobenius da identidade for > 0.01, consideramos treinado
        return float((w - eye).norm()) > 0.01


# ── Captura de attnMass via output_attentions ─────────────────────────────────

def capture_attn_mass(
    model: nn.Module,
    input_embeds: torch.Tensor,       # [1, T_total, D]  — skills + prompt
    attention_mask: torch.Tensor,     # [1, T_total]
    n_skill_tokens: int,              # N: primeiros N tokens são skills injetadas
    skill_alphas: list[float],        # alpha_p por skill (Eq 5, já normalizado)
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calcula attnMass(p) para cada skill p e retorna L_align.

    Eq 7: attnMass(p) = (1/T_tok) * Σ_t Σ_{k=idx(p)} A_{t,k}
    Eq 8: L_align = (1/N) * Σ_p (alpha_p - attnMass_p)²

    Retorna:
        l_align  : tensor escalar com gradiente (para backward no projector)
        attn_mass: tensor [N] com attnMass por skill (para logging)

    Notas:
      - Usa model() com output_attentions=True, não model.generate()
      - O LLM permanece congelado (no_grad sobre seus parâmetros)
      - Apenas o projector recebe gradiente via input_embeds
    """
    if n_skill_tokens == 0 or not skill_alphas:
        zero = torch.tensor(0.0, requires_grad=True, device=device)
        return zero, torch.zeros(0, device=device)

    # Garante que o modelo está em modo eager para suportar output_attentions=True.
    # Se foi carregado com attn_implementation="eager" (projector_trainer.py),
    # este bloco é no-op. Se chamado de outro contexto (sdpa padrão), corrige.
    _orig_impl = getattr(model.config, "_attn_implementation", "eager")
    if _orig_impl != "eager":
        model.config._attn_implementation = "eager"
        for m in model.modules():
            if hasattr(m, "_attn_implementation"):
                m._attn_implementation = "eager"

    with torch.set_grad_enabled(True):
        outputs = model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            output_attentions=True,
            use_cache=False,
        )

    # Restaura se tínhamos mudado
    if _orig_impl != "eager":
        model.config._attn_implementation = _orig_impl
        for m in model.modules():
            if hasattr(m, "_attn_implementation"):
                m._attn_implementation = _orig_impl

    if outputs.attentions is None or len(outputs.attentions) == 0:
        log.warning("projector.no_attentions_returned")
        zero = torch.tensor(0.0, requires_grad=True, device=device)
        return zero, torch.zeros(n_skill_tokens, device=device)

    # Indexa apenas as camadas efetivamente retornadas (não num_hidden_layers)
    n_returned    = len(outputs.attentions)
    n_attn_layers = max(MIN_ATTN_LAYERS, int(n_returned * ATTN_LAYERS_FRAC))
    target_layers = list(range(n_returned - n_attn_layers, n_returned))

    log.debug(
        "projector.attn_capture",
        n_returned=n_returned,
        n_used=len(target_layers),
    )

    attn_stack = torch.stack(
        [outputs.attentions[i] for i in target_layers], dim=0
    )  # [L, 1, H, T_total, T_total]

    # Tokens do prompt: posições [n_skill_tokens : T_total]
    # Atenção deles para as skills: attn[..., n_skill_tokens:, :n_skill_tokens]
    T_total = input_embeds.shape[1]
    T_prompt = T_total - n_skill_tokens

    if T_prompt <= 0:
        zero = torch.tensor(0.0, requires_grad=True, device=device)
        return zero, torch.zeros(n_skill_tokens, device=device)

    # Seleciona atenção dos tokens do prompt para cada posição de skill
    # attn_to_skills: [L, 1, H, T_prompt, N_skills]
    attn_to_skills = attn_stack[
        :, :, :, n_skill_tokens:, :n_skill_tokens
    ]

    # Média sobre layers, batch, heads, prompt_tokens → [N_skills]
    attn_mass = attn_to_skills.mean(dim=(0, 1, 2, 3))  # [N_skills]

    # Normaliza attn_mass para [0,1] (mesmo espaço dos alphas normalizados)
    attn_mass_norm = attn_mass / (attn_mass.sum() + 1e-8)

    # Alpha tensor — sem gradiente (calculado pelo scorer, não pelo projector)
    N = min(n_skill_tokens, len(skill_alphas))
    alpha_t = torch.tensor(
        skill_alphas[:N], dtype=torch.float32, device=device
    )

    # Eq 8: L_align = mean((alpha_p - attnMass_p)²)
    l_align = ((alpha_t - attn_mass_norm[:N]) ** 2).mean()

    return l_align, attn_mass_norm.detach()


# ── inject_skills_to_embeds (atualizado com alphas reais) ─────────────────────

def inject_skills_to_embeds(
    input_embeds: torch.Tensor,          # [1, T, D]
    skill_vectors: list[np.ndarray],
    projector: SkillProjector,
    device: str = "cuda",
    skill_alphas: Optional[list[float]] = None,  # NOVO — Eq 5
) -> torch.Tensor:
    """
    Injeta os vetores de skill como prefix tokens no espaço do LLM.

    Se skill_alphas fornecido (scorer treinado), escala cada skill pelo seu
    alpha_p normalizado * prompt_norm, em vez de escala fixa 0.1.
    """
    if not skill_vectors:
        return input_embeds

    skills_tensor = torch.tensor(
        np.array(skill_vectors),
        dtype=projector.proj.weight.dtype,
        device=device,
    )  # [N, D_skill]

    with torch.no_grad():
        projected = projector(skills_tensor).unsqueeze(0)  # [1, N, D_llm]

        prompt_norm = input_embeds.norm(p=2, dim=-1).mean()

        if skill_alphas and len(skill_alphas) == len(skill_vectors):
            # Escala por alpha_p: skills mais relevantes têm mais energia
            alphas = torch.tensor(
                skill_alphas, dtype=projected.dtype, device=device
            ).unsqueeze(-1)  # [N, 1]
            # alpha já normalizado (soma = 1), multiplica por prompt_norm
            projected = projected * (alphas * prompt_norm).unsqueeze(0)
        else:
            # Fallback: normalização de energia uniforme (comportamento anterior)
            skill_norm = projected.norm(p=2, dim=-1).mean()
            projected  = projected * (prompt_norm / (skill_norm + 1e-8))

    return torch.cat([projected, input_embeds], dim=1)  # [1, N+T, D_llm]


def create_injected_attention_mask(
    original_mask: torch.Tensor,
    num_skills: int,
    device: str = "cuda",
) -> torch.Tensor:
    """Expande a attention mask para incluir os tokens de skill injetados."""
    batch_size = original_mask.shape[0]
    skill_mask = torch.ones(
        (batch_size, num_skills), dtype=original_mask.dtype, device=device
    )
    return torch.cat([skill_mask, original_mask], dim=1)