"""
cross_attention.py — S-Path-RAG Gap 3: Cross-Attention Injection (Eq 6)
========================================================================
Implementa a Eq 6 do paper:

    Attn(Q_tok, K_graph, V_graph) = softmax(Q_tok @ K_graph.T / sqrt(d)) @ V_graph

Onde:
    Q_tok   = hidden states dos tokens do LLM na camada alvo
    K_graph = projeção dos vetores de skill para o espaço de keys
    V_graph = projeção dos vetores de skill para o espaço de values

Diferença da injeção por prefix (Gap 2):
    - Prefix injection: skills entram como tokens extras no início da sequência
      O LLM pode ignorá-las nas camadas profundas onde o contexto já foi consolidado.
    - Cross-attention injection: skills são injetadas diretamente nas camadas
      escolhidas via mecanismo de atenção próprio, garantindo que cada camada
      alvo "veja" o conhecimento das skills independente do que veio antes.

Implementação:
    - CrossAttentionInjector: nn.Module com K_proj, V_proj, gate por camada
    - Instalado via register_forward_hook nas últimas N camadas do Qwen
    - Gate inicializado em 0.0 → efeito nulo na inferência não treinada
    - Gate cresce com treino → efeito controlado (~0.1 a 0.3 na prática)
    - Hooks removíveis: .remove_hooks() → modelo volta ao estado original

Uso:
    injector = CrossAttentionInjector(hidden_size=2048, n_layers=7)
    injector.install(model)           # instala hooks
    injector.set_skill_context(kvecs) # define K_graph e V_graph
    out = model.generate(...)         # geração com cross-attention
    injector.remove_hooks()           # limpa hooks
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import structlog

log = structlog.get_logger()

INJECTOR_SAVE_NAME  = "cross_attn_injector.safetensors"
# Fração das camadas a receber cross-attention (últimas N)
INJECTION_LAYER_FRAC = 0.25   # 25% → Qwen3.5-2B: layers 21-27 (7 de 28)
MIN_INJECTION_LAYERS = 4


# ── Módulo principal ──────────────────────────────────────────────────────────

class CrossAttentionInjector(nn.Module):
    """
    Injeta K_graph e V_graph via cross-attention nas camadas profundas do Qwen.

    Eq 6: output += gate * Attn(Q_tok, K_graph, V_graph)

    Parâmetros treináveis por instância:
        k_proj: Linear(embed_dim, hidden_size)  — projeta skills → keys
        v_proj: Linear(embed_dim, hidden_size)  — projeta skills → values
        gates:  [n_layers]  — gate escalar por camada, init=0.0

    Parâmetros não treináveis (contexto de inferência):
        _K_graph: [1, N_skills, hidden_size]  — keys pré-computadas
        _V_graph: [1, N_skills, hidden_size]  — values pré-computadas
        _alphas:  [N_skills]  — pesos por skill (Eq 5)
    """

    def __init__(
        self,
        embed_dim: int = 384,        # dimensão dos vetores MiniLM
        hidden_size: int = 2048,     # hidden size do Qwen
        n_layers: int = 7,           # quantas camadas recebem cross-attention
        num_heads: int = 16,         # num_key_value_heads do Qwen (para head_dim)
    ):
        super().__init__()
        self.embed_dim   = embed_dim
        self.hidden_size = hidden_size
        self.n_layers    = n_layers
        self.num_heads   = num_heads
        self.head_dim    = hidden_size // num_heads

        # Projetores compartilhados entre todas as camadas
        self.k_proj = nn.Linear(embed_dim, hidden_size, bias=False)
        self.v_proj = nn.Linear(embed_dim, hidden_size, bias=False)

        # Gate por camada — inicializado em 0.0 (sem efeito inicial)
        # Usa tanh gate: output *= tanh(gate) → bounded em (-1, 1)
        self.gates = nn.Parameter(torch.zeros(n_layers))

        # Contexto de inferência (não treináveis, definidos em set_skill_context)
        self._K_graph: Optional[torch.Tensor] = None   # [1, N, H]
        self._V_graph: Optional[torch.Tensor] = None   # [1, N, H]
        self._alphas:  Optional[torch.Tensor] = None   # [N]
        self._hooks: list = []

        # Inicialização de Xavier para K e V proj
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)

    def set_skill_context(
        self,
        skill_vectors: list[np.ndarray],
        skill_alphas: Optional[list[float]] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        """
        Pré-computa K_graph e V_graph a partir dos vetores de skill.
        Deve ser chamado antes de cada geração.

        Eq 5 integrado: K e V são escalados por alpha_p antes da atenção.
        """
        if not skill_vectors:
            self._K_graph = None
            self._V_graph = None
            self._alphas  = None
            return

        # --- CORREÇÃO: Adicionado .to(dtype) no final ---
        skills_t = torch.tensor(
            np.array(skill_vectors), dtype=torch.float32, device=device
        ).to(dtype)

        with torch.no_grad():
            # Projeta para o espaço do LLM
            K = self.k_proj(skills_t)  # [N, hidden_size]
            V = self.v_proj(skills_t)  # [N, hidden_size]

            # --- CORREÇÃO: LIMITADOR DE ENERGIA (NORMALIZAÇÃO L2) ---
            # Impede que a norma exploda para 23 bilhões. Força o tamanho
            # a ser igual à biologia do LLM (sqrt(hidden_size))
            K = F.normalize(K, p=2, dim=-1) * (self.hidden_size ** 0.5)
            V = F.normalize(V, p=2, dim=-1) * (self.hidden_size ** 0.5)
            # --------------------------------------------------------

            # Escala por alpha_p (Eq 5): skills mais relevantes têm keys/values maiores
            if skill_alphas and len(skill_alphas) == len(skill_vectors):
                alpha_t = torch.tensor(
                    skill_alphas, dtype=dtype, device=device
                ).unsqueeze(-1)  # [N, 1]
                K = K * alpha_t
                V = V * alpha_t

            self._K_graph = K.unsqueeze(0)  # [1, N, H]
            self._V_graph = V.unsqueeze(0)  # [1, N, H]

        log.debug(
            "cross_attn.context_set",
            n_skills=len(skill_vectors),
            has_alphas=skill_alphas is not None,
            K_norm=f"{float(K.norm()):.3f}",
        )

    def _cross_attn_output(
        self,
        hidden_states: torch.Tensor,  # [B, T, H]
        layer_idx: int,               # índice relativo (0 = primeira camada alvo)
    ) -> torch.Tensor:
        """
        Calcula Attn(Q_tok, K_graph, V_graph) para uma camada.

        Eq 6: softmax(Q @ K_graph.T / sqrt(d)) @ V_graph
        Saída: [B, T, H] — mesmo shape que hidden_states
        """
        if self._K_graph is None or self._V_graph is None:
            return torch.zeros_like(hidden_states)

        B, T, H = hidden_states.shape
        K_g = self._K_graph.to(hidden_states.device, hidden_states.dtype)  # [1, N, H]
        V_g = self._V_graph.to(hidden_states.device, hidden_states.dtype)  # [1, N, H]

        N = K_g.shape[1]

        # Q: [B, T, H] → reshape para multi-head: [B, num_heads, T, head_dim]
        # K_g, V_g: [B, N, H] → [B, num_heads, N, head_dim]
        # (usamos hidden_size // num_heads como head_dim)
        nH  = self.num_heads
        hD  = self.head_dim

        Q = hidden_states.reshape(B, T, nH, hD).transpose(1, 2)  # [B, nH, T, hD]
        K = K_g.expand(B, -1, -1).reshape(B, N, nH, hD).transpose(1, 2)  # [B, nH, N, hD]
        V = V_g.expand(B, -1, -1).reshape(B, N, nH, hD).transpose(1, 2)  # [B, nH, N, hD]

        # Scaled dot-product attention
        scale = hD ** -0.5
        attn_w = torch.softmax(Q @ K.transpose(-2, -1) * scale, dim=-1)  # [B, nH, T, N]
        attn_out = attn_w @ V                                              # [B, nH, T, hD]

        # Volta para [B, T, H]
        attn_out = attn_out.transpose(1, 2).reshape(B, T, H)

        # Gate por camada (tanh para limitar amplitude)
        gate = torch.tanh(self.gates[layer_idx])

        return attn_out * gate

    def _make_hook(self, layer_idx: int):
        """Cria o forward hook para a camada layer_idx."""

        def hook(module, input, output):
            # output do DecoderLayer é uma tuple: (hidden_states, *extras)
            # extras podem ser: attn_weights, present_kv (opcionais)
            if isinstance(output, tuple):
                hidden_states = output[0]
                rest = output[1:]
            else:
                hidden_states = output
                rest = None

            # Só injeta se o contexto foi definido
            if self._K_graph is None:
                return output

            try:
                delta = self._cross_attn_output(hidden_states, layer_idx)
                new_hidden = hidden_states + delta

                if rest is not None:
                    return (new_hidden,) + rest
                return new_hidden

            except Exception as e:
                log.warning("cross_attn.hook_error", layer=layer_idx, error=str(e))
                return output

        return hook

    def install(self, model: nn.Module) -> None:
        """
        Instala os hooks nas últimas n_layers camadas do Qwen.
        Idempotente: remove hooks antigos antes de instalar novos.
        """
        self.remove_hooks()

        # Navega até as decoder layers
        layers = None
        for attr in ["model", "transformer"]:
            backbone = getattr(model, attr, None)
            if backbone is not None:
                layers = getattr(backbone, "layers", None)
                if layers is not None:
                    break

        if layers is None:
            log.error("cross_attn.install_failed", reason="cannot find decoder layers")
            return

        total = len(layers)
        start = max(0, total - self.n_layers)
        target_layers = list(range(start, total))

        for rel_idx, abs_idx in enumerate(target_layers):
            hook = layers[abs_idx].register_forward_hook(self._make_hook(rel_idx))
            self._hooks.append(hook)

        log.info(
            "cross_attn.installed",
            total_layers=total,
            injecting=target_layers,
            n_gates=self.n_layers,
        )

    def remove_hooks(self) -> None:
        """Remove todos os hooks instalados. Modelo volta ao estado original."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        log.debug("cross_attn.hooks_removed")

    def clear_context(self) -> None:
        """Limpa o contexto de skill após a geração."""
        self._K_graph = None
        self._V_graph = None
        self._alphas  = None

    # ── Persistência ──────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        from safetensors.torch import save_file
        state = {k: v for k, v in self.state_dict().items()
                 if not k.startswith("_")}
        save_file(state, str(path))
        log.info("cross_attn.saved", path=str(path),
                 gates=[f"{float(g):.4f}" for g in self.gates.data])

    @classmethod
    def load(
        cls,
        path: str | Path,
        embed_dim: int,
        hidden_size: int,
        n_layers: int,
        num_heads: int,
        device: str = "cpu",
    ) -> "CrossAttentionInjector":
        from safetensors.torch import load_file
        inj = cls(embed_dim, hidden_size, n_layers, num_heads)
        inj.load_state_dict(load_file(str(path)), strict=False)
        inj.to(device)
        log.info("cross_attn.loaded", path=str(path),
                 gates=[f"{float(g):.4f}" for g in inj.gates.data])
        return inj

    @property
    def is_trained(self) -> bool:
        """True se os gates divergiram de zero."""
        return float(self.gates.abs().max()) > 1e-4


# ── Helper: detecta parâmetros do Qwen automaticamente ───────────────────────

def detect_qwen_params(model: nn.Module) -> dict:
    """
    Detecta hidden_size, num_heads e n_layers a partir do model.config.
    Retorna dict compatível com CrossAttentionInjector.__init__.
    """
    cfg = model.config
    hidden_size = cfg.hidden_size

    # Qwen2/Qwen3 usa num_key_value_heads para GQA
    # Para o cross-attn usamos num_attention_heads (full heads)
    num_heads = getattr(cfg, "num_attention_heads", hidden_size // 128)

    total_layers = cfg.num_hidden_layers
    n_inj = max(MIN_INJECTION_LAYERS,
                int(total_layers * INJECTION_LAYER_FRAC))

    return {
        "hidden_size": hidden_size,
        "num_heads":   num_heads,
        "n_layers":    n_inj,
    }