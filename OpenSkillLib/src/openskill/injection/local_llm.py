"""
local_llm.py — LocalSkillInjectedLLM (Gap 3: cross-attention injection)
=========================================================================
Modos de geração disponíveis:
  verbalization   → extrai constraints do Markdown, injeta como texto (estável)
  prefix          → injeta skills como prefix tokens (Gap 2, usa projector)
  cross_attention → injeta K_graph/V_graph nas camadas profundas (Gap 3)
  auto            → escolhe o melhor modo disponível automaticamente

Hierarquia de auto:
  cross_attention (se injector treinado)
  → prefix (se projector treinado)
  → verbalization (sempre disponível)

Novidades nesta versão:
  - Carrega CrossAttentionInjector de cross_attn_injector.safetensors
  - _generate_cross_attention(): instala hooks, gera, limpa contexto
  - mode="cross_attention" disponível no CLI e na API
  - auto detecta o melhor modo sem configuração manual
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import structlog
from transformers import AutoModelForCausalLM, AutoTokenizer

from openskill.llm.base import BaseLLMProvider, LLMMessage, LLMResponse
from openskill.injection.soft import (
    SkillProjector,
    inject_skills_to_embeds,
    create_injected_attention_mask,
)
from openskill.retrieval.retriever import RetrievalGuidance

log = structlog.get_logger()

PROJECTOR_SAVE_NAME  = "projector_weights.safetensors"
CROSS_ATTN_SAVE_NAME = "cross_attn_injector.safetensors"


class LocalSkillInjectedLLM(BaseLLMProvider):
    def __init__(
        self,
        model_id: str,
        skill_dir: str = "./skills_output",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_4bit: bool = True,
    ):
        self._model_id = model_id
        self.device    = device
        self.skill_dir = Path(skill_dir)

        print(f"Loading LLM Engine ({model_id})...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        load_kwargs = {
            "device_map": "auto",
            "trust_remote_code": True,
            "torch_dtype": self.dtype,
        }

        if use_4bit and device == "cuda":
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self.dtype,
            )

        self.model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        hidden_size = self.model.config.hidden_size

        # Gap 2: Projector
        self.projector = self._load_projector(hidden_size)

        # Gap 3: CrossAttentionInjector
        self.cross_attn_injector = self._load_cross_attn_injector()

    # ── Loaders ──────────────────────────────────────────────────────────────

    def _load_projector(self, hidden_size: int) -> SkillProjector:
        path = self.skill_dir / PROJECTOR_SAVE_NAME
        if path.exists():
            try:
                proj = SkillProjector.load(
                    path, embed_dim=384, llm_hidden_size=hidden_size, device=self.device
                )
                proj = proj.to(self.dtype)
                log.info("local_llm.projector_loaded", path=str(path))
                return proj
            except Exception as e:
                log.warning("local_llm.projector_load_failed", error=str(e))

        proj = SkillProjector(embed_dim=384, llm_hidden_size=hidden_size)
        proj = proj.to(self.device).to(self.dtype)
        torch.nn.init.eye_(proj.proj.weight)
        log.info("local_llm.projector_fallback_eye", hidden_size=hidden_size)
        return proj

    def _load_cross_attn_injector(self):
        """Carrega ou cria o CrossAttentionInjector e instala os hooks."""
        from openskill.injection.cross_attention import (
            CrossAttentionInjector, detect_qwen_params
        )

        params = detect_qwen_params(self.model)
        path   = self.skill_dir / CROSS_ATTN_SAVE_NAME

        if path.exists():
            try:
                inj = CrossAttentionInjector.load(
                    path,
                    embed_dim=384,
                    hidden_size=params["hidden_size"],
                    n_layers=params["n_layers"],
                    num_heads=params["num_heads"],
                    device=self.device,
                )
                inj = inj.to(self.dtype)
                inj.install(self.model)
                log.info(
                    "local_llm.cross_attn_loaded",
                    path=str(path),
                    trained=inj.is_trained,
                    gates=[f"{float(g):.4f}" for g in inj.gates.data],
                )
                return inj
            except Exception as e:
                log.warning("local_llm.cross_attn_load_failed", error=str(e))

        # Injector não treinado: gates=0 → efeito nulo, mas já com hooks
        inj = CrossAttentionInjector(
            embed_dim=384,
            hidden_size=params["hidden_size"],
            n_layers=params["n_layers"],
            num_heads=params["num_heads"],
        )
        inj = inj.to(self.device).to(self.dtype)
        inj.install(self.model)
        log.info(
            "local_llm.cross_attn_untrained",
            n_layers=params["n_layers"],
            hint=f"Run: python cross_attn_trainer.py --skill-dir {self.skill_dir}",
        )
        return inj

    @property
    def model_id(self) -> str:
        return self._model_id

    # ── Decisor de pipeline ───────────────────────────────────────────────────

    async def generate_with_guidance(
        self,
        query: str,
        guidance: RetrievalGuidance,
        mode: str = "auto",
        max_new_tokens: int = 2048,
    ) -> LLMResponse:
        has_vectors  = bool(guidance.skill_vectors)
        has_contents = bool(guidance.skill_contents)

        if mode == "auto":
            if has_vectors and self.cross_attn_injector and self.cross_attn_injector.is_trained:
                mode = "cross_attention"
            elif has_vectors and self.projector.is_trained:
                mode = "prefix"
            elif has_contents:
                mode = "verbalization"
            else:
                mode = "plain"
            log.info("local_llm.auto_mode", selected=mode)

        if mode == "cross_attention" and has_vectors and self.cross_attn_injector:
            return await self._generate_cross_attention(query, guidance, max_new_tokens)

        if mode in ("injection", "prefix") and has_vectors:
            return await self._generate_prefix(query, guidance, max_new_tokens)

        if mode == "verbalization" and has_contents:
            return await self._generate_verbalized_strict(query, guidance, max_new_tokens)

        return await self.generate([LLMMessage(role="user", content=query)])

    # ── Modo 1: Verbalization ─────────────────────────────────────────────────

    async def _generate_verbalized_strict(
        self, query: str, guidance: RetrievalGuidance, max_tokens: int
    ) -> LLMResponse:
        strict_rules = []
        for content in guidance.skill_contents:
            for line in content.split("\n"):
                l = line.strip().lower()
                if l.startswith("- enforce") or l.startswith("- avoid") or l.startswith("- ⚠️"):
                    strict_rules.append(line.strip())

        rules_text = "\n".join(strict_rules[:10])
        prompt = (
            f"<|im_start|>system\n"
            f"You are a strict technical assistant. Use ONLY these rules:\n"
            f"{rules_text}\n<|im_end|>\n"
            f"<|im_start|>user\n{query}<|im_end|>\n"
            f"<|im_start|>assistant\nTo implement this correctly:\n1."
        )
        inputs  = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        stop_ids = [self.tokenizer.eos_token_id]
        if hasattr(self.tokenizer, "im_end_id"):
            stop_ids.append(self.tokenizer.convert_tokens_to_ids("<|im_end|>"))

        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=max_tokens,
                repetition_penalty=1.0, temperature=0.1, do_sample=True,
                eos_token_id=stop_ids, pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = self.tokenizer.decode(
            out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        )
        final = generated.split("user")[0].split("<|im_start|>")[0].strip()
        return LLMResponse(
            content="1. " + final,
            raw={"mode": "verbalization", "confidence": guidance.confidence},
        )

    # ── Modo 2: Prefix injection (Gap 2) ─────────────────────────────────────

    async def _generate_prefix(
        self, query: str, guidance: RetrievalGuidance, max_tokens: int
    ) -> LLMResponse:
        prompt  = f"<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"
        inputs  = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        proj_ok = self.projector.is_trained

        with torch.no_grad():
            embeds = self.model.get_input_embeddings()(inputs.input_ids)
            combined = inject_skills_to_embeds(
                input_embeds=embeds,
                skill_vectors=guidance.skill_vectors,
                projector=self.projector,
                device=self.device,
                skill_alphas=guidance.skill_alphas if proj_ok else None,
            )
            mask = create_injected_attention_mask(
                inputs.attention_mask, len(guidance.skill_vectors), self.device,
            )
            out = self.model.generate(
                inputs_embeds=combined, attention_mask=mask,
                max_new_tokens=max_tokens, repetition_penalty=1.3,
                no_repeat_ngram_size=3, temperature=0.2, do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return LLMResponse(
            content=self.tokenizer.decode(out[0], skip_special_tokens=True),
            raw={"mode": "prefix", "projector_trained": proj_ok,
                 "n_skills": len(guidance.skill_vectors)},
        )

    # ── Modo 3: Cross-attention injection (Gap 3) ─────────────────────────────

    async def _generate_cross_attention(
        self, query: str, guidance: RetrievalGuidance, max_tokens: int
    ) -> LLMResponse:
        """
        Eq 6: K_graph e V_graph injetados nas últimas N camadas via forward hooks.

        Fluxo:
          1. set_skill_context() pré-computa K_graph, V_graph escalados por alpha_p
          2. model.generate() — hooks interceptam output de cada camada alvo:
             hidden += tanh(gate) * Attn(Q, K_graph, V_graph)
          3. clear_context() — limpa para não vazar para próxima geração
        """
        inj = self.cross_attn_injector

        inj.set_skill_context(
            skill_vectors=guidance.skill_vectors,
            skill_alphas=guidance.skill_alphas or None,
            device=self.device,
            dtype=self.dtype,
        )

        prompt = f"<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        log.info(
            "local_llm.cross_attn_generate",
            n_skills=len(guidance.skill_vectors),
            alphas=[f"{a:.3f}" for a in (guidance.skill_alphas or [])],
            gates=[f"{float(g):.4f}" for g in inj.gates.data],
            is_trained=inj.is_trained,
        )

        try:
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=800,
                    repetition_penalty=1.0,

                    temperature=0.4,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
        finally:
            inj.clear_context()

        generated = self.tokenizer.decode(
            out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        )
        return LLMResponse(
            content=generated,
            raw={
                "mode": "cross_attention",
                "n_skills": len(guidance.skill_vectors),
                "alphas": guidance.skill_alphas,
                "gates": [float(g) for g in inj.gates.data],
                "is_trained": inj.is_trained,
            },
        )

    # ── Genérico e embed ──────────────────────────────────────────────────────

    async def generate(self, messages: list[LLMMessage], **kwargs) -> LLMResponse:
        prompt = self.tokenizer.apply_chat_template(
            [{"role": m.role, "content": m.content} for m in messages],
            tokenize=False, add_generation_prompt=True,
        )
        inputs  = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        max_tok = kwargs.get("max_tokens", 512)

        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=max_tok,
                repetition_penalty=1.2, temperature=0.7,
                do_sample=True, pad_token_id=self.tokenizer.eos_token_id,
            )
        return LLMResponse(
            content=self.tokenizer.decode(
                out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
            )
        )

    async def embed(self, text: str) -> list[float]:
        from sentence_transformers import SentenceTransformer
        if not hasattr(self, "_embedder"):
            self._embedder = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2", device=self.device
            )
        return self._embedder.encode(text, normalize_embeddings=True).tolist()

    def __del__(self):
        if hasattr(self, "cross_attn_injector") and self.cross_attn_injector:
            try:
                self.cross_attn_injector.remove_hooks()
            except Exception:
                pass