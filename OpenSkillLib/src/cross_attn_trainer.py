"""
cross_attn_trainer.py — Treino do CrossAttentionInjector (Gap 3 via RepE)
==========================================================================
Treina os gates e projetores K/V do CrossAttentionInjector usando
Engenharia de Representação (RepE).

Em vez de forçar o modelo a gerar texto (o que degrada a gramática via
Repetition Degeneration), treinamos o injetor para alinhar o estado
cognitivo do LLM ao de um especialista via MSE no espaço latente.

Paradigma "Brain & Soul":
  Cérebro (Trace2Skill + Verbalization): regras explícitas no System Prompt
  Alma (CrossAttn + RepE): altera o estado latente para o domínio da skill

Loss total:
  L_total = L_repe + lambda_align * L_align + L_reg
  L_repe  = MSE(R_injetado[-1], R_persona[-1])   — alinha estado cognitivo
  L_align = (alpha_p - attnMass_p)²              — alinha atenção com scorer
  L_reg   = gates.pow(2).mean() * 1e-4           — suave, sem alvo fixo

Correções aplicadas em relação à versão anterior:
  [BUG 1 — CRASH]   Índice de camada usava hidden_size=2048 como base.
                     Corrigido para num_hidden_layers (ex: 28).
  [BUG 2 — GRAD]    loss_align era float Python (0.0) quando batch vazio,
                     saía do grafo computacional. Agora é tensor com grad.
  [BUG 3 — DESIGN]  Gate regularizer forçava 0.20 arbitrário.
                     Substituído por L2 suave nos gates.
  [BUG 4 — GRAD]    F.normalize eliminava gradiente de magnitude de k_proj.
                     k_proj só aprendia direção, não intensidade. Removido.
                     Escala suave via norma do embedding table do LLM.
  [EXTRA — DTYPE]   mse_loss entre bfloat16 e float32 pode silenciar gradientes.
                     Ambos os tensores castados para float32 antes do MSE.
"""

from __future__ import annotations

import asyncio
import argparse
import sys
import random
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import structlog

log = structlog.get_logger()

CROSS_ATTN_SAVE_NAME = "cross_attn_injector.safetensors"
SCORER_SAVE_NAME     = "path_scorer.safetensors"
BATCH_SIZE           = 4
LAMBDA_ALIGN         = 1.0


# ── Carregamento ──────────────────────────────────────────────────────────────

def load_frozen_llm(model_id: str, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"  Carregando LLM ({model_id}) em modo eager (congelado)...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    cfg = model.config
    print(f"  LLM pronto: hidden={cfg.hidden_size}, layers={cfg.num_hidden_layers}")
    return model, tokenizer


def get_decoder_layers(model):
    """Navega até as decoder layers para Qwen2/Qwen3."""
    for attr in ["model", "transformer"]:
        backbone = getattr(model, attr, None)
        if backbone is not None:
            layers = getattr(backbone, "layers", None)
            if layers is not None:
                return layers
    return None


def resolve_target_layers(model, n_layers: int) -> list[int]:
    """
    BUG 1 CORRIGIDO: base = num_hidden_layers, não hidden_size.

    Antes (errado): range(2048 - 7, 2048) → [2041..2047] → IndexError
    Depois (certo): range(28 - 7, 28)     → [21..27]     → OK
    """
    total  = model.config.num_hidden_layers
    start  = max(0, total - n_layers)
    result = list(range(start, total))
    log.info("cross_attn.target_layers", total=total, injecting=result)
    return result


# ── Forward com injeção + captura attnMass ────────────────────────────────────

def forward_with_injection_and_capture(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    injector,
    skill_alphas: list[float],
    target_layer_indices: list[int],
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Forward pass com hooks de injeção ativos.
    Instala hooks de captura adicionais para medir attnMass.

    Retorna:
      R_injected : hidden_states[-1] [B, T, H] — com gradiente via K_graph
      l_align    : (alpha_p - attnMass_p)² — com gradiente
      attn_norm  : attnMass detached — para logging
    """
    captured: list[torch.Tensor] = []

    def make_capture_hook():
        def hook(module, inp, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if injector._K_graph is None:
                return output

            B, T, H = hidden.shape
            K_g = injector._K_graph.to(hidden.device, hidden.dtype)
            N, nH, hD = K_g.shape[1], injector.num_heads, injector.head_dim

            Q = hidden.reshape(B, T, nH, hD).transpose(1, 2)
            K = K_g.expand(B, -1, -1).reshape(B, N, nH, hD).transpose(1, 2)
            attn_w = torch.softmax(
                Q @ K.transpose(-2, -1) * (hD ** -0.5), dim=-1
            )  # [B, nH, T, N]
            captured.append(attn_w.mean(dim=(0, 1, 2)))   # [N]
            return output
        return hook

    layers  = get_decoder_layers(model)
    handles = []
    if layers is not None:
        for abs_idx in target_layer_indices:
            handles.append(layers[abs_idx].register_forward_hook(make_capture_hook()))

    with torch.set_grad_enabled(True):
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

    R_full = outputs.hidden_states[-1]   # [B, T, H]
    for h in handles:
        h.remove()

    # BUG 2 CORRIGIDO: retorna tensor com grad quando sem captures
    if not captured:
        zero = torch.tensor(0.0, requires_grad=True, device=device)
        return R_full, zero, torch.zeros(len(skill_alphas), device=device)

    attn_mass = torch.stack(captured, dim=0).mean(dim=0)   # [N]
    attn_norm = attn_mass / (attn_mass.sum() + 1e-8)

    N_a     = min(len(skill_alphas), attn_norm.shape[0])
    alpha_t = torch.tensor(skill_alphas[:N_a], dtype=torch.float32, device=device)
    l_align = ((alpha_t - attn_norm[:N_a]) ** 2).mean()

    return R_full, l_align, attn_norm.detach()


# ── Loop de treino ────────────────────────────────────────────────────────────

async def train_cross_attn(
    skill_dir: str,
    model_id: str,
    data_cache: str,
    epochs: int = 20,
    lr: float = 1e-3,
    lambda_align: float = LAMBDA_ALIGN,
):
    device     = "cuda" if torch.cuda.is_available() else "cpu"
    skill_path = Path(skill_dir)

    # 1. Scorer
    scorer_path = skill_path / SCORER_SAVE_NAME
    if not scorer_path.exists():
        print(f"ERRO: {scorer_path} não encontrado.")
        print("Execute: python train_scorer.py --skill-dir", skill_dir)
        sys.exit(1)

    from safetensors.torch import load_file as st_load
    from openskill.core.trainer import PathScorerModel
    from openskill.injection.cross_attention import CrossAttentionInjector, detect_qwen_params

    scorer = PathScorerModel(embed_dim=384).to(device)
    scorer.load_state_dict(st_load(str(scorer_path)))
    scorer.eval()
    print(f"  Scorer: {scorer_path}")

    # 2. LLM
    model, tokenizer = load_frozen_llm(model_id, device)
    params = detect_qwen_params(model)

    # BUG 1 CORRIGIDO aqui
    target_abs_idx = resolve_target_layers(model, params["n_layers"])

    # 3. Injector
    inj_path = skill_path / CROSS_ATTN_SAVE_NAME
    if inj_path.exists():
        inj = CrossAttentionInjector.load(
            inj_path, embed_dim=384,
            hidden_size=params["hidden_size"],
            n_layers=params["n_layers"],
            num_heads=params["num_heads"],
            device=device,
        )
        print(f"  Injector carregado: {inj_path}")
    else:
        inj = CrossAttentionInjector(
            embed_dim=384,
            hidden_size=params["hidden_size"],
            n_layers=params["n_layers"],
            num_heads=params["num_heads"],
        )
        print(f"  Injector novo: {params['n_layers']} camadas × {params['hidden_size']}d")

    inj = inj.to(device).to(torch.float32)
    inj.install(model)
    inj.train()

    # 4. Dataset
    cache = Path(data_cache)
    if not cache.exists():
        print(f"ERRO: {cache} não encontrado.")
        print("Execute: python train_scorer.py --skill-dir", skill_dir, "--only-data")
        sys.exit(1)

    from bootstrap_data import load_dataset
    train_data, _ = load_dataset(str(cache))
    positives = [(q, p) for q, p, is_pos in train_data if is_pos]
    if not positives:
        print("ERRO: sem amostras positivas.")
        sys.exit(1)
    print(f"  {len(positives)} amostras positivas")

    from openskill.storage.local import LocalDiskStore
    store     = LocalDiskStore(skill_dir)
    all_metas = await store.list_skills()
    skill_index: list[tuple[str, np.ndarray]] = []
    for m in all_metas:
        title   = getattr(m, 'title', '') or ''
        vectors = getattr(m, 'vectors', {})
        for p in vectors.values():
            if getattr(p, 'dimension', 0) == 384 and p.embedding:
                skill_index.append((title, np.array(p.embedding, dtype=np.float32)))
                break

    skill_vecs   = [v for _, v in skill_index]
    skill_titles = [t for t, _ in skill_index]
    print(f"  {len(skill_vecs)} skills com vetores 384d")

    # 5. Optimizer
    # BUG 3 CORRIGIDO: weight_decay=1e-4 já regulariza; gates recebem L2 suave
    # adicional dentro da loss, sem forçar valor fixo
    optimizer = torch.optim.AdamW(inj.parameters(), lr=lr, weight_decay=1e-4)
    steps_per_epoch = math.ceil(len(positives) / BATCH_SIZE)
    total_steps     = epochs * steps_per_epoch
    warmup_steps    = max(1, total_steps // 10)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=total_steps,
        pct_start=warmup_steps / total_steps,
    )

    print(f"\n  Treinando (RepE Brain & Soul) por {epochs} epochs...")
    print(f"  LR: {lr}  |  Lambda_align: {lambda_align}  |  Batch: {BATCH_SIZE}")
    print()

    best_loss   = float("inf")
    model_dtype = next(model.parameters()).dtype

    # Referência de escala: norma do embedding table (estável, representa espaço do LLM)
    with torch.no_grad():
        emb_norm = model.get_input_embeddings().weight.norm(dim=-1).mean().item()
    log.info("cross_attn.emb_norm_ref", emb_norm=f"{emb_norm:.2f}")

    for epoch in range(epochs):
        random.shuffle(positives)
        epoch_loss = 0.0
        n_batches  = 0

        for start in range(0, len(positives), BATCH_SIZE):
            batch = positives[start:start + BATCH_SIZE]
            optimizer.zero_grad()

            batch_repe  = []
            batch_align = []

            for q_vec, s_vec in batch:

                # Path: positiva + 1 negativa
                neg_cands = [
                    i for i in range(len(skill_vecs))
                    if not np.allclose(skill_vecs[i], s_vec, atol=1e-4)
                ]
                path_vecs = (
                    [s_vec, skill_vecs[random.choice(neg_cands)]]
                    if neg_cands else [s_vec]
                )

                # Alpha via scorer
                with torch.no_grad():
                    q_t = torch.tensor(q_vec, dtype=torch.float32, device=device)
                    us, vs = [], []
                    for pv in path_vecs:
                        p_t = torch.tensor(pv, dtype=torch.float32, device=device)
                        u, v = scorer(q_t.unsqueeze(0), p_t.unsqueeze(0))
                        us.append(u.squeeze()); vs.append(v.squeeze())
                    u_t   = torch.stack(us); v_t = torch.stack(vs)
                    w     = torch.softmax(u_t, dim=0)
                    alpha = (w * v_t)
                    alpha = (alpha / (alpha.sum() + 1e-8)).tolist()

                # Query text
                sims = [
                    float(np.dot(q_vec, sv) /
                          (np.linalg.norm(q_vec) * np.linalg.norm(sv) + 1e-8))
                    for sv in skill_vecs
                ]
                best_i     = int(np.argmax(sims))
                skill_name = skill_titles[best_i] or "Software Engineering"
                query_text = f"How to solve this: {skill_name}?"

                neutral_prompt = (
                    f"<|im_start|>user\n{query_text}<|im_end|>\n"
                    f"<|im_start|>assistant\n"
                )
                persona_prompt = (
                    f"<|im_start|>system\nYou are a Senior Expert in {skill_name}.\n"
                    f"<|im_end|>\n<|im_start|>user\n{query_text}<|im_end|>\n"
                    f"<|im_start|>assistant\n"
                )

                enc_neutral = tokenizer(neutral_prompt, return_tensors="pt").to(device)
                enc_persona = tokenizer(persona_prompt, return_tensors="pt").to(device)

                # PASSO A: R_target — LLM com persona, sem injeção
                inj.clear_context()
                with torch.no_grad():
                    out_t = model(
                        input_ids=enc_persona.input_ids,
                        attention_mask=enc_persona.attention_mask,
                        output_hidden_states=True,
                        use_cache=False,
                    )
                    # EXTRA CORRIGIDO: float32 evita dtype mismatch no MSE
                    R_target = out_t.hidden_states[-1][:, -1, :].float().detach()

                # PASSO B: K_graph e V_graph
                skills_t = torch.tensor(
                    np.array(path_vecs), dtype=torch.float32, device=device
                )

                # BUG 4 CORRIGIDO: sem F.normalize — k_proj aprende direção E amplitude
                # Escala suave baseada na norma do embedding table (referência estável)
                K = inj.k_proj(skills_t).to(model_dtype)  # [N, H] — grad preservado
                V = inj.v_proj(skills_t).to(model_dtype)

                K_scale = K.norm(dim=-1).mean().clamp(min=1e-6)
                V_scale = V.norm(dim=-1).mean().clamp(min=1e-6)
                K = K * (emb_norm / K_scale)
                V = V * (emb_norm / V_scale)

                if alpha:
                    a_t = torch.tensor(alpha, dtype=model_dtype, device=device).unsqueeze(-1)
                    K = K * a_t
                    V = V * a_t

                inj._K_graph = K.unsqueeze(0)
                inj._V_graph = V.unsqueeze(0)

                # PASSO C: forward injetado
                R_full, l_align, attn_mass = forward_with_injection_and_capture(
                    model=model,
                    input_ids=enc_neutral.input_ids,
                    attention_mask=enc_neutral.attention_mask,
                    injector=inj,
                    skill_alphas=alpha,
                    target_layer_indices=target_abs_idx,
                    device=device,
                )

                # PASSO D: L_repe = MSE no espaço latente
                # EXTRA CORRIGIDO: float32 em ambos
                R_injected = R_full[:, -1, :].float()
                l_repe     = F.mse_loss(R_injected, R_target)

                batch_repe.append(l_repe)
                if l_align.requires_grad:
                    batch_align.append(l_align)

                log.debug(
                    "cross_attn.sample",
                    l_repe=f"{float(l_repe):.4f}",
                    l_align=f"{float(l_align):.4f}",
                    attn=[f"{float(m):.3f}" for m in attn_mass.tolist()],
                    gates=[f"{float(g):.4f}" for g in inj.gates.data],
                )

            inj.clear_context()
            if not batch_repe:
                continue

            loss_repe  = torch.stack(batch_repe).mean()

            # BUG 2 CORRIGIDO: tensor com requires_grad quando lista vazia
            loss_align = (
                torch.stack(batch_align).mean()
                if batch_align
                else torch.tensor(0.0, requires_grad=True, device=device)
            )

            # BUG 3 CORRIGIDO: L2 suave, sem alvo fixo de 0.20
            l_reg = inj.gates.pow(2).mean() * 1e-4

            total = loss_repe + lambda_align * loss_align + l_reg
            total.backward()
            torch.nn.utils.clip_grad_norm_(inj.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += total.item()
            n_batches  += 1

        if n_batches == 0:
            continue

        avg = epoch_loss / n_batches
        if avg < best_loss:
            best_loss = avg
            inj.save(str(inj_path))

        if epoch % 5 == 0 or epoch == epochs - 1:
            log.info(
                "cross_attn.epoch",
                epoch=epoch,
                loss=f"{avg:.4f}",
                best=f"{best_loss:.4f}",
                gates=[f"{float(g):.4f}" for g in inj.gates.data],
                lr=f"{scheduler.get_last_lr()[0]:.6f}",
            )

    print(f"\n  Melhor Loss:  {best_loss:.4f}")
    print(f"  Salvo em:     {inj_path}")
    print(f"  Gates finais: {[f'{float(g):.4f}' for g in inj.gates.data]}")
    print(f"  Treinado:     {inj.is_trained}")

    inj.remove_hooks()
    return {"best_loss": best_loss, "path": str(inj_path)}


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Treina CrossAttentionInjector via RepE (S-Path-RAG Gap 3)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--skill-dir",    default="./skills_output")
    parser.add_argument("--model-id",     default="Qwen/Qwen3.5-2B")
    parser.add_argument("--data-cache",   default="train_data.npz")
    parser.add_argument("--epochs",       type=int,   default=20)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--lambda-align", type=float, default=LAMBDA_ALIGN)
    args = parser.parse_args()

    if not Path(args.skill_dir).exists():
        print(f"ERRO: '{args.skill_dir}' não encontrada.")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("  CrossAttentionInjector — RepE Brain & Soul")
    print("=" * 60)

    asyncio.run(train_cross_attn(
        skill_dir    = args.skill_dir,
        model_id     = args.model_id,
        data_cache   = args.data_cache,
        epochs       = args.epochs,
        lr           = args.lr,
        lambda_align = args.lambda_align,
    ))

    print("\n  Próximos passos:")
    print("  1. O local_llm.py carrega o injector automaticamente ao iniciar")
    print("  2. Use --mode auto (verbalization + cross-attention juntos)")
    print("  3. openskill retrieve --local --mode auto --query 'sua query'")