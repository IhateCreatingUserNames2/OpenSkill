"""
projector_trainer.py — Treino do SkillProjector com L_align (Gap 2)
====================================================================
Treina o SkillProjector para que o LLM realmente atenda aos vetores injetados.

Eq 7: attnMass(p) = (1/T_tok) * Σ_t Σ_{k ∈ idx(p)} A_{t,k}
Eq 8: L_align = (1/|P_sel|) * Σ_p (alpha_p - attnMass_p)²

Pipeline:
  1. Carrega o scorer já treinado (path_scorer.safetensors)
  2. Carrega o LLM (Qwen) — congelado, sem gradiente
  3. Carrega as skills + embeddings do store
  4. Para cada query do dataset:
     a. Scorer calcula alpha_p por skill (Eq 5)
     b. Projector injeta as skills como prefix tokens
     c. Forward pass com output_attentions=True
     d. L_align = (alpha_p - attnMass_p)²
     e. Backward apenas no projector (LLM congelado)
  5. Salva projector_weights.safetensors

Uso:
    python projector_trainer.py --skill-dir ./skills_output --model-id Qwen/Qwen2.5-0.5B

    # Usar dataset existente
    python projector_trainer.py --skill-dir ./skills_output --data-cache train_data.npz

    # Ajustar epochs e LR
    python projector_trainer.py --skill-dir ./skills_output --epochs 30 --lr 5e-4
"""

from __future__ import annotations

import asyncio
import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import structlog

log = structlog.get_logger()

TrainSample = Tuple[np.ndarray, np.ndarray, bool]

PROJECTOR_SAVE_NAME  = "projector_weights.safetensors"
SCORER_SAVE_NAME     = "path_scorer.safetensors"
DEFAULT_MODEL_ID     = "Qwen/Qwen3.5-2B"
BATCH_SIZE           = 4    # pequeno — cada sample exige um forward pass do LLM
LAMBDA_ALIGN         = 0.5  # peso do L_align na loss total


# ── Carregamento do LLM congelado ─────────────────────────────────────────────

def load_frozen_llm(model_id: str, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    print(f"  Carregando LLM ({model_id}) — congelado...")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # Deixe o transformers escolher o dtype ideal (geralmente BF16 para Qwen)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )

    for param in model.parameters():
        param.requires_grad_(False)

    model.eval()
    return model, tokenizer


# ── Coleta de alpha_p via scorer ──────────────────────────────────────────────

def compute_alphas_for_sample(
    scorer,
    query_vec: np.ndarray,
    skill_vecs: list[np.ndarray],
    device: str,
) -> list[float]:
    """
    Roda o scorer para cada skill e retorna alpha_p = softmax(u_p) * v_eta_p,
    normalizado (Eq 4 × Eq 5).
    """
    if not skill_vecs:
        return []

    q_t = torch.tensor(query_vec, dtype=torch.float32, device=device)
    us, vetas = [], []

    with torch.no_grad():
        for sv in skill_vecs:
            q_in = q_t.unsqueeze(0)
            p_in = torch.tensor(sv, dtype=torch.float32, device=device).unsqueeze(0)
            u, v = scorer(q_in, p_in)
            us.append(u.squeeze())
            vetas.append(v.squeeze())

    u_t     = torch.stack(us)                    # [N]
    v_t     = torch.stack(vetas)                 # [N]
    w_tilde = torch.softmax(u_t, dim=0)          # [N]
    alpha   = w_tilde * v_t                      # [N]
    alpha   = alpha / (alpha.sum() + 1e-8)       # normaliza

    return alpha.tolist()


# ── Forward pass para L_align ─────────────────────────────────────────────────

def compute_lalign_for_batch(
    model,
    tokenizer,
    projector,
    scorer,
    queries: list[str],
    skill_vecs_per_query: list[list[np.ndarray]],
    alphas_per_query: list[list[float]],
    device: str,
) -> torch.Tensor:
    """
    Calcula L_align para um batch de queries.

    Para cada query:
      1. Tokeniza o prompt
      2. Projeta as skills → prefix tokens (com gradiente no projector)
      3. Forward pass com output_attentions=True
      4. Extrai attnMass para cada skill
      5. L_align = mean((alpha_p - attnMass_p)²)

    Retorna a média de L_align sobre o batch.
    """
    from openskill.injection.soft import capture_attn_mass, create_injected_attention_mask

    batch_align_losses = []

    for query, skill_vecs, alphas in zip(queries, skill_vecs_per_query, alphas_per_query):
        if not skill_vecs or not alphas:
            continue

        N = len(skill_vecs)

        # Tokeniza o prompt (sem input_ids no generate — usamos embeds)
        prompt = f"<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"
        enc = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            # Embedding dos tokens do prompt [1, T, D_llm]
            token_embeds = model.get_input_embeddings()(enc.input_ids)

        # Projeta skills → [N, D_skill] → [N, D_llm]  (COM gradiente)
        skills_t = torch.tensor(
            np.array(skill_vecs),
            dtype=torch.float32,  # Mantemos a entrada em float32 para o projector
            device=device,
        )
        projected = projector(skills_t)  # Aqui o output é Float32

        # --- ADICIONE ESTA LINHA ABAIXO ---
        projected = projected.to(model.dtype)
        # ---------------------------------

        # Escala de energia
        with torch.no_grad():
            token_embeds = model.get_input_embeddings()(enc.input_ids)
            prompt_norm = token_embeds.norm(p=2, dim=-1).mean()

        proj_norm = projected.norm(p=2, dim=-1).mean()
        projected_scaled = projected * (prompt_norm / (proj_norm + 1e-8))

        # Agora o concat funcionará pois ambos são model.dtype (BFloat16)
        combined = torch.cat(
            [projected_scaled.unsqueeze(0), token_embeds], dim=1
        )

        # Máscara de atenção expandida [1, N + T]
        combined_mask = create_injected_attention_mask(
            enc.attention_mask, N, device
        )

        # Captura attnMass — l_align tem gradiente via combined (que depende do projector)
        l_align, attn_mass = capture_attn_mass(
            model=model,
            input_embeds=combined,
            attention_mask=combined_mask,
            n_skill_tokens=N,
            skill_alphas=alphas,
            device=device,
        )

        if l_align.requires_grad:
            batch_align_losses.append(l_align)

        log.debug(
            "projector.lalign_sample",
            query=query[:40],
            alphas=[f"{a:.3f}" for a in alphas],
            attn_mass=[f"{float(m):.3f}" for m in attn_mass.tolist()],
            l_align=f"{float(l_align):.4f}",
        )

    if not batch_align_losses:
        return torch.tensor(0.0, requires_grad=True, device=device)

    return torch.stack(batch_align_losses).mean()


# ── Loop de treino do projector ───────────────────────────────────────────────

async def train_projector(
    skill_dir: str,
    model_id: str,
    data_cache: str,
    epochs: int = 30,
    lr: float = 5e-4,
    lambda_align: float = LAMBDA_ALIGN,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    skill_path = Path(skill_dir)

    # ── 1. Verifica scorer ────────────────────────────────────────────────────
    scorer_path = skill_path / SCORER_SAVE_NAME
    if not scorer_path.exists():
        print(f"\nERRO: {scorer_path} não encontrado.")
        print("Execute primeiro: python train_scorer.py --skill-dir", skill_dir)
        sys.exit(1)

    from safetensors.torch import load_file as st_load
    from openskill.core.trainer import PathScorerModel
    from openskill.injection.soft import SkillProjector

    scorer = PathScorerModel(embed_dim=384).to(device)
    scorer.load_state_dict(st_load(str(scorer_path)))
    scorer.eval()
    print(f"  Scorer carregado: {scorer_path}")

    # ── 2. Carrega LLM congelado ──────────────────────────────────────────────
    model, tokenizer = load_frozen_llm(model_id, device)
    hidden_size = model.config.hidden_size

    # ── 3. Inicializa projector ───────────────────────────────────────────────
    projector_path = skill_path / PROJECTOR_SAVE_NAME
    if projector_path.exists():
        projector = SkillProjector.load(
            projector_path, embed_dim=384, llm_hidden_size=hidden_size, device=device
        )
        projector = projector.to(torch.float32)
        print(f"  Projector existente carregado: {projector_path}")
    else:
        projector = SkillProjector(embed_dim=384, llm_hidden_size=hidden_size)
        # Inicializa com pequena perturbação em torno de zero
        # (NÃO eye_ — eye_ só funciona para matrizes quadradas e é sub-ótimo aqui)
        torch.nn.init.xavier_uniform_(projector.proj.weight)
        torch.nn.init.zeros_(projector.proj.bias)
        projector = projector.to(device).to(torch.float32)
        print(f"  Projector novo inicializado (Xavier): {hidden_size}d")

    projector.train()

    # ── 4. Carrega dataset ────────────────────────────────────────────────────
    cache = Path(data_cache)
    if cache.exists():
        from bootstrap_data import load_dataset
        train_data, val_data = load_dataset(str(cache))
    else:
        print(f"\n  Dataset não encontrado em {cache}.")
        print("  Execute primeiro: python train_scorer.py --skill-dir", skill_dir, "--only-data")
        sys.exit(1)

    # Filtra apenas os positivos — só queries que têm resposta certa
    # são úteis para L_align (queremos que o LLM atenda à skill correta)
    positive_samples = [(q, p) for q, p, is_pos in train_data if is_pos]
    print(f"\n  {len(positive_samples)} amostras positivas para treino do projector")

    if len(positive_samples) == 0:
        print("ERRO: Nenhuma amostra positiva no dataset.")
        sys.exit(1)

    # Carrega as skills do store — guarda (title, vec) para queries representativas
    from openskill.storage.local import LocalDiskStore
    store = LocalDiskStore(skill_dir)
    all_metas = await store.list_skills()

    skill_index: list[tuple[str, np.ndarray]] = []
    for m in all_metas:
        title = getattr(m, 'title', '') or ''
        vectors = getattr(m, 'vectors', {})
        for p in vectors.values():
            if getattr(p, 'dimension', 0) == 384 and p.embedding:
                skill_index.append((title, np.array(p.embedding, dtype=np.float32)))
                break

    if not skill_index:
        print("ERRO: Nenhum skill com embedding 384d encontrado.")
        print("Execute: openskill embed --local --skill-id <id>")
        sys.exit(1)

    skill_vecs_store = [v for _, v in skill_index]
    skill_titles     = [t for t, _ in skill_index]
    print(f"  {len(skill_vecs_store)} skills com vetores 384d disponíveis")

    # ── 5. Optimizer ─────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(projector.parameters(), lr=lr, weight_decay=1e-4)

    # LR scheduler: cosine com warmup
    import math
    steps_per_epoch = math.ceil(len(positive_samples) / BATCH_SIZE)
    total_steps = epochs * steps_per_epoch
    warmup_steps = max(1, total_steps // 10)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        total_steps=total_steps,
        pct_start=warmup_steps / total_steps,
    )

    # ── 6. Loop de treino ─────────────────────────────────────────────────────
    print(f"\n  Treinando projector por {epochs} epochs...")
    print(f"  LR: {lr}  |  Lambda_align: {lambda_align}  |  Batch: {BATCH_SIZE}")
    print(f"  Layers de atenção usadas: últimos {int(model.config.num_hidden_layers * 0.5)}")
    print()

    best_loss = float("inf")
    import random

    for epoch in range(epochs):
        random.shuffle(positive_samples)
        epoch_loss = 0.0
        n_batches  = 0

        for start in range(0, len(positive_samples), BATCH_SIZE):
            batch = positive_samples[start:start + BATCH_SIZE]
            optimizer.zero_grad()

            # Usa o título da skill como query representativa em vez de "query_N"
            # Isso ancora o prompt do LLM ao domínio semântico real da skill
            batch_skill_vecs = []
            batch_alphas     = []
            queries          = []

            for q_vec, s_vec in batch:
                # Encontra a skill mais próxima do q_vec para nomear a query
                sims = [float(np.dot(q_vec, sv) /
                              (np.linalg.norm(q_vec) * np.linalg.norm(sv) + 1e-8))
                        for sv in skill_vecs_store]
                best_idx = int(np.argmax(sims))
                query_text = f"How to implement {skill_titles[best_idx]}?" \
                             if skill_titles[best_idx] else "How to solve this technical problem?"
                queries.append(query_text)

                # Com múltiplas skills: inclui a skill correta + 1 negativa aleatória
                # Isso treina o projector na mixture, não só em skills isoladas
                if len(skill_vecs_store) > 1:
                    neg_candidates = [i for i in range(len(skill_vecs_store))
                                      if not np.allclose(skill_vecs_store[i], s_vec, atol=1e-4)]
                    if neg_candidates:
                        neg_idx = random.choice(neg_candidates)
                        path_vecs = [s_vec, skill_vecs_store[neg_idx]]
                    else:
                        path_vecs = [s_vec]
                else:
                    path_vecs = [s_vec]

                alphas = compute_alphas_for_sample(scorer, q_vec, path_vecs, device)
                batch_skill_vecs.append(path_vecs)
                batch_alphas.append(alphas)

            # Calcula L_align para o batch
            l_align = compute_lalign_for_batch(
                model=model,
                tokenizer=tokenizer,
                projector=projector,
                scorer=scorer,
                queries=queries,
                skill_vecs_per_query=batch_skill_vecs,
                alphas_per_query=batch_alphas,
                device=device,
            )

            # Regularização L2 nos pesos do projector (evita divergência)
            l_reg = sum(p.norm() ** 2 for p in projector.parameters()) * 1e-5

            loss = lambda_align * l_align + l_reg
            loss.backward()

            torch.nn.utils.clip_grad_norm_(projector.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches  += 1

        if n_batches == 0:
            continue

        avg_loss = epoch_loss / n_batches

        if avg_loss < best_loss:
            best_loss = avg_loss
            projector.save(str(projector_path))

        if epoch % 5 == 0 or epoch == epochs - 1:
            log.info(
                "projector.epoch",
                epoch=epoch,
                loss=f"{avg_loss:.4f}",
                best=f"{best_loss:.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.6f}",
            )

    # ── 7. Verificação final ──────────────────────────────────────────────────
    print(f"\n  Melhor L_align: {best_loss:.4f}")
    print(f"  Projector salvo em: {projector_path}")

    # Confirma que o projector divergiu de eye_
    proj = SkillProjector.load(
        projector_path, embed_dim=384, llm_hidden_size=hidden_size, device=device
    )
    print(f"  Projector treinado: {proj.is_trained}")

    return {"best_loss": best_loss, "path": str(projector_path)}


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Treina o SkillProjector com L_align (S-Path-RAG Gap 2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--skill-dir",  default="./skills_output")
    parser.add_argument("--model-id",   default=DEFAULT_MODEL_ID,
                        help=f"Modelo Qwen (default: {DEFAULT_MODEL_ID})")
    parser.add_argument("--data-cache", default="train_data.npz")
    parser.add_argument("--regen-data", action="store_true",
                        help="Regenera o dataset mesmo que o cache exista (use ao adicionar novas skills)")
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--lr",         type=float, default=5e-4)
    parser.add_argument("--lambda-align", type=float, default=LAMBDA_ALIGN,
                        help="Peso do L_align na loss (default: 0.5)")
    args = parser.parse_args()

    if not Path(args.skill_dir).exists():
        print(f"ERRO: '{args.skill_dir}' não encontrada.")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("  SkillProjector Training — L_align (S-Path-RAG Eq 7+8)")
    print("=" * 60)

    # Aviso quando há cache antigo e regen não foi pedido
    cache = Path(args.data_cache)
    if cache.exists() and not args.regen_data:
        print(f"\n  AVISO: Usando dataset em cache '{args.data_cache}'.")
        print("  Se você adicionou novas skills, use --regen-data para incluí-las.")
        print()

    asyncio.run(train_projector(
        skill_dir    = args.skill_dir,
        model_id     = args.model_id,
        data_cache   = args.data_cache if not args.regen_data else "__force_regen__",
        epochs       = args.epochs,
        lr           = args.lr,
        lambda_align = args.lambda_align,
    ))

    print("\n  Próximos passos:")
    print("  1. O local_llm.py carrega o projector automaticamente")
    print("  2. Use --mode injection para ativar a injeção treinada")
    print("  3. Execute: openskill retrieve --local --mode injection --query 'sua query'")


if __name__ == "__main__":
    main()