"""
train_scorer.py — Script Completo de Treino do Path Scorer
===========================================================
Substitui o comando `openskill train-bootstrap` com um pipeline
mais robusto: gera dados → treina → avalia → salva.

CORREÇÃO: Auto-detecta a dimensão dos embeddings das skills.
          Não requer mais --embed-dim manual.

Uso:
    # Treino completo (dimensão auto-detectada)
    python train_scorer.py --skill-dir ./skills_output

    # Reusar dataset já gerado (mais rápido em iterações)
    python train_scorer.py --skill-dir ./skills_output --data-cache train_data.npz

    # Só gerar o dataset sem treinar
    python train_scorer.py --skill-dir ./skills_output --only-data

    # Forçar dimensão específica (se auto-detecção falhar)
    python train_scorer.py --skill-dir ./skills_output --embed-dim 1536

    # Customizar epochs e learning rate
    python train_scorer.py --skill-dir ./skills_output --epochs 100 --lr 0.0003

Saída:
    ./skills_output/path_scorer.safetensors   ← scorer pronto para uso
    ./train_data.npz                          ← dataset em cache

O scorer é carregado automaticamente pelo SkillGraph na próxima
chamada a find_paths() — sem nenhuma mudança de código necessária.
"""

from __future__ import annotations

import asyncio
import argparse
import sys
from pathlib import Path

import structlog

log = structlog.get_logger()


def _detect_dim_from_skills(skill_dir: str) -> int:
    """
    Detecta a dimensão dos embeddings inspecionando os meta.json das skills.
    Retorna a dimensão mais comum (ex: 1536 para OpenAI, 384 para MiniLM local).
    """
    import json
    skills_path = Path(skill_dir) / "skills"
    dim_counts: dict[int, int] = {}

    for meta_file in skills_path.rglob("meta.json"):
        try:
            data = json.loads(meta_file.read_text(encoding="utf-8"))
            vectors = data.get("vectors", {})
            for profile in vectors.values():
                d = profile.get("dimension", 0)
                emb = profile.get("embedding")
                prov = profile.get("provider", "")
                if d > 0 and emb and prov != "OpenSkillGNN":
                    dim_counts[d] = dim_counts.get(d, 0) + 1
            # Fallback: campo embedding raiz
            root_emb = data.get("embedding")
            if root_emb and isinstance(root_emb, list) and len(root_emb) > 0:
                d = len(root_emb)
                dim_counts[d] = dim_counts.get(d, 0) + 1
        except Exception:
            continue

    if not dim_counts:
        return 0  # sinaliza "não encontrado"

    detected = max(dim_counts, key=lambda d: dim_counts[d])
    print(f"      Dimensões encontradas: {dict(sorted(dim_counts.items()))}")
    print(f"      Usando: {detected}d ({dim_counts[detected]} skills)")
    return detected


async def run_training(args):
    """Pipeline completo: dados → treino → avaliação."""

    print("\n" + "=" * 60)
    print("  OpenSkill Path Scorer — Bootstrap Training")
    print("=" * 60)

    # ── 0. Detecta dimensão ───────────────────────────────────────

    embed_dim = args.embed_dim
    if embed_dim == 0:
        print(f"\n[0/3] Auto-detectando dimensão dos embeddings em '{args.skill_dir}'...")
        embed_dim = _detect_dim_from_skills(args.skill_dir)
        if embed_dim == 0:
            print("\n  ERRO: Nenhum embedding encontrado nas skills.")
            print("  As skills precisam ter embeddings para o treino do scorer.")
            print("\n  Execute um dos comandos abaixo para gerar embeddings:")
            print("    openskill embed <skill-id> --api-key <sua-chave>   (OpenAI 1536d)")
            print("    openskill embed <skill-id> --local                 (MiniLM 384d)")
            sys.exit(1)
        print(f"      ✓ Dimensão detectada: {embed_dim}d")
    else:
        print(f"\n[0/3] Usando dimensão especificada: {embed_dim}d")

    # ── 1. Dataset ────────────────────────────────────────────────

    cache = Path(args.data_cache)

    if cache.exists() and not args.regen_data:
        print(f"\n[1/3] Carregando dataset do cache: {cache}")
        from bootstrap_data import load_dataset
        train_data, val_data = load_dataset(str(cache))
        # Verifica se a dimensão do cache bate com a detectada
        if train_data:
            cached_dim = train_data[0][0].shape[0]
            if cached_dim != embed_dim:
                print(f"\n  AVISO: Cache tem dim={cached_dim}d mas skills têm dim={embed_dim}d.")
                print("  Regenerando dataset...")
                cache.unlink()
                train_data, val_data = None, None

    if not cache.exists() or args.regen_data:
        print(f"\n[1/3] Gerando dataset sintético de '{args.skill_dir}'...")
        print("      (isso pode levar 1-2 min para baixar o embedder na primeira vez)")
        from bootstrap_data import generate_bootstrap_dataset, save_dataset
        train_data, val_data = await generate_bootstrap_dataset(
            skill_dir=args.skill_dir,
            embed_dim=embed_dim,
            val_split=0.15,
            hard_neg_ratio=0.4,
            seed=args.seed,
        )
        save_dataset(train_data, val_data, str(cache))
        print(f"      Dataset salvo em: {cache}")

    print(f"\n      Train: {len(train_data)} amostras")
    print(f"      Val:   {len(val_data)} amostras")
    pos = sum(1 for s in train_data if s[2])
    print(f"      Positivos: {pos} ({pos/max(len(train_data),1)*100:.1f}%)")
    print(f"      Negativos: {len(train_data)-pos} ({(len(train_data)-pos)/max(len(train_data),1)*100:.1f}%)")
    print(f"      Dimensão:  {embed_dim}d")

    if args.only_data:
        print("\nModo --only-data: dataset gerado, pulando treino.")
        return

    # ── 2. Treino ─────────────────────────────────────────────────

    save_path = str(Path(args.skill_dir) / "path_scorer.safetensors")

    print(f"\n[2/3] Treinando PathScorerModel por {args.epochs} epochs...")
    print(f"      LR: {args.lr}  |  Batch: 32  |  InfoNCE + BCE verifier")
    print(f"      Gumbel temperature: 1.0 → 0.1 (annealed)")
    print(f"      Embed dim: {embed_dim}d")
    print()

    from openskill.core.trainer import train_path_scorer, evaluate_scorer

    metrics = await train_path_scorer(
        train_data=train_data,
        embed_dim=embed_dim,
        save_path=save_path,
        epochs=args.epochs,
        lr=args.lr,
    )

    print(f"\n      Best loss: {metrics['best_loss']:.4f}")
    print(f"      Modelo salvo em: {save_path}")

    # ── 3. Avaliação ──────────────────────────────────────────────

    print(f"\n[3/3] Avaliando no conjunto de validação ({len(val_data)} amostras)...")

    if len(val_data) >= 4:
        eval_metrics = evaluate_scorer(save_path, val_data, embed_dim=embed_dim)
        print(f"\n      Verifier Accuracy: {eval_metrics['verifier_accuracy']:.2%}")
        print(f"      MRR:               {eval_metrics['mrr']:.4f}")
        print(f"      Amostras testadas: {eval_metrics['n_test']}")

        if eval_metrics["verifier_accuracy"] < 0.6:
            print("\n  AVISO: accuracy abaixo de 60%. Considere:")
            print("         - Mais skills diversas (recomendado: ≥5 domínios diferentes)")
            print("         - Mais epochs: --epochs 150")
            print("         - Verificar se os embeddings foram gerados corretamente")
    else:
        print("      (poucos dados de validação — avaliação pulada)")

    # ── Sumário final ─────────────────────────────────────────────

    print("\n" + "=" * 60)
    print("  TREINO CONCLUÍDO")
    print("=" * 60)
    print(f"\n  Scorer salvo em: {save_path}")
    print(f"  Dimensão: {embed_dim}d")
    print("\n  Próximos passos:")
    print("    1. Execute uma query: openskill retrieve --query 'sua query'")
    print("    2. O SkillGraph vai carregar o scorer automaticamente")
    print("    3. Observe 'graph.neural_scorer_loaded' no log")
    print("    4. A confidence deve subir de 0.00 para > 0.35")
    print("\n  Para re-treinar com mais dados:")
    print(f"    python train_scorer.py --skill-dir {args.skill_dir} --regen-data --epochs {args.epochs}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Treina o PathScorerModel (S-Path-RAG) a partir das skills existentes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--skill-dir",
        default="./skills_output",
        help="Pasta com as skills (default: ./skills_output)",
    )
    parser.add_argument(
        "--data-cache",
        default="train_data.npz",
        help="Arquivo .npz para cache do dataset (default: train_data.npz)",
    )
    parser.add_argument(
        "--embed-dim",
        type=int,
        default=0,
        help="Dimensão dos embeddings (0=auto-detectar, default: 0)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=80,
        help="Número de epochs de treino (default: 80)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate inicial (default: 3e-4)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed para reprodutibilidade (default: 42)",
    )
    parser.add_argument(
        "--regen-data",
        action="store_true",
        help="Regenera o dataset mesmo se o cache existir",
    )
    parser.add_argument(
        "--only-data",
        action="store_true",
        help="Apenas gera o dataset, sem treinar",
    )

    args = parser.parse_args()

    if not Path(args.skill_dir).exists():
        print(f"\nERRO: Pasta '{args.skill_dir}' não encontrada.")
        print("Crie skills primeiro com: openskill create --task 'sua tarefa'")
        sys.exit(1)

    asyncio.run(run_training(args))


if __name__ == "__main__":
    main()