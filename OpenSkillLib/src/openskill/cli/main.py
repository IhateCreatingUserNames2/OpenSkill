"""
openskill CLI — The Swiss Army Knife for Geometric Skill Memory
==============================================================
Comandos:
    create   - MemCollab: Cria nova skill a partir de tarefa.
    retrieve - S-Path-RAG: Busca inteligente (texto ou injeção vetorial).
    evolve   - Trace2Skill: Melhora skill com base em trajetórias.
    list     - Lista biblioteca local.
    graph    - Inspeciona a topologia do cérebro de skills.
    embed    - Força a re-quantização TurboQuant de uma skill.
    serve    - Sobe o servidor MCP/FastAPI para o Cursor IDE.
"""

from __future__ import annotations

import asyncio
import sys
import json
import os
from pathlib import Path
from typing import Optional

import click
import structlog
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown

from openskill import OpenSkillClient, LocalDiskStore
from openskill.llm.openrouter import OpenRouterProvider
from openskill.llm.ollama import OllamaProvider
from openskill.core.vector import unpack_qvector

# Logger e Console para UI rica no terminal
log = structlog.get_logger()
console = Console()

def get_client(skill_dir: str, api_key: Optional[str], local: bool, model_id: Optional[str] = None) -> OpenSkillClient:
    """
    Factory de cliente.
    Se 'local' for True, tenta carregar o motor de injeção vetorial local.
    """
    store = LocalDiskStore(skill_dir)

    if local:
        # Se for local, usamos o provedor que suporta injeção vetorial (LocalSkillInjectedLLM)
        from openskill.injection.local_llm import LocalSkillInjectedLLM
        # Default para o Qwen3.5 que o usuário baixou
        mid = model_id or "Qwen/Qwen3.5-2B"
        console.print(f"[bold yellow]Loading Local Engine ([/bold yellow]{mid}[bold yellow])...[/bold yellow]")
        llm = LocalSkillInjectedLLM(model_id=mid)
    elif api_key:
        llm = OpenRouterProvider(api_key=api_key)
    else:
        key = os.getenv("OPENROUTER_API_KEY", "")
        llm = OpenRouterProvider(api_key=key) if key else None

    return OpenSkillClient(store=store, llm=llm)

@click.group()
@click.version_option()
def cli():
    """OpenSkill — Unified lifecycle for agent skills."""
    pass

# ── COMANDO: CREATE (MemCollab) ──────────────────────────────────────────────

@cli.command()
@click.argument("task")
@click.option("--skill-dir", default="./skills_output", help="Pasta das skills")
@click.option("--api-key", help="OpenRouter API Key")
@click.option("--weak", default="openai/gpt-4o-mini", help="Modelo Agente Fraco")
@click.option("--strong", default="anthropic/claude-3-5-sonnet", help="Modelo Agente Forte")
def create(task: str, skill_dir: str, api_key: str, weak: str, strong: str):
    """Cria uma nova skill usando análise contrastiva MemCollab."""
    client = get_client(skill_dir, api_key, local=False)

    async def _run():
        with console.status("[bold green]Distilling skill via MemCollab..."):
            meta = await client.craft(task=task, weak_model=weak, strong_model=strong)

        console.print(Panel(
            f"[bold str]ID:[/bold str] {meta.id}\n"
            f"[bold str]Title:[/bold str] {meta.title}\n"
            f"[bold str]Category:[/bold str] {meta.category}/{meta.subcategory}",
            title="Skill Created Successfully", border_style="green"
        ))

    asyncio.run(_run())

# ── COMANDO: RETRIEVE (S-Path-RAG + TurboQuant) ──────────────────────────────

@cli.command()
@click.argument("query")
@click.option("--skill-dir", default="./skills_output")
@click.option("--local", is_flag=True, help="Usa Injeção Vetorial Local")
@click.option("--use-gnn", is_flag=True, help="Força a busca pelos vetores enriquecidos da GNN")
# --- CORREÇÃO: Adicionado 'cross_attention' e 'prefix' às escolhas de modo ---
@click.option("--mode", default="auto",
              type=click.Choice(["injection", "prefix", "cross_attention", "verbalization", "auto"]),
              help="Modo de geração")
@click.option("--model-id", help="Model ID para execução local")
@click.option("--top-k", default=3)
def retrieve(query: str, skill_dir: str, local: bool, use_gnn: bool, mode: str, model_id: str, top_k: int):
    """Busca skills e gera resposta (TurboQuant + S-Path-RAG)."""

    client = get_client(skill_dir, None, local=local, model_id=model_id)

    async def _run():
        with console.status("[bold blue]Navigating Skill Graph (Neural-Socratic)..."):
            # 1. Recupera a orientação (Guidance)
            guidance = await client.retriever.retrieve(query, top_k=top_k, use_gnn=use_gnn)

        # Mostra o rastro do grafo
        console.print(f"\n[bold]Socratic Trace:[/bold]\n[dim]{guidance.reasoning_trace}[/dim]\n")

        # Se estiver em modo local, gera com a orientação recuperada
        if local and hasattr(client.llm, 'generate_with_guidance'):
            with console.status(f"[bold magenta]Generating answer (mode={mode})..."):
                resp = await client.llm.generate_with_guidance(query, guidance, mode=mode)

            # Ajuste de exibição no painel
            conf = guidance.confidence if hasattr(guidance, 'confidence') else 0.0
            panel_title = f"Generated Answer | mode={mode} | conf={conf:.2f}"
            console.print(Panel(resp.content, title=panel_title, border_style="magenta"))
        else:
            # Apenas mostra as skills encontradas (Modo sem geração local)
            for i, content in enumerate(guidance.skill_contents):
                console.print(f"[bold cyan]Skill {i + 1}:[/bold cyan]")
                console.print(Markdown(content))
                console.print("-" * 40)

    asyncio.run(_run())


@cli.command()
@click.option("--skill-dir", default="./skills_output")
@click.option("--api-key", help="OpenRouter API key para gerar dados")
@click.option("--epochs", default=50, help="Número de épocas de treino")
def train_bootstrap(skill_dir: str, api_key: str, epochs: int):
    """
    Fase 2: Usa o LLM Forte para gerar dados e treina o Path Scorer local (384d).
    """
    from openskill.core.trainer import train_path_scorer
    from sentence_transformers import SentenceTransformer
    import numpy as np

    # Usamos o OpenRouter apenas para gerar o TEXTO das perguntas
    client = get_client(skill_dir, api_key, local=False)

    async def _run():
        console.print("[bold blue]Starting Bootstrap Training (Fase 2)...[/bold blue]")

        # Carregamos o embutidor de 384d explicitamente para alinhar os dados
        console.print("Loading local embedder for training alignment (384d)...")
        local_embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        all_metas = await client.store.list_skills()
        if len(all_metas) < 2:
            console.print("[red]Erro: Você precisa de pelo menos 2 skills para o treino contrastivo.[/red]")
            return

        train_data = []

        for meta in all_metas:
            console.print(f" Generating samples for: [cyan]{meta.title}[/cyan]")

            # 1. Gera texto via LLM Forte
            prompt = f"""Generate 10 diverse user queries that would be solved by this technical skill. 
            Format: Just the queries, one per line.
            SKILL TITLE: {meta.title}
            SKILL CONTENT: {meta.task}"""

            from openskill.llm.base import LLMMessage
            resp = await client.llm.generate([LLMMessage(role="user", content=prompt)])
            queries = [q.strip() for q in resp.content.split("\n") if q.strip() and len(q) > 10]

            # 2. Pega o vetor local da skill (384d)
            skill_vec = None
            vectors_dict = getattr(meta, 'vectors', {})
            for p in vectors_dict.values():
                if getattr(p, 'dimension', 0) == 384:
                    skill_vec = np.array(p.embedding)
                    break

            if skill_vec is None:
                console.print(f"[yellow]Aviso: Skill {meta.id} não tem perfil de 384d. Pulando...[/yellow]")
                continue

            # 3. Cria pares Positivos e Negativos
            for q_text in queries:
                # EMBEDDING DA QUERY (Forçado em 384d para bater com a skill local)
                q_vec = local_embedder.encode(q_text, normalize_embeddings=True)

                # EXEMPLO POSITIVO
                train_data.append((q_vec, skill_vec, True))

                # EXEMPLO NEGATIVO (Pega uma skill aleatória que não seja esta)
                other_metas = [m for m in all_metas if m.id != meta.id]
                if other_metas:
                    other_meta = np.random.choice(other_metas)
                    other_vec = None
                    for p in other_meta.vectors.values():
                        if getattr(p, 'dimension', 0) == 384:
                            other_vec = np.array(p.embedding)
                            break

                    if other_vec is not None:
                        train_data.append((q_vec, other_vec, False))

        if not train_data:
            console.print("[red]Erro: Nenhum dado de treino gerado.[/red]")
            return

        # 4. DISPARA O TREINAMENTO
        save_path = str(client.store.workspace_path / "path_scorer.safetensors")
        console.print(f"[bold yellow]Training neural scorer on {len(train_data)} samples...[/bold yellow]")

        await train_path_scorer(
            train_data=train_data,
            embed_dim=384,  # Agora garantido!
            save_path=save_path,
            epochs=epochs
        )

        console.print(f"[bold green]✓ Phase 2 Complete! Scorer saved to {save_path}[/bold green]")

    asyncio.run(_run())


@cli.command()
@click.option("--skill-dir", default="./skills_output")
@click.option("--use-gnn", is_flag=True, help="Ativa o refinamento neural GNN")
def build_graph(skill_dir: str, use_gnn: bool):
    """Varre todas as skills e reconstrói o grafo (com opção de GNN)."""
    from openskill.core.graph import register_skill_in_graph
    client = get_client(skill_dir, None, local=False)

    async def _run():
        console.print(f"[bold blue]Rebuilding Skill Graph {'(Neural GNN Mode)' if use_gnn else ''}...")

        all_metas = await client.store.list_skills()
        metas_dict = {m.id: m.to_dict() for m in all_metas}

        # Limpa o grafo atual para evitar duplicidade
        from openskill.storage.base import SkillGraphData
        await client.store.update_graph(SkillGraphData())

        for meta in all_metas:
            # Passamos o use_gnn para a função de registro
            await register_skill_in_graph(
                meta.id, meta, metas_dict, client.store, use_gnn=use_gnn
            )
            console.print(f" Registered: {meta.title}")

        console.print("[bold green]✓ Graph and Encodings Rebuilt Successfully.")

    asyncio.run(_run())

# ---- CONVERT

@cli.command()
@click.argument("skill-id")
@click.option("--skill-dir", default="./skills_output", help="Pasta das skills")  # Adicionado aqui
@click.option("--local", is_flag=True, help="Converter para 384d (Local)")
@click.option("--api-key", help="Converter para 1536d (Remoto)")
def convert(skill_id: str, local: bool, api_key: str, skill_dir: str):  # skill_dir adicionado na assinatura
    """Gera uma nova 'View' geométrica para uma skill existente."""
    # Agora passamos o skill_dir que vem da opção (ou o default)
    sid = skill_id.strip()
    client = get_client(skill_dir, api_key, local=local)

    async def _run():
        # DEBUG: Vamos imprimir onde ele está procurando
        # console.print(f"[dim]Checking path: {client.store.skills_dir}[/dim]")

        meta = await client.store.get_skill_meta(skill_id)
        content = await client.store.get_skill_md(skill_id)

        if not meta or not content:
            console.print(f"[red]Erro: Skill {skill_id} não encontrada em {client.store.skills_dir}[/red]")
            return

        target = "LOCAL (384d)" if local else "REMOTE (1536d)"
        with console.status(f"[bold yellow]Converting skill {skill_id} to {target}..."):
            await client._embed_and_register(skill_id, content, meta)

        console.print(f"[green]✓ Skill {skill_id} convertida para {target}.[/green]")

    asyncio.run(_run())

# ── COMANDO: EVOLVE (Trace2Skill) ────────────────────────────────────────────

@cli.command()
@click.argument("skill-id")
@click.option("--skill-dir", default="./skills_output")
@click.option("--tasks", help="Tarefas separadas por vírgula para teste")
def evolve(skill_id: str, skill_dir: str, tasks: str):
    """Evolui uma skill baseada em evidência de erro (Trace2Skill)."""
    client = get_client(skill_dir, None, local=False)

    async def _run():
        task_list = [t.strip() for t in tasks.split(",")] if tasks else []
        with console.status(f"[bold yellow]Evolving skill {skill_id} in parallel fleet..."):
            result = await client.evolve(skill_id=skill_id, tasks=task_list)

        console.print(f"[bold green]Success![/bold green] Applied {result['patch_count']} patches.")
        console.print(f"Fleet Success Rate: {result['success_rate']*100:.1f}%")

    asyncio.run(_run())

# ── COMANDO: LIST ────────────────────────────────────────────────────────────

@cli.command()
@click.option("--skill-dir", default="./skills_output")
def list(skill_dir: str):
    """Lista todas as skills e o status da memória geométrica."""
    client = get_client(skill_dir, None, local=False)

    async def _run():
        metas = await client.list_skills()
        table = Table(title="OpenSkill Library")
        table.add_column("ID", style="dim")
        table.add_column("Title", style="cyan")
        table.add_column("Category")
        table.add_column("TurboQuant", justify="center")
        table.add_column("Evolutions", justify="center")

        for m in metas:
            status = "[green]✓[/green]" if m.qvector else "[red]✗[/red]"
            table.add_row(m.id, m.title, f"{m.category}/{m.subcategory}", status, str(m.evolution_count))

        console.print(table)

    asyncio.run(_run())

# ── COMANDO: GRAPH ───────────────────────────────────────────────────────────

@cli.command()
@click.option("--skill-dir", default="./skills_output")
def graph(skill_dir: str):
    """Visualiza a topologia do grafo de conhecimento."""
    store = LocalDiskStore(skill_dir)
    g = store.get_graph()

    console.print(f"[bold]Nodes:[/bold] {len(g.nodes)} | [bold]Edges:[/bold] {len(g.edges)}")

    table = Table(show_header=False, box=None)
    for e in g.edges:
        table.add_row(f"[cyan]{e['from']}[/cyan]", f"──([italic]{e['type']}[/italic])──>", f"[cyan]{e['to']}[/cyan]")

    console.print(table)

# ── COMANDO: EMBED (TurboQuant Fix) ──────────────────────────────────────────

@cli.command()
@click.argument("skill-id")
@click.option("--skill-dir", default="./skills_output")
@click.option("--api-key", help="OpenRouter API key")  # Adicionado
@click.option("--local", is_flag=True, help="Usa motor local")  # Adicionado
def embed(skill_id: str, skill_dir: str, api_key: str, local: bool):
    """Re-calcula o vetor TurboQuant para uma skill específica."""
    # Factory agora recebe os parâmetros corretamente
    client = get_client(skill_dir, api_key, local=local)

    async def _run():
        if client.llm is None:
            console.print("[red]Erro: Provedor LLM não configurado. Use --api-key ou --local.[/red]")
            return

        with console.status("[bold blue]Recalculating TurboQuant geometry..."):
            meta = await client.store.get_skill_meta(skill_id)
            content = await client.store.get_skill_md(skill_id)

            if not meta or not content:
                console.print(f"[red]Skill {skill_id} não encontrada no disco.[/red]")
                return

            # Chama o helper que ajustamos no client.py
            await client._embed_and_register(skill_id, content, meta)

        console.print(f"[green]✓ Skill {skill_id} quantizada e registrada no grafo.[/green]")

    asyncio.run(_run())

# ── COMANDO: SERVE (FastAPI/MCP) ──────────────────────────────────────────────

@cli.command()
@click.option("--port", default=8000)
@click.option("--skill-dir", default="./skills_output")
def serve(port: int, skill_dir: str):
    """Inicia o servidor API e o endpoint MCP para o Cursor IDE."""
    import uvicorn
    # Importação tardia para não pesar a CLI
    from openskill.mcp.server import main as run_mcp

    console.print(f"[bold green]Starting OpenSkill Server on port {port}...[/bold green]")
    console.print(f"MCP Endpoint active for Cursor/Windsurf.")

    # Aqui você pode escolher rodar o MCP server ou o FastAPI app
    # Para o Cursor, o MCP via stdio é o padrão.
    asyncio.run(run_mcp())

def main():
    cli()

if __name__ == "__main__":
    main()