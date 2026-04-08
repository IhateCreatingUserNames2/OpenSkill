"""
MCP Server — Model Context Protocol para IDEs
==============================================
Expoõe o OpenSkill como ferramenta MCP para:
  - Cursor IDE
  - Windsurf (Codeium)
  - Claude Desktop (Anthropic)
  - Qualquer cliente MCP-compatible

Uso no Cursor:
  1. Adicionar em ~/.cursor/mcp.json:
     {
       "mcpServers": {
         "openskill": {
           "command": "python",
           "args": ["-m", "openskill.mcp.server"]
         }
       }
     }

  2. No Cursor, perguntar: "Use the Raft consensus skill to help me
     handle a network partition"
"""

from __future__ import annotations

import asyncio
import json
import structlog
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# MCP SDK — usa a spec oficial da Anthropic
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent
    from mcp.server.notification import NotificationOptions
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

from openskill import OpenSkillClient
from openskill.storage.local import LocalDiskStore

log = structlog.get_logger()


@dataclass
class MCPServerConfig:
    """Configuração do servidor MCP."""
    skill_dir: str = "./skills_output"
    default_weak_model: str = "openai/gpt-4o-mini"
    default_strong_model: str = "anthropic/claude-3-5-sonnet"
    api_key: Optional[str] = None


@dataclass
class MCPToolContext:
    """Contexto compartilhado entre tools MCP."""
    client: OpenSkillClient
    config: MCPServerConfig


# ── Ferramentas disponíveis via MCP ──────────────────────────────────────────

TOOLS: list[dict] = [
    {
        "name": "openskill_craft",
        "description": (
            "Cria uma nova skill de raciocínio a partir de um problema. "
            "Usa análise contrastiva entre um modelo fraco e um forte para "
            "extrair invariantes, padrões de violação e constraints normativas. "
            "Retorna uma skill estruturada que pode ser usada para guiar "
            "futuras gerações."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "Descrição do problema ou tarefa",
                    "example": "Implement Raft consensus over a high-latency network"
                },
                "weak_model": {
                    "type": "string",
                    "description": "Modelo mais fraco (fallback)",
                    "default": "openai/gpt-4o-mini"
                },
                "strong_model": {
                    "type": "string",
                    "description": "Modelo mais forte (target)",
                    "default": "anthropic/claude-3-5-sonnet"
                },
            },
            "required": ["task"],
        },
    },
    {
        "name": "openskill_retrieve",
        "description": (
            "Busca skills relevantes para resolver um problema usando "
            "busca vetorial TurboQuant + grafo semântico S-Path-RAG. "
            "Retorna constraints normativas e padrões de raciocínio que "
            "devem ser aplicados ao resolver o problema."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A pergunta ou problema do usuário",
                    "example": "How to handle leader election failure in a distributed system?"
                },
                "top_k": {
                    "type": "integer",
                    "description": "Número de skills a retornar",
                    "default": 3,
                },
                "format": {
                    "type": "string",
                    "enum": ["constraints", "full", "summary"],
                    "description": "Formato do retorno",
                    "default": "constraints",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "openskill_evolve",
        "description": (
            "Evolui uma skill existente a partir de trajetórias de execução. "
            "Usa Trace2Skill com frota de sub-agentes para propor patches "
            "que são consolidados hierarquicamente."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "skill_id": {
                    "type": "string",
                    "description": "ID da skill a evoluir",
                },
                "trajectories": {
                    "type": "array",
                    "description": "Lista de trajetórias [{task, trajectory, success}]",
                    "items": {"type": "object"},
                },
                "tasks": {
                    "type": "array",
                    "description": "Ou: lista de tarefas para gerar trajetórias automaticamente",
                    "items": {"type": "string"},
                },
            },
            "required": ["skill_id"],
        },
    },
    {
        "name": "openskill_list",
        "description": "Lista todas as skills disponíveis no repositório local.",
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "openskill_graph",
        "description": "Retorna o grafo de relationships entre skills.",
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
]


# ── Handler das ferramentas ───────────────────────────────────────────────────

async def handle_tool_call(
    ctx: MCPToolContext,
    tool_name: str,
    arguments: dict,
) -> TextContent:
    """Dispatch de chamada de ferramenta MCP."""

    try:
        if tool_name == "openskill_craft":
            meta = await ctx.client.craft(
                task=arguments["task"],
                weak_model=arguments.get("weak_model"),
                strong_model=arguments.get("strong_model"),
            )
            return TextContent(
                type="text",
                text=json.dumps({
                    "status": "created",
                    "skill_id": meta.id,
                    "title": meta.title,
                    "category": f"{meta.category}/{meta.subcategory}",
                    "message": (
                        f"Skill '{meta.title}' criada com sucesso. "
                        f"ID: {meta.id}. Use openskill_retrieve para buscar guidance."
                    ),
                }, indent=2),
            )

        elif tool_name == "openskill_retrieve":
            result = await ctx.client.retrieve(
                query=arguments["query"],
                top_k=arguments.get("top_k", 3),
            )

            fmt = arguments.get("format", "constraints")
            skills = result.get("skills", [])

            if fmt == "summary":
                output = "\n".join(
                    f"• [{s.get('title', '?')}] — {s.get('domain', '')}"
                    for s in skills
                )
            elif fmt == "full":
                output = json.dumps(result, indent=2, default=str)
            else:  # constraints
                lines = []
                for i, s in enumerate(skills, 1):
                    md = s.get("content", "")
                    # Extrai só as constraints do markdown
                    lines.append(f"## Skill {i}: {s.get('title', '?')}")
                    if "## Normative Constraints" in md:
                        start = md.index("## Normative Constraints")
                        end = md.index("\n##", start + 1) if "\n##" in md[start+1:] else len(md)
                        lines.append(md[start:end])
                    lines.append("")
                output = "\n".join(lines)

            return TextContent(
                type="text",
                text=output or "Nenhuma skill encontrada para esta query.",
            )

        elif tool_name == "openskill_evolve":
            result = await ctx.client.evolve(
                skill_id=arguments["skill_id"],
                trajectories=arguments.get("trajectories"),
                tasks=arguments.get("tasks"),
            )
            return TextContent(
                type="text",
                text=json.dumps({
                    "status": "evolved",
                    "patch_count": result["patch_count"],
                    "fleet_size": result["fleet_size"],
                    "success_rate": f"{result['success_rate']:.1%}",
                }, indent=2),
            )

        elif tool_name == "openskill_list":
            metas = await ctx.client.list_skills()
            return TextContent(
                type="text",
                text=json.dumps([
                    {
                        "id": m.id,
                        "title": m.title,
                        "category": f"{m.category}/{m.subcategory}",
                        "domain": m.domain,
                        "evolution_count": m.evolution_count,
                        "created_at": m.created_at,
                    }
                    for m in metas
                ], indent=2, default=str),
            )

        elif tool_name == "openskill_graph":
            graph = ctx.client.store.get_graph()
            return TextContent(
                type="text",
                text=json.dumps(graph.to_dict(), indent=2, default=str),
            )

        else:
            return TextContent(
                type="text",
                text=f"Unknown tool: {tool_name}",
            )

    except Exception as e:
        log.error("mcp_tool.error", tool=tool_name, error=str(e))
        return TextContent(
            type="text",
            text=f"Error executing {tool_name}: {e}",
        )


# ── Entry point ───────────────────────────────────────────────────────────────

async def main(config: MCPServerConfig | None = None) -> None:
    """Roda o servidor MCP via stdio."""

    if not MCP_AVAILABLE:
        print(
            "ERROR: mcp package not installed.\n"
            "Install with: pip install openskill[mcp]",
            file=sys.stderr,
        )
        return

    import sys

    cfg = config or MCPServerConfig()

    # Inicializa cliente OpenSkill
    from openskill.llm.openrouter import OpenRouterProvider
    llm = OpenRouterProvider(api_key=cfg.api_key or "") if cfg.api_key else None
    store = LocalDiskStore(cfg.skill_dir)
    client = OpenSkillClient(store=store, llm=llm)

    ctx = MCPToolContext(client=client, config=cfg)
    server = Server("openskill")

    # ── Registra capabilities ────────────────────────────────────────────────

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name=t["name"],
                description=t["description"],
                inputSchema=t["input_schema"],
            )
            for t in TOOLS
        ]

    @server.call_tool()
    async def call_tool(
        name: str,
        arguments: dict[str, Any],
    ) -> list[TextContent]:
        result = await handle_tool_call(ctx, name, arguments)
        return [result]

    # ── Roda servidor stdio ──────────────────────────────────────────────────
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(
                notification_options=NotificationOptions(),
            ),
        )


def run() -> None:
    """Entry point para: python -m openskill.mcp.server"""
    import sys
    asyncio.run(main())


if __name__ == "__main__":
    run()
