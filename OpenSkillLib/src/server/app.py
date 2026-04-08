"""
OpenSkill FastAPI Server
========================
Ponto de entrada para a API HTTP do OpenSkill.
Gerencia o carregamento de modelos locais e orquestra as rotas.
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import structlog
from pathlib import Path

from openskill import OpenSkillClient, LocalDiskStore
from server.routes import craft, evolve, retrieve

log = structlog.get_logger()


def create_app(skill_dir: str = "./skills_output", local_model: bool = False):
    app = FastAPI(
        title="OpenSkill API",
        description="Geometric Skill Memory Engine - TurboQuant + S-Path-RAG",
        version="0.1.0"
    )

    # Configuração de CORS (Essencial para o Frontend SaaS)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Inicializa o Cliente OpenSkill como Singleton no estado da aplicação
    # Se local_model=True, ele carrega o Qwen na VRAM no startup
    store = LocalDiskStore(skill_dir)
    app.state.client = OpenSkillClient(store=store, embed=True)

    if local_model:
        from openskill.injection.local_llm import LocalSkillInjectedLLM
        log.info("server.loading_local_llm", model="Qwen/Qwen3.5-2B")
        app.state.client.llm = LocalSkillInjectedLLM(model_id="Qwen/Qwen3.5-2B")

    # Inclusão das rotas
    app.include_router(craft.router, prefix="/api/v1/craft", tags=["MemCollab"])
    app.include_router(evolve.router, prefix="/api/v1/evolve", tags=["Trace2Skill"])
    app.include_router(retrieve.router, prefix="/api/v1/retrieve", tags=["S-Path-RAG"])

    @app.get("/health")
    async def health():
        return {"status": "healthy", "engine": "openskill"}

    return app