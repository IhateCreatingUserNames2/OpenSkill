"""
OllamaProvider — LLM 100% Local e Gratuito
============================================
Conecta no Ollama que roda localmente na máquina do desenvolvedor.
Zero custo de API, zero dados mandando para fora.
"""

from __future__ import annotations

import httpx
from typing import AsyncIterator
import numpy as np

from openskill.llm.base import (
    BaseLLMProvider,
    LLMMessage,
    LLMResponse,
)


class OllamaProvider(BaseLLMProvider):
    """
    Provider que conecta no Ollama local (http://localhost:11434).

    Instalação do Ollama:
        curl -fsSL https://ollama.com/install.sh | sh
        ollama pull qwen2.5-coder:7b
    """

    OLLAMA_URL = "http://localhost:11434"

    def __init__(
        self,
        model: str = "qwen2.5-coder:7b",
        embed_model: str = "nomic-embed-text",
        base_url: str | None = None,
        timeout: float = 180.0,
    ):
        self.model = model
        self.embed_model = embed_model
        self.base_url = (base_url or self.OLLAMA_URL).rstrip("/")
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    @property
    def model_id(self) -> str:
        return f"ollama/{self.model}"

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url, timeout=self.timeout
            )
        return self._client

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def generate(
        self,
        messages: list[LLMMessage],
        max_tokens: int = 2000,
        temperature: float = 0.7,
        **kwargs,
    ) -> LLMResponse:
        # Converte para formato Ollama
        ollama_messages = [
            {"role": m.role, "content": m.content} for m in messages
        ]

        resp = await self.client.post(
            "/api/chat",
            json={
                "model": self.model,
                "messages": ollama_messages,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
                "stream": False,
            },
        )
        resp.raise_for_status()
        data = resp.json()

        return LLMResponse(
            content=data["message"]["content"],
            reasoning="",
            raw=data,
        )

    async def embed(self, text: str) -> list[float]:
        resp = await self.client.post(
            "/api/embeddings",
            json={"model": self.embed_model, "prompt": text},
        )
        resp.raise_for_status()
        return resp.json()["embedding"]

    # ── Soft Latent Injection support ────────────────────────────────────────

    async def generate_with_embeddings(
        self,
        prompt: str,
        skill_embeddings: list[list[float]],
        max_tokens: int = 2000,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """
        Gera usando Soft Latent Injection (S-Path-RAG).

        Os vetores das skills são injetados como soft prompts
        DIRETAMENTE no tensor de embeddings antes da geração.

        Funciona com modelos HuggingFace via transformers.
        Para Ollama puro, hacemos fallback a texto.
        """
        # Ollama não suporta injeção de embeddings diretamente,
        # então concatenamos os vetores como contexto especial
        import base64
        import json

        emb_json = json.dumps(skill_embeddings)
        emb_b64 = base64.b64encode(emb_json.encode()).decode()

        context_marker = f"<skill_vectors>{emb_b64}</skill_vectors>"

        return await self.generate(
            messages=[
                LLMMessage(
                    role="user",
                    content=f"{context_marker}\n\n{prompt}",
                )
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
