"""
LLM Provider Abstraction — Plugin Pattern
==========================================
Qualquer provedor de LLM (OpenRouter, OpenAI, Anthropic, Ollama, vLLM)
implementa esta interface.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator


@dataclass
class LLMMessage:
    role: str       # "system" | "user" | "assistant"
    content: str


@dataclass
class LLMResponse:
    content: str
    reasoning: str = ""       # Thinking/reasoning tag (se disponível)
    raw: dict = None         # Resposta crua da API


class BaseLLMProvider(ABC):
    """Interface que TODO provedor de LLM deve implementar."""

    @abstractmethod
    async def generate(
        self,
        messages: list[LLMMessage],
        max_tokens: int = 2000,
        temperature: float = 0.7,
        **kwargs,
    ) -> LLMResponse:
        """
        Gera uma resposta de texto.

        Args:
            messages: Lista de mensagens no formato {role, content}
            max_tokens: Limite de tokens na resposta
            temperature: Temperatura de amostragem
        Returns:
            LLMResponse com conteúdo e metadata
        """
        ...

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Gera embedding semântico (para busca vetorial)."""
        ...

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Identificador do modelo usado."""
        ...

    async def close(self) -> None:
        """Cleanup de recursos (opcional)."""
        pass
