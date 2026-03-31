"""
local_llm.py — Local Inference & Soft Latent Injection Engine
=============================================================
Replaces OpenRouter with local HuggingFace models.
Implements the S-Path-RAG Soft Latent Injection.
"""
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

# ── Configuração de Modelos ──────────────────────────────────────────────────
# Usamos o modelo de Chat (sem -Base) para suportar o apply_chat_template
LLM_MODEL_ID = "Qwen/Qwen3.5-0.8B"  # Modelo de Chat/Instruct
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Carregando Modelo de Embedding: {EMBED_MODEL_ID}...")
embedder = SentenceTransformer(EMBED_MODEL_ID, device=device)
EMBED_DIM = embedder.get_sentence_embedding_dimension()

print(f"Carregando LLM: {LLM_MODEL_ID}...")
# Para modelos Qwen muito recentes, trust_remote_code=True é essencial
tokenizer = AutoTokenizer.from_pretrained(
    LLM_MODEL_ID,
    trust_remote_code=True
)

llm_model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True # Adicionado aqui!
)
LLM_HIDDEN_SIZE = llm_model.config.hidden_size


# ── Camada de Projeção (S-Path-RAG Alignment) ────────────────────────────────
# O paper S-Path-RAG projeta o vetor da Skill para a dimensão interna do LLM.
# Idealmente, esta camada deve ser treinada (Alignment Loss). Para inferência
# zero-shot, inicializamos uma camada linear.
class SkillProjector(nn.Module):
    def __init__(self, embed_dim, llm_dim):
        super().__init__()
        self.proj = nn.Linear(embed_dim, llm_dim, dtype=torch.float16)

    def forward(self, x):
        return self.proj(x)


projector = SkillProjector(EMBED_DIM, LLM_HIDDEN_SIZE).to(device)


# ── Funções Core ─────────────────────────────────────────────────────────────

def get_embedding(text: str) -> np.ndarray:
    """Gera um embedding real usando o SentenceTransformer."""
    with torch.no_grad():
        vec = embedder.encode(text, normalize_embeddings=True)
    return vec


def generate_text(prompt: str, max_tokens: int = 1500) -> str:
    """Geração padrão de texto (usado para o Trace2Skill e MemCollab)."""
    messages = [{"role": "user", "content": prompt}]
    text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text_input, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = llm_model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            do_sample=True
        )

    # Decodifica apenas os novos tokens gerados
    generated = outputs[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def generate_with_soft_latents(prompt: str, skill_vectors: list[np.ndarray], max_tokens: int = 1500) -> str:
    """
    S-Path-RAG Soft Latent Injection:
    Anexa vetores quantizados de-quantizados DIRETAMENTE nos tensores do LLM.
    """
    messages = [{"role": "user", "content": prompt}]
    text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text_input, return_tensors="pt").to(device)

    # 1. Pega os embeddings de token normais do prompt do usuário
    with torch.no_grad():
        prompt_embeds = llm_model.get_input_embeddings()(inputs.input_ids)

    # 2. Prepara os Soft Latents das Skills
    if skill_vectors:
        # Converte lista de numpy arrays para tensor [Num_Skills, Embed_Dim]
        skills_tensor = torch.tensor(skill_vectors, dtype=torch.float16, device=device)

        # 3. Projeta da dimensão de Embedding (ex: 384) para a dimensão do LLM (ex: 4096)
        with torch.no_grad():
            projected_skills = projector(skills_tensor).unsqueeze(0)  # [1, Num_Skills, LLM_Dim]

        # 4. Injeção Vetorial (Concatenação no início)
        combined_embeds = torch.cat([projected_skills, prompt_embeds], dim=1)

        # 5. Ajusta a máscara de atenção para o novo tamanho
        batch_size = inputs.attention_mask.shape[0]
        num_skills = projected_skills.shape[1]
        skill_mask = torch.ones((batch_size, num_skills), dtype=inputs.attention_mask.dtype, device=device)
        combined_mask = torch.cat([skill_mask, inputs.attention_mask], dim=1)
    else:
        combined_embeds = prompt_embeds
        combined_mask = inputs.attention_mask

    # 6. Gera a resposta condicionada geometricamente
    with torch.no_grad():
        outputs = llm_model.generate(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            max_new_tokens=max_tokens,
            temperature=0.7,
            do_sample=True
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)