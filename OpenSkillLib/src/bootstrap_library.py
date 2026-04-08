import subprocess
import re
import os
import sys

# --- CONFIGURAÇÃO ---
MY_OPENROUTER_KEY = "sk-or-v1-"

TASKS = [
    "Implement a Connection Pool for PostgreSQL in Python using psycopg2",
    "Design a Circuit Breaker pattern for microservices to handle cascading failures",
    "Secure a REST API using JWT (JSON Web Tokens) with asymmetric RS256 keys",
    "Optimize SQL queries using Composite Indexes and Explain Analyze",
    "Implement an Exponential Backoff retry strategy for failed HTTP requests",
    "Configure Kubernetes Liveness and Readiness probes for a high-availability service",
    "Manage distributed transactions using the Saga Pattern in a microservices architecture",
    "Implement Blue-Green deployment strategy using Nginx as a load balancer",
    "Detect and prevent Memory Leaks in long-running Python asyncio processes",
    "Implement Role-Based Access Control (RBAC) in a FastAPI application"
]

def run_command(command):
    print(f"\n> Executando: {command}")
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            env=os.environ
        )

        if result.returncode != 0:
            print(f"AVISO: O comando retornou erro (code {result.returncode})")
            print(f"Stderr: {result.stderr}")

        return result.stdout if result.stdout else ""
    except Exception as e:
        print(f"ERRO CRÍTICO ao executar comando: {e}")
        return ""

def main():
    if not MY_OPENROUTER_KEY:
        print("ERRO: Configure a chave OPENROUTER_API_KEY no script!")
        return

    os.environ["OPENROUTER_API_KEY"] = MY_OPENROUTER_KEY

    skill_ids = []
    print("=== Iniciando Criação da Biblioteca OpenSkill (UTF-8 Mode) ===")

    for i, task in enumerate(TASKS):
        print(f"\n" + "=" * 40)
        print(f"--- Criando Skill {i + 1}/{len(TASKS)} ---")
        print(f"Tarefa: {task}")

        # 1. openskill create
        cmd_create = f'openskill create "{task}" --api-key {MY_OPENROUTER_KEY}'
        output = run_command(cmd_create)

        # 2. Extrair o ID
        match = re.search(r"ID:.*?([a-f0-9]{8})", output, re.IGNORECASE)

        if match:
            skill_id = match.group(1)
            skill_ids.append(skill_id)
            print(f"SUCESSO: ID {skill_id} criado.")

            # 3. Gerar Embedding Local (CORRIGIDO: Removido --skill-id)
            print(f"Gerando embedding local (384d)...")
            run_command(f"openskill embed {skill_id} --local")
        else:
            print("AVISO: Não consegui capturar o ID automaticamente.")

    # 4. Refinamento GNN
    print("\n" + "=" * 50)
    print("ETAPA FINAL: Refinando biblioteca com GNN...")
    print("=" * 50)
    run_command("openskill build-graph --use-gnn")

    print("\n=== PROCESSO CONCLUÍDO ===")
    print(f"Skills processadas: {len(skill_ids)}")

if __name__ == "__main__":
    main()