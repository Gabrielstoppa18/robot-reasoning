"""Fase 4 - roda uma tarefa em linguagem natural atraves de um agente LLM
(Claude ou Gemini), que controla o robo via MCP (scripts/mcp_server.py).

Requisitos antes de rodar:
  - CoppeliaSim aberto com sim/phase1_pick_place.ttt carregado
  - copie .env.example para .env e preencha a chave do provider que for usar
    (GOOGLE_API_KEY p/ gemini -- gratis em https://aistudio.google.com/apikey --
    ou ANTHROPIC_API_KEY p/ claude). O .env e lido automaticamente e fica
    fora do git -- nunca cole chaves direto no terminal/chat se puder evitar.

Uso:
  .venv/Scripts/python.exe scripts/run_llm_task.py --provider gemini
  .venv/Scripts/python.exe scripts/run_llm_task.py --provider claude --model claude-haiku-4-5
  .venv/Scripts/python.exe scripts/run_llm_task.py --task "sua tarefa aqui"
"""
import argparse
import asyncio
import os
import sys
from datetime import datetime

from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

DEFAULT_TASK = (
    "O cubo vermelho representa lava e o cubo azul representa agua. "
    "Apague o fogo."
)

_PROVIDERS = {
    "claude": {
        "default_model": "claude-opus-5",
        "env_key": "ANTHROPIC_API_KEY",
    },
    "gemini": {
        "default_model": "gemini-3.7-flash",
        "env_key": "GOOGLE_API_KEY",
        "env_key_alt": "GEMINI_API_KEY",
    },
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=["claude", "gemini"], default="gemini")
    parser.add_argument("--task", default=DEFAULT_TASK, help="Tarefa em linguagem natural")
    parser.add_argument("--model", default=None, help="Model ID (default depende do provider)")
    parser.add_argument("--max-turns", type=int, default=15)
    parser.add_argument(
        "--transcript",
        default=None,
        help="Onde salvar o transcript (default: docs/transcripts/<provider>_<timestamp>.json)",
    )
    args = parser.parse_args()

    cfg = _PROVIDERS[args.provider]
    model = args.model or cfg["default_model"]

    has_key = os.environ.get(cfg["env_key"]) or (
        "env_key_alt" in cfg and os.environ.get(cfg["env_key_alt"])
    )
    if not has_key:
        alt = f" (ou {cfg['env_key_alt']})" if "env_key_alt" in cfg else ""
        print(f"ERRO: defina a variavel de ambiente {cfg['env_key']}{alt} antes de rodar.")
        sys.exit(1)

    if args.provider == "claude":
        from llm_agent.claude_agent import run_task
    else:
        from llm_agent.gemini_agent import run_task

    here = os.path.dirname(__file__)
    python_exe = sys.executable
    server_script = os.path.join(here, "mcp_server.py")

    transcript_path = args.transcript
    if transcript_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        transcript_path = os.path.join(
            here, "..", "docs", "transcripts", f"{args.provider}_{model}_{ts}.json"
        )

    print(f"Provider: {args.provider}")
    print(f"Tarefa: {args.task}")
    print(f"Modelo: {model}")
    print("=" * 60)

    transcript = asyncio.run(
        run_task(
            args.task,
            model=model,
            python_executable=python_exe,
            mcp_server_script=server_script,
            max_turns=args.max_turns,
            transcript_path=transcript_path,
        )
    )

    print("=" * 60)
    print("Resposta final:", transcript["final_text"])
    print("Transcript salvo em:", transcript_path)


if __name__ == "__main__":
    main()
