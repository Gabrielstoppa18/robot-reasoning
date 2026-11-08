"""Roda o plano de episodios do benchmark (Cap. 6 / benchmark/PROTOCOLO.md).

Requisitos (backend coppelia): CoppeliaSim aberto com sim/phase1_pick_place.ttt
e as chaves no .env. O backend `fake` roda sem simulador (so para testar o
pipeline -- seus resultados nao valem como dado).

Uso:
  .venv/Scripts/python.exe scripts/bench_run.py --config benchmark/pilot.toml --plan-only
  .venv/Scripts/python.exe scripts/bench_run.py --config benchmark/pilot.toml --backend fake --models base-oraculo
  .venv/Scripts/python.exe scripts/bench_run.py --config benchmark/pilot.toml
  .venv/Scripts/python.exe scripts/bench_run.py --config benchmark/experiment.toml   # exige FROZEN.json
"""
import argparse
import asyncio
import os
import sys
from collections import Counter

from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from benchmark.backends import make_backend  # noqa: E402
from benchmark.config import REPO_ROOT, check_frozen, load_config, resolve_path  # noqa: E402
from benchmark.runner import Runner  # noqa: E402
from benchmark.schedule import build_plan  # noqa: E402
from benchmark.tasks import load_tasks  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--backend", choices=["coppelia", "fake"], default="coppelia")
    ap.add_argument("--models", default=None, help="labels separados por virgula (default: todos)")
    ap.add_argument("--limit", type=int, default=None, help="no maximo N episodios nesta execucao")
    ap.add_argument("--plan-only", action="store_true", help="so mostra o plano (contagens) e sai")
    ap.add_argument("--results-dir", default=None, help="sobrescreve experiment.results_dir")
    ap.add_argument("--no-images", action="store_true")
    ap.add_argument("--allow-unfrozen", action="store_true",
                    help="roda o split test sem bater com FROZEN.json (fica registrado no manifesto)")
    args = ap.parse_args()

    cfg = load_config(resolve_path(args.config))
    exp = cfg["experiment"]
    tasks = load_tasks(resolve_path(exp["tasks_file"]))
    models = args.models.split(",") if args.models else None
    plan = build_plan(cfg, tasks, models)

    by_model = Counter(ep["model"]["label"] for ep in plan)
    by_cond = Counter(ep["condition"]["name"] for ep in plan)
    print(f"Experimento {exp['name']} (split {exp['split']}): {len(plan)} episodios planejados")
    print("  por modelo:", dict(by_model))
    print("  por condicao:", dict(by_cond))
    if not plan:
        print("ERRO: plano vazio -- ha instrucoes no split escolhido?")
        sys.exit(1)
    if args.plan_only:
        return

    unfrozen = False
    if exp["split"] == "test" and args.backend == "coppelia":
        problems = check_frozen(cfg, tasks, resolve_path(exp["lock_file"]))
        if problems:
            print("O experimento nao bate com o congelado:\n  " + "\n  ".join(problems))
            if not args.allow_unfrozen:
                sys.exit(2)
            unfrozen = True

    bad_ids = sorted({ep["model"]["model"] for ep in plan if "CONFIRMAR" in str(ep["model"]["model"])})
    if bad_ids and args.backend == "coppelia":
        print(f"ERRO: confirme os IDs de modelo no config: {bad_ids}")
        sys.exit(2)

    env_keys = {"claude": ("ANTHROPIC_API_KEY",), "gemini": ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
                "openai": ("OPENAI_API_KEY",)}
    missing = sorted({ep["model"]["label"] for ep in plan if ep["model"]["provider"] in env_keys
                      and not any(os.environ.get(k) for k in env_keys[ep["model"]["provider"]])})
    if missing:
        print(f"ERRO: chave de API vazia no .env para {missing} -- preencha ou use --models para tira-los")
        sys.exit(2)

    out_dir = resolve_path(args.results_dir or exp["results_dir"])
    if args.backend == "fake":
        out_dir = out_dir.parent / (out_dir.name + "-FAKE")
    backend = make_backend(args.backend, repo_root=REPO_ROOT, python_executable=sys.executable,
                           settle_steps=cfg["scene"].get("settle_steps", 20))
    runner = Runner(cfg, tasks, backend, out_dir, save_images=not args.no_images)
    manifest = runner.write_manifest(plan, unfrozen, {"backend_kind": args.backend, "models_filter": models})
    print("Manifesto:", manifest)
    summary = asyncio.run(runner.run(plan, limit=args.limit, budget_usd=exp.get("budget_usd")))
    print("Resumo:", summary, "->", out_dir)


if __name__ == "__main__":
    main()
