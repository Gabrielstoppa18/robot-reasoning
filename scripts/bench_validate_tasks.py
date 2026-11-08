"""Valida o conjunto de instrucoes (esquema, cotas, contrabalanceamento e
solubilidade pelo oraculo no backend falso). Nao precisa de simulador.

Uso:
  .venv/Scripts/python.exe scripts/bench_validate_tasks.py --config benchmark/pilot.toml
  .venv/Scripts/python.exe scripts/bench_validate_tasks.py --config benchmark/experiment.toml --out benchmark/validacao_test.md
Sai com codigo 1 se houver erros.
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from benchmark.config import load_config, resolve_path  # noqa: E402
from benchmark.tasks import load_tasks  # noqa: E402
from benchmark.validate import to_markdown, validate  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default=None, help="default: o do config")
    ap.add_argument("--out", default=None, help="grava o relatorio em markdown")
    args = ap.parse_args()
    cfg = load_config(resolve_path(args.config))
    tasks = load_tasks(resolve_path(cfg["experiment"]["tasks_file"]))
    res = validate(tasks, cfg, args.split)
    md = to_markdown(res)
    print(md)
    if args.out:
        Path(resolve_path(args.out)).write_text(md, encoding="utf-8")
    sys.exit(1 if res["errors"] or res["n"] == 0 else 0)


if __name__ == "__main__":
    main()
