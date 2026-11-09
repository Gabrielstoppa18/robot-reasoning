"""Congela (pre-registra) o experimento principal: grava benchmark/FROZEN.json
com o hash do conjunto de teste, do experiment.toml e das tolerancias. Depois
disso, nenhum ajuste de prompt, ferramenta, verificador ou tolerancia (PROTOCOLO.md, 4).

So congela se a validacao do split passar sem erros. Recomenda-se, em
seguida, um commit e uma etiqueta git:  git tag pre-registro-v1

Uso:
  .venv/Scripts/python.exe scripts/bench_freeze.py --config benchmark/experiment.toml
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from benchmark.config import freeze_record, load_config, resolve_path  # noqa: E402
from benchmark.tasks import load_tasks  # noqa: E402
from benchmark.validate import validate  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--force", action="store_true", help="sobrescreve um FROZEN.json existente")
    args = ap.parse_args()
    cfg = load_config(resolve_path(args.config))
    exp = cfg["experiment"]
    tasks = load_tasks(resolve_path(exp["tasks_file"]))
    res = validate(tasks, cfg)
    if res["n"] == 0 or res["errors"]:
        print(f"Nao congelado: {res['n']} instrucoes no split {exp['split']}, {len(res['errors'])} erros.")
        for e in res["errors"]:
            print(" -", e)
        sys.exit(1)
    bad = [m["model"] for m in cfg.get("models", []) if "CONFIRMAR" in m["model"]]
    if bad:
        print("Nao congelado: confirme os IDs de modelo:", bad)
        sys.exit(1)
    lock = resolve_path(exp["lock_file"])
    if lock.exists() and not args.force:
        print(f"{lock} ja existe. Use --force so se o congelamento anterior for invalido (e registre o motivo).")
        sys.exit(1)
    rec = freeze_record(cfg, tasks)
    rec["warnings_at_freeze"] = res["warnings"]
    lock.write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(rec, indent=2, ensure_ascii=False))
    print(f"\nCongelado em {lock}. Faca commit e git tag agora.")


if __name__ == "__main__":
    main()
