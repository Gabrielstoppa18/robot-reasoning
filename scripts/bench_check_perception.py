"""Portao de PERCEPCAO (PROTOCOLO.md, 2): antes de qualquer episodio com LLM,
a visao precisa localizar todas as cores da paleta em cenas sorteadas.

Para N cenas aleatorias (2 a 5 cubos, posicoes sorteadas na area de trabalho
do experimento), cria a cena pelo mesmo caminho do harness, roda o
VisionPipeline e compara com a verdade-terreno.

Criterios de aprovacao (todos):
  - deteccao: 100 % dos cubos presentes detectados, nenhuma cor espuria;
  - erro no plano: maximo <= 5 mm e media <= 3 mm (o vies observado no Cap. 5
    foi de 1,6 a 3,4 mm em duas posicoes).
Grava o relatorio (JSON) com o erro por cor e por posicao -- serve tambem de
dado para a secao de percepcao da dissertacao.

Uso (CoppeliaSim aberto com a cena):
  .venv/Scripts/python.exe scripts/bench_check_perception.py --config benchmark/pilot.toml --scenes 40
"""
import argparse
import json
import os
import random
import statistics
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from benchmark.backends.coppelia import CoppeliaBackend  # noqa: E402
from benchmark.config import REPO_ROOT, load_config, resolve_path, workspace  # noqa: E402
from benchmark.geometry import dist_xy, sample_positions  # noqa: E402
from benchmark.palette import PALETTE  # noqa: E402
from robot_control import SimConnection  # noqa: E402
from vision import VisionPipeline  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--scenes", type=int, default=40)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--max-mm", type=float, default=5.0)
    ap.add_argument("--mean-mm", type=float, default=3.0)
    args = ap.parse_args()
    cfg = load_config(resolve_path(args.config))
    ws, min_sep = workspace(cfg), cfg["scene"].get("min_sep", 0.10)
    rng = random.Random(args.seed)
    be = CoppeliaBackend(REPO_ROOT)
    results, missing, spurious = [], 0, 0

    for k in range(args.scenes):
        n = rng.randint(2, len(PALETTE))
        colors = rng.sample(PALETTE, n)
        pts = sample_positions(rng, n, ws, min_sep)
        scene = {c: [round(p[0], 4), round(p[1], 4), p[2]] for c, p in zip(colors, pts)}
        be.reset({"scene": scene})
        conn = SimConnection()
        conn.set_stepping(True)
        conn.start_simulation()
        for _ in range(3):
            conn.step()
        detected = VisionPipeline(conn).get_object_positions()
        truth = be.snapshot(colors)["objects"]
        conn.stop_simulation()
        conn.set_stepping(False)
        for c in colors:
            if c not in detected:
                missing += 1
                results.append({"scene": k, "color": c, "truth": truth[c], "detected": None, "err_mm": None})
            else:
                results.append({"scene": k, "color": c, "truth": truth[c], "detected": list(detected[c]),
                                "err_mm": round(dist_xy(detected[c], truth[c]) * 1000, 3)})
        spurious += len([c for c in detected if c not in colors])
        print(f"cena {k + 1}/{args.scenes}: {n} cubos, detectados {len([c for c in colors if c in detected])}, "
              f"espurios {[c for c in detected if c not in colors]}")

    errs = [r["err_mm"] for r in results if r["err_mm"] is not None]
    by_color = {c: [r["err_mm"] for r in results if r["color"] == c and r["err_mm"] is not None] for c in PALETTE}
    summary = {
        "when": datetime.now().isoformat(timespec="seconds"), "scenes": args.scenes, "cubes": len(results),
        "missing": missing, "spurious": spurious,
        "max_mm": max(errs) if errs else None, "mean_mm": statistics.fmean(errs) if errs else None,
        "by_color": {c: {"n": len(v), "mean_mm": statistics.fmean(v) if v else None, "max_mm": max(v) if v else None}
                     for c, v in by_color.items()},
    }
    summary["passed"] = bool(errs) and missing == 0 and spurious == 0 and summary["max_mm"] <= args.max_mm \
        and summary["mean_mm"] <= args.mean_mm
    out = REPO_ROOT / "benchmark" / "results" / f"perception_gate_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "results": results}, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("APROVADO" if summary["passed"] else "REPROVADO", "->", out)
    sys.exit(0 if summary["passed"] else 1)


if __name__ == "__main__":
    main()
