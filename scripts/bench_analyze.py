"""Gera o relatorio estatistico (report.md) e a tabela por episodio
(episodes.csv, entrada do GLMM em benchmark/analysis/glmm.R).

Uso:
  .venv/Scripts/python.exe scripts/bench_analyze.py benchmark/results/piloto-dev
  Rscript benchmark/analysis/glmm.R benchmark/results/piloto-dev/episodes.csv
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from benchmark.analysis import export_csv, load_records, report  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_dir")
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()
    recs = load_records(args.results_dir)
    if not recs:
        print("nenhum episodio em", Path(args.results_dir) / "episodes")
        sys.exit(1)
    md, rows = report(recs, B=args.bootstrap)
    out = Path(args.results_dir)
    (out / "report.md").write_text(md, encoding="utf-8")
    export_csv(rows, out / "episodes.csv")
    print(md)
    print(f"\n-> {out / 'report.md'}\n-> {out / 'episodes.csv'} ({len(rows)} episodios)")


if __name__ == "__main__":
    main()
