"""Auditoria humana do relato final (fidelidade, H3) e da atribuicao de falhas.

export: gera audit_sheet.csv com (a) todos os episodios cujo relato o
        classificador marcou como unclear/none e (b) uma amostra aleatoria
        (default 20 %) dos demais, estratificada por modelo. A ordem e
        embaralhada e o modelo NAO aparece na planilha (avaliacao cega).
        O avaliador preenche report_human (claims_success | reports_failure |
        asks_clarification | none) e, se quiser, attribution_human e notes.
import: le a planilha preenchida, grava audit_labels.csv (usado por
        bench_analyze.py, que da precedencia ao rotulo humano) e imprime a
        concordancia classificador x humano (kappa de Cohen). Com uma
        segunda planilha (--second), imprime a concordancia entre avaliadores.

Uso:
  .venv/Scripts/python.exe scripts/bench_audit.py export benchmark/results/principal-v1
  .venv/Scripts/python.exe scripts/bench_audit.py import benchmark/results/principal-v1 audit_sheet_preenchida.csv
"""
import argparse
import csv
import os
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from benchmark.analysis import load_records  # noqa: E402

LABELS = ("claims_success", "reports_failure", "asks_clarification", "none")


def kappa(a, b):
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def export(results_dir, frac, seed):
    recs = [r for r in load_records(results_dir) if r["status"] == "done" and r["provider"] != "baseline"]
    rng = random.Random(seed)
    chosen = [r for r in recs if r["metrics"]["report"] in ("unclear", "none")]
    rest = [r for r in recs if r not in chosen]
    by_model = {}
    for r in rest:
        by_model.setdefault(r["model_label"], []).append(r)
    for rs in by_model.values():
        chosen += rng.sample(rs, max(1, round(len(rs) * frac))) if rs else []
    rng.shuffle(chosen)
    path = Path(results_dir) / "audit_sheet.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["episode_id_hidden", "instruction", "expected", "final_text", "S", "report_auto",
                    "report_human", "attribution_auto", "attribution_human", "notes"])
        key = []
        for i, r in enumerate(chosen):
            m = r["metrics"]
            w.writerow([f"A{i:04d}", r["spec"]["prompt"], r["spec"]["expected"],
                        (r["transcript"].get("final_text") or "").replace("\n", " "), int(m["S"]), m["report"], "",
                        m["attribution"] or "", "", ""])
            key.append((f"A{i:04d}", r["episode_id"]))
    with (Path(results_dir) / "audit_key.csv").open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([("audit_id", "episode_id"), *key])
    print(f"{len(chosen)} episodios -> {path} (chave em audit_key.csv; nao entregue a chave ao avaliador)")
    print("Rotulos validos para report_human:", ", ".join(LABELS))


def _read_sheet(path):
    with open(path, encoding="utf-8") as f:
        return {r["episode_id_hidden"]: r for r in csv.DictReader(f)}


def import_(results_dir, sheet, second=None):
    with (Path(results_dir) / "audit_key.csv").open(encoding="utf-8") as f:
        key = {r["audit_id"]: r["episode_id"] for r in csv.DictReader(f)}
    rows = _read_sheet(sheet)
    bad = [k for k, r in rows.items() if r["report_human"] and r["report_human"] not in LABELS]
    if bad:
        print("rotulos invalidos em:", bad[:10])
        sys.exit(1)
    filled = {k: r for k, r in rows.items() if r["report_human"]}
    with (Path(results_dir) / "audit_labels.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["episode_id", "report_human", "attribution_human", "notes"])
        for k, r in filled.items():
            w.writerow([key[k], r["report_human"], r.get("attribution_human", ""), r.get("notes", "")])
    auto = [r["report_auto"] for r in filled.values() if r["report_auto"] != "unclear"]
    hum = [r["report_human"] for r in filled.values() if r["report_auto"] != "unclear"]
    print(f"{len(filled)} rotulos humanos gravados. Classificador x humano (excluindo unclear): "
          f"acordo {sum(a == b for a, b in zip(auto, hum))}/{len(auto)}, kappa = {kappa(auto, hum):.3f}")
    if second:
        r2 = _read_sheet(second)
        common = [k for k in filled if r2.get(k, {}).get("report_human")]
        a = [filled[k]["report_human"] for k in common]
        b = [r2[k]["report_human"] for k in common]
        print(f"Entre avaliadores: {len(common)} itens em comum, kappa = {kappa(a, b):.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("results_dir")
    e.add_argument("--fraction", type=float, default=0.20)
    e.add_argument("--seed", type=int, default=7)
    i = sub.add_parser("import")
    i.add_argument("results_dir")
    i.add_argument("sheet")
    i.add_argument("--second", default=None, help="planilha de um segundo avaliador (concordancia)")
    args = ap.parse_args()
    if args.cmd == "export":
        export(args.results_dir, args.fraction, args.seed)
    else:
        import_(args.results_dir, args.sheet, args.second)


if __name__ == "__main__":
    main()
