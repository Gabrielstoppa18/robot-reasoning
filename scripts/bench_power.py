"""Analise de poder por simulacao para o delineamento pareado (H1, H2 e
comparacoes entre modelos), usada para fixar o numero de instrucoes e de
sementes (PROTOCOLO.md, secao 6).

Modelo gerador (o mesmo da analise): logit P(S) = mu + u_instrucao + v_celula
+ delta * condicao, com u ~ N(0, sd_instr), v ~ N(0, sd_seed). As duas
condicoes comparadas compartilham u e v (mesma instrucao, mesma cena); o
ruido restante e o sorteio de Bernoulli de cada episodio. Teste: McNemar
exato bilateral, alfa = 0,05 (Holm para k contrastes: alfa/k no pior caso).

Os defaults sao suposicoes; depois do piloto, troque por --sd-instr e --p0
estimados (o GLMM do piloto da sd_instr).

Uso:
  .venv/Scripts/python.exe scripts/bench_power.py
  .venv/Scripts/python.exe scripts/bench_power.py --p0 0.7 --sd-instr 1.2 --diffs 0.10 0.15
"""
import argparse
import math

import numpy as np


def mcnemar_p(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def delta_for(p0, diff, sd_instr, sd_seed, rng, n=200000):
    """delta (logit) que produz a diferenca MARGINAL `diff` dada a heterogeneidade."""
    z = rng.normal(0, 1, n) * math.hypot(sd_instr, sd_seed)
    lo, hi = -10.0, 10.0
    mu = _solve(lambda m: np.mean(1 / (1 + np.exp(-(m + z)))) - p0, -10, 10)
    target = p0 - diff
    d = _solve(lambda dd: np.mean(1 / (1 + np.exp(-(mu + dd + z)))) - target, lo, hi)
    return mu, d


def _solve(f, lo, hi, it=60):
    for _ in range(it):
        mid = (lo + hi) / 2
        if f(lo) * f(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def power(n_instr, n_seeds, p0, diff, sd_instr, sd_seed, alpha, sims, rng):
    mu, d = delta_for(p0, diff, sd_instr, sd_seed, rng)
    hits = 0
    for _ in range(sims):
        u = np.repeat(rng.normal(0, sd_instr, n_instr), n_seeds)
        v = rng.normal(0, sd_seed, n_instr * n_seeds)
        pa = 1 / (1 + np.exp(-(mu + u + v)))
        pb = 1 / (1 + np.exp(-(mu + d + u + v)))
        a = rng.random(pa.size) < pa
        b = rng.random(pb.size) < pb
        if mcnemar_p(int(np.sum(a & ~b)), int(np.sum(~a & b))) < alpha:
            hits += 1
    return hits / sims


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p0", type=float, default=0.65, help="taxa de sucesso na condicao de referencia")
    ap.add_argument("--diffs", type=float, nargs="+", default=[0.10, 0.15, 0.20])
    ap.add_argument("--instr", type=int, nargs="+", default=[28, 42, 56, 70])
    ap.add_argument("--seeds", type=int, nargs="+", default=[2, 3, 5])
    ap.add_argument("--sd-instr", type=float, default=1.5, help="DP do efeito de instrucao (logit)")
    ap.add_argument("--sd-seed", type=float, default=0.5, help="DP do efeito de cena (logit)")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--contrasts", type=int, default=1, help="k contrastes (Holm, pior caso alfa/k)")
    ap.add_argument("--sims", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    alpha = args.alpha / args.contrasts
    print(f"p0={args.p0}  sd_instr={args.sd_instr}  sd_seed={args.sd_seed}  alfa={alpha:.4f}  sims={args.sims}\n")
    print("| diferenca | instrucoes | " + " | ".join(f"{s} sementes" for s in args.seeds) + " |")
    print("|---|---|" + "---|" * len(args.seeds))
    for diff in args.diffs:
        for n in args.instr:
            cells = [f"{power(n, s, args.p0, diff, args.sd_instr, args.sd_seed, alpha, args.sims, rng):.2f}"
                     for s in args.seeds]
            print(f"| {diff * 100:.0f} p.p. | {n} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
