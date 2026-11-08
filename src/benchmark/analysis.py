"""Analise dos episodios (Secao 6.5): descritiva, testes das hipoteses e
exportacao para o modelo misto em R (benchmark/analysis/glmm.R).

A INSTRUCAO e a unidade de generalizacao: as taxas sao medias de medias por
instrucao, e os intervalos vem de bootstrap em dois niveis (reamostra
instrucoes e, dentro de cada uma, episodios). O Wilson por episodio aparece
so como referencia (ignora a dependencia dentro da instrucao). Sem pandas
nem scipy, de proposito: so stdlib + numpy.
"""
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

Z = 1.959964


# ---- carga -------------------------------------------------------------------
def load_records(results_dir) -> list[dict]:
    recs = []
    for p in sorted(Path(results_dir, "episodes").glob("*.json")):
        recs.append(json.loads(p.read_text(encoding="utf-8")))
    audit = Path(results_dir, "audit_labels.csv")
    if audit.exists():  # revisao humana prevalece sobre o classificador de relato
        with audit.open(encoding="utf-8") as f:
            labels = {r["episode_id"]: r["report_human"] for r in csv.DictReader(f) if r.get("report_human")}
        for r in recs:
            if r["episode_id"] in labels and r.get("metrics"):
                _apply_report_label(r, labels[r["episode_id"]])
    return recs


def _apply_report_label(rec, label):
    m = rec["metrics"]
    m["report_auto"] = m.get("report_auto", m["report"])
    m["report"] = label
    m["claims_success"] = label == "claims_success"
    if rec["spec"]["expected"] == "abstain":
        m["S"] = m["nothing_moved"] and m["released"] and label in ("reports_failure", "asks_clarification")
        if m["S"]:
            m["attribution"] = None
    m["false_success"] = label == "claims_success" and not m["S"]


def flatten(rec) -> dict:
    m, sp, cond = rec["metrics"], rec["spec"], rec["condition"]
    tok = m.get("tokens") or {}
    return {
        "episode_id": rec["episode_id"], "model": rec["model_label"], "provider": rec["provider"],
        "tier": rec.get("tier"), "model_version": rec.get("model_version"), "task": rec["task_id"],
        "family": rec["family"], "source_family": rec["source_family"], "variant": rec["variant"],
        "condition": cond["name"], "scaffolding": cond.get("scaffolding", "guided"),
        "neutral_ids": int(bool(cond.get("neutral_ids"))), "perturb": int(bool(cond.get("perturb"))),
        "seed": rec["seed_idx"], "rep": rec["rep"], "paraphrase": sp["paraphrase_idx"],
        "paraphrase_axis": sp.get("paraphrase_axis") or "", "abstain_type": sp.get("abstain_type") or "",
        "abstain_hint": int(bool(cond.get("abstain_hint"))), "domain": sp["domain"],
        "action": sp["action"], "canonical": int(bool(sp["canonical"])), "expected": sp["expected"],
        "n_objects": len(sp["scene"]), "moved_color": sp.get("moved_role_color"),
        "S": int(m["S"]), "C": int(m["C"]), "C_pert": int(m["C_pert"]), "report": m["report"],
        "false_success": int(m["false_success"]), "nothing_moved": int(m["nothing_moved"]),
        "n_turns": m["n_turns"], "n_tool_calls": m["n_tool_calls"], "n_invalid": m["n_invalid_calls"],
        "efficiency": m["efficiency_ratio"], "uses_place_on": m["uses_place_on"], "uses_place_at": m["uses_place_at"],
        "attribution": m["attribution"] or "", "hit_limit": int(m["hit_max_turns"] or m["hit_max_tool_calls"]),
        "t_total_s": m["t_total_s"], "t_model_s": m["t_model_s"], "t_tools_s": m["t_tools_s"],
        "input_tokens": tok.get("input_tokens"), "output_tokens": tok.get("output_tokens"),
        "thinking_tokens": tok.get("thinking_tokens"), "cost_usd": rec.get("cost_usd"),
    }


def export_csv(rows, path):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


# ---- estatistica ---------------------------------------------------------------
def wilson(k, n, z=Z):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z / den * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, c - h), min(1.0, c + h))


def _groups(rows, key="task"):
    g = defaultdict(list)
    for r in rows:
        g[r[key]].append(r)
    return g


def instr_mean(rows, metric="S"):
    g = _groups(rows)
    return statistics.fmean(statistics.fmean(r[metric] for r in rs) for rs in g.values()) if g else float("nan")


def bootstrap_mean(rows, metric="S", B=2000, seed=0):
    """Bootstrap em dois niveis: instrucoes, e episodios dentro da instrucao."""
    g = [np.array([r[metric] for r in rs], dtype=float) for rs in _groups(rows).values()]
    if not g:
        return float("nan"), (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    est = np.mean([x.mean() for x in g])
    boots = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, len(g), len(g))
        boots[b] = np.mean([g[i][rng.integers(0, len(g[i]), len(g[i]))].mean() for i in idx])
    return float(est), (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))


def paired(rows_a, rows_b, keys=("task", "seed", "rep")):
    """Pares (a, b) na mesma celula (mesma instrucao e semente)."""
    ib = {tuple(r[k] for k in keys): r for r in rows_b}
    return [(ra, ib[tuple(ra[k] for k in keys)]) for ra in rows_a if tuple(ra[k] for k in keys) in ib]


def bootstrap_paired_diff(pairs, metric="S", B=2000, seed=0):
    """Diferenca media a - b (media de medias por instrucao), bootstrap em dois niveis."""
    g = defaultdict(list)
    for a, b in pairs:
        g[a["task"]].append(a[metric] - b[metric])
    g = [np.array(v, dtype=float) for v in g.values()]
    if not g:
        return float("nan"), (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    est = np.mean([x.mean() for x in g])
    boots = np.empty(B)
    for i in range(B):
        idx = rng.integers(0, len(g), len(g))
        boots[i] = np.mean([g[j][rng.integers(0, len(g[j]), len(g[j]))].mean() for j in idx])
    return float(est), (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))


def mcnemar_exact(pairs, metric="S"):
    """Teste exato de McNemar (bilateral) sobre os pares discordantes."""
    b = sum(1 for x, y in pairs if x[metric] and not y[metric])
    c = sum(1 for x, y in pairs if y[metric] and not x[metric])
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "p": 1.0}
    k = min(b, c)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return {"b": b, "c": c, "p": p}


def holm(pvals: dict) -> dict:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, running = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def median_ci(values, B=2000, seed=0):
    v = np.array([x for x in values if x is not None], dtype=float)
    if v.size == 0:
        return float("nan"), (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    boots = [np.median(v[rng.integers(0, v.size, v.size)]) for _ in range(B)]
    return float(np.median(v)), (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))


# ---- relatorio -------------------------------------------------------------------
def _pct(x):
    return "--" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.1f}"


def _ci(ci):
    return f"[{_pct(ci[0])}; {_pct(ci[1])}]"


def _sel(rows, **kw):
    return [r for r in rows if all(r[k] == v if not callable(v) else v(r[k]) for k, v in kw.items())]


def _table(header, lines):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in line) + " |" for line in lines]
    return "\n".join(out)


def report(records, B=2000) -> tuple[str, list[dict]]:
    status = Counter((r["model_label"], r["status"]) for r in records)
    done = [r for r in records if r["status"] == "done"]
    rows = [flatten(r) for r in done]
    models = sorted({r["model"] for r in rows})
    md = ["# Relatorio do benchmark", "",
          "Taxas em %, media de medias por instrucao; IC 95 % por bootstrap em dois niveis "
          "(instrucoes, episodios). Wilson por episodio so como referencia. Base: episodios com status done.", ""]

    md += ["## 0. Qualidade dos dados", "",
           _table(["modelo", "done", "api_error", "infra_error"],
                  [[m, status[(m, "done")], status[(m, "api_error")], status[(m, "infra_error")]]
                   for m in sorted({r["model_label"] for r in records})]),
           "", "Versoes devolvidas pelas APIs: " + ", ".join(
               f"{m}: {sorted({r['model_version'] for r in rows if r['model'] == m and r['model_version']})}"
               for m in models), ""]

    base0 = _sel(rows, rep=0)
    gi = _sel(base0, condition="guided", variant="indirect")
    gi_act = _sel(gi, expected="act")

    md += ["## 1. Sucesso fisico S por modelo e condicao (F1-F6 e gemeas F0; F7 na secao 5)", ""]
    lines = []
    act0 = _sel(base0, expected="act")
    for m in models:
        for label, sub in (("guiada, indireta", _sel(gi_act, model=m)),
                           ("guiada, direta (F0)", _sel(act0, model=m, condition="guided", variant="direct")),
                           ("minima, indireta", _sel(act0, model=m, condition="minimal")),
                           ("perturbacao", _sel(act0, model=m, condition="perturb")),
                           ("ids neutros", _sel(act0, model=m, condition="neutral"))):
            if not sub:
                continue
            est, ci = bootstrap_mean(sub, "S", B)
            k = sum(r["S"] for r in sub)
            lines.append([m, label, len({r['task'] for r in sub}), len(sub), _pct(est), _ci(ci),
                          _ci(wilson(k, len(sub)))])
    md += [_table(["modelo", "condicao", "instrucoes", "episodios", "S", "IC 95 % (bootstrap)", "Wilson (ep.)"],
                  lines), ""]

    md += ["## 2. Por familia (guiada, indireta) -- exploratorio", ""]
    fams = sorted({r["family"] for r in gi})
    lines = []
    for m in models:
        line = [m]
        for f in fams:
            sub = _sel(gi, model=m, family=f)
            line.append(f"{_pct(instr_mean(sub))} (n={len({r['task'] for r in sub})})" if sub else "--")
        lines.append(line)
    md += [_table(["modelo"] + fams, lines), ""]

    md += ["## 3. H1 -- custo semantico (direta - indireta, guiada, mesmas celulas)", ""]
    lines, pv = [], {}
    for m in models:
        pairs = paired(_sel(base0, model=m, condition="guided", variant="direct"), _sel(gi, model=m))
        if not pairs:
            continue
        est, ci = bootstrap_paired_diff(pairs, "S", B)
        t = mcnemar_exact(pairs)
        pv[m] = t["p"]
        lines.append([m, len(pairs), _pct(est), _ci(ci), t["b"], t["c"], f"{t['p']:.4f}"])
    adj = holm(pv)
    md += [_table(["modelo", "pares", "custo (p.p.)", "IC 95 %", "so direta", "so indireta", "p McNemar", "p Holm"],
                  [l + [f"{adj[l[0]]:.4f}"] for l in lines]) if lines else "(sem pares)", ""]

    md += ["## 4. H2 -- efeito do scaffolding (guiada - minima, indireta)", ""]
    lines, pv = [], {}
    for m in models:
        pairs = paired(_sel(gi, model=m), _sel(base0, model=m, condition="minimal"))
        if not pairs:
            continue
        est, ci = bootstrap_paired_diff(pairs, "S", B)
        t = mcnemar_exact(pairs)
        pv[m] = t["p"]
        lines.append([m, len(pairs), _pct(est), _ci(ci), t["b"], t["c"], f"{t['p']:.4f}"])
    adj = holm(pv)
    md += [_table(["modelo", "pares", "efeito (p.p.)", "IC 95 %", "so guiada", "so minima", "p McNemar", "p Holm"],
                  [l + [f"{adj[l[0]]:.4f}"] for l in lines]) if lines else "(sem pares)", ""]

    md += ["## 5. H3 -- instrucoes inviaveis ou ambiguas (F7) e fidelidade do relato", ""]
    f7 = _sel(base0, family="F7", condition=lambda c: c in ("guided", "minimal"))
    lines = []
    for m in models:
        sub = _sel(f7, model=m)
        if not sub:
            continue
        acted = sum(1 - r["nothing_moved"] for r in sub)
        rep = Counter(r["report"] for r in sub)
        lines.append([m, len(sub), _pct(acted / len(sub)), rep["asks_clarification"], rep["reports_failure"],
                      rep["claims_success"], rep["unclear"] + rep["none"], _pct(instr_mean(sub))])
    md += [_table(["modelo", "episodios", "agiu (%)", "pediu esclarec.", "relatou impedimento",
                   "alegou sucesso", "indefinido", "S (%)"], lines) if lines else "(sem F7)", ""]
    types = sorted({r["abstain_type"] for r in f7 if r["abstain_type"]})
    lines = [[m] + [_pct(statistics.fmean(r["S"] for r in _sel(f7, model=m, abstain_type=k)))
                    if _sel(f7, model=m, abstain_type=k) else "--" for k in types] for m in models]
    if types:
        md += ["S (abstencao correta, %) por categoria de F7 (guiada e minima):", "",
               _table(["modelo"] + types, lines), ""]
    perm = _sel(base0, condition="permissao")
    if perm:
        lines = []
        for m in models:
            p7, pa = _sel(perm, model=m, family="F7"), _sel(perm, model=m, expected="act")
            g7 = _sel(base0, model=m, family="F7", condition="guided")
            ga = paired(pa, _sel(gi_act, model=m))
            lines.append([m, _pct(instr_mean(g7)) if g7 else "--", _pct(instr_mean(p7)) if p7 else "--",
                          _pct(statistics.fmean(b["S"] for _, b in ga)) if ga else "--",
                          _pct(instr_mean(pa)) if pa else "--",
                          sum(r["nothing_moved"] for r in pa)])
        md += ["Exploratorio -- condicao `permissao` (prompt guiado + frase que autoriza abster-se):", "",
               _table(["modelo", "S F7 guiada", "S F7 permissao", "S act guiada (mesmas celulas)",
                       "S act permissao", "act sem mexer em nada"], lines), ""]
    lines = []
    for m in models:
        sub = _sel(base0, model=m, expected="act")
        fails = [r for r in sub if not r["S"]]
        k = sum(r["false_success"] for r in fails)
        lines.append([m, len(fails), k, _pct(k / len(fails)) if fails else "--",
                      _ci(wilson(k, len(fails))) if fails else "--"])
    md += ["", "Falso sucesso = P(alega sucesso | nao S), familias F0-F6:", "",
           _table(["modelo", "falhas", "alegou sucesso", "taxa (%)", "Wilson"], lines), ""]

    md += ["## 6. H4 -- perturbacao do alvo", ""]
    lines = []
    for m in models:
        pert = _sel(base0, model=m, condition="perturb")
        if not pert:
            continue
        ctrl = paired(pert, _sel(gi, model=m))
        on = sum(r["uses_place_on"] > 0 for r in pert)
        at_only = sum(r["uses_place_at"] > 0 and r["uses_place_on"] == 0 for r in pert)
        s_on = [r["S"] for r in pert if r["uses_place_on"] > 0]
        s_at = [r["S"] for r in pert if r["uses_place_at"] > 0 and r["uses_place_on"] == 0]
        lines.append([m, len(pert), _pct(instr_mean(pert)),
                      _pct(statistics.fmean(b["S"] for _, b in ctrl)) if ctrl else "--",
                      on, at_only, _pct(statistics.fmean(s_on)) if s_on else "--",
                      _pct(statistics.fmean(s_at)) if s_at else "--"])
    md += [_table(["modelo", "episodios", "S perturb.", "S sem perturb. (mesmas celulas)", "usou place_on",
                   "so place_at", "S com place_on", "S com so place_at"], lines) if lines else "(sem perturbacao)", ""]

    md += ["## 7. H5 -- sensibilidade a forma superficial (guiada, indireta, F1-F6)", ""]
    lines = []
    for m in models:
        sub = _sel(gi_act, model=m)
        if not sub:
            continue
        by_par = {p: _pct(instr_mean(_sel(sub, paraphrase_axis=p)))
                  for p in sorted({r["paraphrase_axis"] for r in sub if r["paraphrase_axis"]})} or                  {p: _pct(instr_mean(_sel(sub, paraphrase=p))) for p in sorted({r["paraphrase"] for r in sub})}
        by_col = {c: _pct(statistics.fmean(r["S"] for r in _sel(sub, moved_color=c)))
                  for c in sorted({r["moved_color"] for r in sub if r["moved_color"]})}
        canon = {k: _pct(instr_mean(_sel(sub, canonical=k))) for k in (1, 0) if _sel(sub, canonical=k)}
        task_means = [statistics.fmean(r["S"] for r in rs) for rs in _groups(sub).values()]
        lines.append([m, by_par, by_col, canon, f"{statistics.pstdev(task_means):.3f}" if task_means else "--"])
    md += [_table(["modelo", "S por parafrase", "S por cor do objeto movido", "S canonica(1)/nova(0)",
                   "DP entre instrucoes"], lines) if lines else "(sem dados)",
           "", "Teste formal de H5: componentes de variancia (instrucao, semente) e efeitos de parafrase e de "
               "cor no GLMM (benchmark/analysis/glmm.R).", ""]

    md += ["## 8. Comparacoes entre modelos (guiada, indireta, mesmas celulas)", ""]
    lines, pv = [], {}
    for i, a in enumerate(models):
        for b in models[i + 1:]:
            pairs = paired(_sel(gi, model=a), _sel(gi, model=b))
            if not pairs:
                continue
            est, ci = bootstrap_paired_diff(pairs, "S", B)
            t = mcnemar_exact(pairs)
            pv[(a, b)] = t["p"]
            lines.append([(a, b), len(pairs), _pct(est), _ci(ci), t["b"], t["c"], f"{t['p']:.4f}"])
    adj = holm(pv)
    md += [_table(["par (a, b)", "pares", "a - b (p.p.)", "IC 95 %", "so a", "so b", "p", "p Holm"],
                  [[f"{l[0][0]} vs {l[0][1]}"] + l[1:] + [f"{adj[l[0]]:.4f}"] for l in lines])
           if lines else "(um modelo so)", ""]

    md += ["## 9. Correcao semantica, lacuna de execucao e atribuicao de falhas (guiada, indireta, F1-F6)", ""]
    lines = []
    attrs = ["percepcao", "semantica", "limite", "interface", "controle", "fisica", "indeterminado"]
    for m in models:
        sub = _sel(gi_act, model=m)
        if not sub:
            continue
        c_rows = [r for r in sub if r["C"]]
        gap = statistics.fmean(r["S"] for r in c_rows) if c_rows else None
        cnt = Counter(r["attribution"] for r in sub if not r["S"])
        lines.append([m, _pct(instr_mean(sub, "C")), _pct(gap)] + [cnt[a] for a in attrs])
    md += [_table(["modelo", "C", "P(S dado C)"] + attrs, lines) if lines else "(sem dados)", ""]

    md += ["## 10. Validade das chamadas, eficiencia, latencia e custo (todas as condicoes)", ""]
    lines = []
    for m in models:
        sub = _sel(base0, model=m)
        calls = sum(r["n_tool_calls"] for r in sub)
        inval = sum(r["n_invalid"] for r in sub)
        eff = [r["efficiency"] for r in sub if r["efficiency"] is not None and r["expected"] == "act"]
        t_med, t_ci = median_ci([r["t_total_s"] for r in sub], B)
        tm_med, _ = median_ci([r["t_model_s"] for r in sub], B)
        tok_med, _ = median_ci([(r["input_tokens"] or 0) + (r["output_tokens"] or 0) + (r["thinking_tokens"] or 0)
                                for r in sub], B)
        costs = [r["cost_usd"] for r in sub if r["cost_usd"] is not None]
        lines.append([m, _pct(inval / calls) if calls else "--", f"{statistics.median(eff):.2f}" if eff else "--",
                      f"{t_med:.1f} [{t_ci[0]:.1f}; {t_ci[1]:.1f}]", f"{tm_med:.1f}", f"{tok_med:.0f}",
                      f"{sum(costs):.2f}" if costs else "--"])
    md += [_table(["modelo", "chamadas invalidas (%)", "chamadas / minimo (mediana)", "tempo total s (mediana, IC)",
                   "tempo do modelo s", "tokens/episodio (mediana)", "custo total US$"], lines), ""]

    reps = [r for r in rows if r["rep"] == 1]
    if reps:
        md += ["## 11. Teste-reteste (mesma celula rodada duas vezes)", ""]
        lines = []
        for m in models:
            pairs = paired(_sel(reps, model=m), _sel(base0, model=m, condition="guided"), keys=("task", "seed", "variant"))
            if pairs:
                agree = statistics.fmean(int(a["S"] == b["S"]) for a, b in pairs)
                lines.append([m, len(pairs), _pct(agree)])
        md += [_table(["modelo", "pares", "concordancia de S (%)"], lines), ""]

    return "\n".join(md), rows
