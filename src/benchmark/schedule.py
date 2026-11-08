"""Plano de episodios: quais combinacoes rodam, e em que ordem.

Delineamento pareado: todo modelo roda toda celula (instrucao x variante x
semente x condicao). A ORDEM e aleatorizada em blocos: cada bloco e uma
celula, com todos os modelos em ordem sorteada, e os blocos sao embaralhados.
Assim mudancas de disponibilidade ou de versao dos modelos ao longo do tempo
nao se confundem com diferencas entre provedores ou entre familias
(Secao 6.5, protocolo iii).
"""
import random

from .tasks import stable_int, variants_of


def _model_entries(cfg, only=None):
    entries = [dict(m) for m in cfg.get("models", [])]
    entries += [dict(b, provider="baseline") for b in cfg.get("baselines", [])]
    for m in entries:
        m.setdefault("label", m["model"])
    if only:
        entries = [m for m in entries if m["label"] in only]
    return entries


def _selected(task, cond, global_seed) -> bool:
    fams = cond.get("families", "all")
    if fams != "all" and task["family"] not in fams:
        return False
    if cond.get("only_perturbable") and not task.get("perturb"):
        return False
    if task["family"] in cond.get("include_families", []):
        return True  # sempre entram, fora do sorteio de subset_fraction
    frac = cond.get("subset_fraction", 1.0)
    return frac >= 1.0 or (stable_int(global_seed, "subset", cond["name"], task["id"]) % 10000) / 10000 < frac


def build_plan(cfg, tasks, models=None) -> list[dict]:
    exp = cfg["experiment"]
    split, seed = exp["split"], exp["global_seed"]
    entries = _model_entries(cfg, models)
    blocks = []
    for cond in cfg["conditions"]:
        n_seeds = cond.get("seeds", exp["seeds"])
        for t in tasks:
            if t["split"] != split or not _selected(t, cond, seed):
                continue
            variants = [v for v in variants_of(t) if v in cond.get("variants", ["indirect", "direct"])]
            for v in variants:
                for s in range(n_seeds):
                    blocks.append({"task_id": t["id"], "family": t["family"], "variant": v, "seed_idx": s,
                                   "condition": cond, "rep": 0})
    rep_cfg = cfg.get("replicates", {})
    if rep_cfg.get("fraction", 0) > 0:
        rng = random.Random(stable_int(seed, "replicates"))
        pool = [b for b in blocks if b["condition"]["name"] == rep_cfg.get("condition", "guided")]
        k = round(len(pool) * rep_cfg["fraction"])
        blocks += [dict(b, rep=1) for b in rng.sample(pool, k)]

    rng = random.Random(stable_int(seed, "order"))
    rng.shuffle(blocks)
    plan = []
    for b in blocks:
        order = list(entries)
        rng.shuffle(order)
        for m in order:
            if m["provider"] == "baseline" and b["rep"] > 0:
                continue  # baselines sao deterministicas
            if not m.get("conditions", "all") == "all" and b["condition"]["name"] not in m["conditions"]:
                continue
            ep_id = (f"{split}__{b['task_id']}__{b['variant']}__s{b['seed_idx']}__{b['condition']['name']}"
                     f"__r{b['rep']}__{m['label']}")
            plan.append({**b, "episode_id": ep_id, "model": m})
    return plan
