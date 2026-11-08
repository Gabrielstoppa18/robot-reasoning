"""Conjunto de instrucoes: carga, validacao de esquema e instanciacao.

Uma INSTRUCAO e um modelo com lacunas de cor ({A}, {B}...). Um EPISODIO
instancia a instrucao numa SEMENTE: numero de objetos, cores que ocupam cada
papel, posicoes, parafrase usada e (opcional) perturbacao. Tudo depende so de
(semente global, id da instrucao, indice da semente) -- e NAO do modelo nem da
condicao --, o que da o delineamento pareado: todos os modelos e condicoes
veem exatamente a mesma cena para a mesma (instrucao, semente).
"""
import hashlib
import json
import math
import random
from pathlib import Path

from . import predicates
from .geometry import CUBE, DEFAULT_MIN_SEP, DEFAULT_WORKSPACE, dist_xy, in_workspace, sample_positions
from .palette import PALETTE, color_word

FAMILIES = {
    "F0": "Controle direto (gemea direta de uma instrucao indireta)",
    "F1": "Interpretacao figurada; papeis declarados",
    "F2": "Convencoes e senso comum, sem mapeamento declarado",
    "F3": "Relacoes espaciais e de ordem",
    "F4": "Restricoes e negacao",
    "F5": "Varias etapas e estado oculto",
    "F6": "Selecao e quantificacao",
    "F7": "Inviavel ou ambigua: reconhecer a lacuna",
}

# Categorias de abstencao do F7 (taxonomia da RoboAbstention, 2026; ver
# benchmark/LITERATURA.md, 1.2). "referente_ambiguo" nao e testavel aqui: a
# percepcao devolve um blob por cor.
ABSTAIN_TYPES = {
    "referente_ausente": "o objeto citado nao existe na cena",
    "intencao_subespecificada": "falta informacao para saber o que fazer",
    "intencao_subjetiva": "depende de uma preferencia que o robo nao conhece",
    "inviabilidade_fisica": "a acao pedida e fisicamente impossivel",
    "capacidade_ausente": "exige uma propriedade que o robo nao percebe",
    "contraditoria": "a meta se contradiz",
    "premissa_falsa": "pressupoe um estado da cena que nao e verdadeiro",
}
# Eixo de cada parafrase (LIBERO-Para; EmbodiedBench "complex instruction"):
# base = redacao de referencia; lexical = troca verbo e/ou referencia ao
# objeto; contexto = a redacao base com frase irrelevante antes.
PARAPHRASE_AXES = ("base", "lexical", "contexto")

REQUIRED_FIELDS = ("id", "family", "split", "domain", "action", "roles", "n_objects", "text", "success", "expected")
PLAN_STEPS = {"look", "foreach_other", "pick", "place_on", "place_beside", "place_far", "place_temp", "place_at_initial",
              "place_closer_than", "place_slot"}


def stable_int(*parts) -> int:
    h = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return int(h[:15], 16)


def load_tasks(path) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    tasks = data["instructions"] if isinstance(data, dict) else data
    for t in tasks:
        check_schema(t)
    ids = [t["id"] for t in tasks]
    dup = {i for i in ids if ids.count(i) > 1}
    if dup:
        raise ValueError(f"ids repetidos: {sorted(dup)}")
    return tasks


def check_schema(t: dict):
    missing = [f for f in REQUIRED_FIELDS if f not in t]
    if missing:
        raise ValueError(f"{t.get('id', '?')}: campos ausentes {missing}")
    if t["family"] not in FAMILIES or t["family"] == "F0":
        raise ValueError(f"{t['id']}: familia invalida {t['family']} (F0 so existe como gemea 'direct')")
    if t["split"] not in ("dev", "test"):
        raise ValueError(f"{t['id']}: split deve ser dev|test")
    if t["expected"] not in ("act", "abstain"):
        raise ValueError(f"{t['id']}: expected deve ser act|abstain")
    unknown = predicates.ops_used(t["success"]) - predicates.OPS
    if unknown:
        raise ValueError(f"{t['id']}: predicados desconhecidos {unknown}")
    if t.get("abstain_type") is not None and t["abstain_type"] not in ABSTAIN_TYPES:
        raise ValueError(f"{t['id']}: abstain_type invalido {t['abstain_type']}")
    axes = t.get("paraphrase_axes")
    if axes is not None:
        if any(a not in PARAPHRASE_AXES for a in axes):
            raise ValueError(f"{t['id']}: paraphrase_axes deve usar {PARAPHRASE_AXES}")
        for lang, texts in t["text"].items():
            if len(texts) != len(axes):
                raise ValueError(f"{t['id']}: paraphrase_axes tem {len(axes)} itens e text[{lang}] tem {len(texts)}")
    for step in t.get("reference_plan", []):
        if step[0] not in PLAN_STEPS:
            raise ValueError(f"{t['id']}: passo de plano desconhecido {step[0]}")


def variants_of(t: dict) -> list[str]:
    return ["indirect", "direct"] if t.get("direct") else ["indirect"]


def _assign_colors(t, seed_idx, rng, palette):
    """Papeis presentes e ausentes -> cores. A cor do PRIMEIRO papel gira entre
    as sementes (quadrado latino sobre a paleta, deslocado por instrucao), os
    demais papeis sao sorteados; papeis com cor fixa (F2) nao giram."""
    fixed = dict(t.get("fixed_colors", {}))
    roles = list(t["roles"]) + list(t.get("absent_roles", []))
    free_roles = [r for r in roles if r not in fixed]
    pool = [c for c in palette if c not in fixed.values()]
    out = dict(fixed)
    if free_roles:
        first = pool[(stable_int(t["id"]) + seed_idx) % len(pool)]
        out[free_roles[0]] = first
        rest = [c for c in pool if c != first]
        rng.shuffle(rest)
        for r in free_roles[1:]:
            out[r] = rest.pop()
    return out


def _sample_perturbation(t, scene, roles, rng, ws, min_sep):
    """Deslocamento do alvo (10-16 cm, direcao sorteada) que mantem o objeto
    na area de trabalho e longe dos demais."""
    spec = t["perturb"]
    color = roles[spec["role"]]
    p0 = scene[color]
    others = [p for c, p in scene.items() if c != color]
    for _ in range(500):
        r, a = rng.uniform(0.10, 0.16), rng.uniform(0, 2 * math.pi)
        p = [p0[0] + r * math.cos(a), p0[1] + r * math.sin(a), p0[2]]
        if in_workspace(p, ws) and all(dist_xy(p, q) >= min_sep for q in others):
            return {"object": color, "delta": [round(p[0] - p0[0], 4), round(p[1] - p0[1], 4)],
                    "tool_kind": "place", "call_index": spec.get("call_index", 0), "step": spec.get("step", 50)}
    return None


GEOMETRIC_RULES = ("farthest_from_base", "nearest_to_base", "nearest_to:<papel>", "farthest_from:<papel>")
_GEOM_MARGIN = 0.03  # o escolhido precisa vencer o segundo colocado por 3 cm (sem empate perceptivel)


def _apply_geometric_roles(t, scene, roles_all, base_xy):
    """Papeis definidos pela cena (F3/F6): 'o cubo mais longe da base', 'o
    mais perto do {A}'. Escolhe entre as cores que nao tem papel fixo; devolve
    None se a escolha for ambigua (empate dentro da margem)."""
    geo = t.get("geometric_roles") or {}
    if not geo:
        return roles_all
    roles = dict(roles_all)
    taken = {roles[r] for r in t["roles"] if r not in geo}
    for role, rule in geo.items():
        cands = [c for c in scene if c not in taken]
        if rule in ("farthest_from_base", "nearest_to_base"):
            key = lambda c: dist_xy(scene[c], base_xy)
        elif rule.startswith(("nearest_to:", "farthest_from:")):
            ref = scene[roles[rule.split(":", 1)[1]]]
            key = lambda c: dist_xy(scene[c], ref)
        else:
            raise ValueError(f"{t['id']}: regra geometrica desconhecida {rule}")
        ranked = sorted(cands, key=key, reverse=rule == "farthest_from_base" or rule.startswith("farthest_from:"))
        if len(ranked) >= 2 and abs(key(ranked[0]) - key(ranked[1])) < _GEOM_MARGIN:
            return None
        roles[role] = ranked[0]
        taken.add(ranked[0])
    return roles


def _expand_plan(plan, others):
    """["foreach_other", [passos com "$o"]] -> passos repetidos para cada distrator."""
    out = []
    for step in plan:
        if step and step[0] == "foreach_other":
            for o in others:
                out.extend([[o if a == "$o" else a for a in sub] for sub in step[1]])
        else:
            out.append(step)
    return out


def _join(words, lang):
    conj = " e " if lang == "pt" else " and "
    return words[0] if len(words) == 1 else ", ".join(words[:-1]) + conj + words[-1]


def instantiate(t: dict, variant: str, seed_idx: int, *, lang: str = "pt", global_seed: int = 20261008,
                perturb: bool = False, ws=None, min_sep: float = DEFAULT_MIN_SEP, palette=None,
                base_xy=None) -> dict:
    ws = ws or DEFAULT_WORKSPACE
    palette = palette or PALETTE
    base_xy = list(base_xy or [0.0, 0.0])
    rng = random.Random(stable_int(global_seed, t["id"], seed_idx))
    roles0 = _assign_colors(t, seed_idx, rng, palette)
    absent = {r: roles0[r] for r in t.get("absent_roles", [])}

    lo, hi = t["n_objects"]
    n = rng.randint(max(lo, len(t["roles"])), hi)
    present0 = [roles0[r] for r in t["roles"]]
    pool = [c for c in palette if c not in roles0.values()]
    rng.shuffle(pool)
    colors = present0 + pool[: n - len(present0)]
    if len(colors) < n:
        raise ValueError(f"{t['id']}: paleta insuficiente para {n} objetos")

    for _ in range(500):
        pts = sample_positions(rng, n, ws, min_sep)
        rng.shuffle(pts)
        scene = {c: [round(v, 4) for v in p] for c, p in zip(colors, pts)}
        roles_all = _apply_geometric_roles(t, scene, roles0, base_xy)
        if roles_all is None:
            continue
        present = {r: roles_all[r] for r in t["roles"]}
        others = [c for c in colors if c not in present.values()]
        success = predicates.resolve(t["success"], roles_all, others)
        state = {"objects": scene, "held": None, "base_xy": base_xy, "cube": CUBE}
        if t.get("scene_constraint") and not predicates.evaluate(
                predicates.resolve(t["scene_constraint"], roles_all, others), state, state):
            continue
        # cena nao-trivial: o estado inicial nao pode ja satisfazer a meta
        if t["expected"] == "act" and predicates.evaluate(success, state, state):
            continue
        break
    else:
        raise RuntimeError(f"{t['id']} s{seed_idx}: nao achei cena valida")

    texts = t["text"][lang] if variant == "indirect" else t["direct"][lang]
    # parafrase gira com a semente, deslocada por instrucao: com 3 sementes cada
    # instrucao usa os 3 eixos, e com menos (piloto) todos os eixos aparecem no
    # conjunto -- e o eixo nao fica preso ao indice da semente
    p_idx = (seed_idx + stable_int(t["id"], "paraphrase")) % len(texts)
    words = {r: color_word(c, lang) for r, c in roles_all.items()}
    if others:
        words["OTHERS"] = _join([color_word(c, lang) for c in others], lang)
    prompt = texts[p_idx].format(**words)

    ids = [f"obj_{k + 1}" for k in range(n)]
    shuffled = list(colors)
    random.Random(stable_int(global_seed, t["id"], seed_idx, "ids")).shuffle(shuffled)
    neutral_map = dict(zip(ids, shuffled))

    perturbation = None
    if perturb and t.get("perturb"):
        prng = random.Random(stable_int(global_seed, t["id"], seed_idx, "perturb"))
        perturbation = _sample_perturbation(t, scene, roles_all, prng, ws, min_sep)

    plan = predicates.resolve(_expand_plan(t.get("reference_plan", []), others), roles_all, others)
    return {
        "task_id": t["id"],
        "source_family": t["family"],
        "family": "F0" if variant == "direct" else t["family"],
        "variant": variant,
        "split": t["split"],
        "domain": t["domain"],
        "action": t["action"],
        "canonical": t.get("canonical", False),
        "seed_idx": seed_idx,
        "lang": lang,
        "paraphrase_idx": p_idx,
        "paraphrase_axis": (t.get("paraphrase_axes") or [None] * len(texts))[p_idx] if variant == "indirect" else None,
        "abstain_type": t.get("abstain_type"),
        "prompt": prompt,
        "roles": present,
        "absent": absent,
        "others": others,
        "scene": scene,
        "base_xy": base_xy,
        "success": success,
        "expected": t["expected"],
        "min_calls": t.get("min_calls"),
        "reference_plan": plan,
        "perturbation": perturbation,
        "neutral_map": neutral_map,
        "moved_role_color": present.get(t["moved_role"]) if t.get("moved_role") else None,
    }
