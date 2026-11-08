"""Validacao do conjunto de instrucoes antes do congelamento (PROTOCOLO.md, 3).

Erros (bloqueiam o congelamento):
  - esquema, lacunas desconhecidas, nomes de ferramenta ou dicas de
    procedimento no texto;
  - instrucao que nao instancia em alguma semente;
  - plano de referencia (oraculo, backend falso) que NAO satisfaz o
    verificador em alguma semente -> verificador, plano ou tolerancia
    inconsistentes;
  - split test: menos de 3 parafrases, parafrases sem eixo (paraphrase_axes
    com base, lexical e contexto), F7 sem abstain_type, sem revisor distinto
    do autor, F1-F6 sem gemea direta e sem justificativa (no_twin_reason).
Avisos (decisao do pesquisador, registrada no relatorio):
  - cotas do plano de amostragem (familias, dominios, acoes, empilhar <= 25 %);
  - desbalanceamento de cor do objeto movido;
  - F7 com menos de 5 categorias de abstencao; menos de 2 instrucoes F5 com
    4 ou mais movimentos (LITERATURA.md, 4);
  - instrucoes que a heuristica lexical (first_mentioned) resolve em todas as
    sementes (atalho: o sucesso ali nao exige inferencia).
"""
import asyncio
import re
import string
from collections import Counter

from .agents.scripted import FirstMentionedAgent, OracleAgent
from .backends.fake import FakeBackend
from .palette import PALETTE
from .scoring import score_episode
from .tasks import FAMILIES, PARAPHRASE_AXES, instantiate, variants_of
from .tool_specs import SYSTEM_PROMPTS, TOOL_NAMES

_FORBIDDEN = [*TOOL_NAMES, "ferramenta", "coordenada", "get_scene", "chame ", "use a função"]
_STACK_ACTIONS = re.compile(r"(p[oô]r sobre|empilhar)", re.I)


def _moves(t) -> int:
    """Movimentos (pick) do plano de referencia; foreach_other conta 1."""
    return sum(1 for s in t.get("reference_plan", []) if s[0] in ("pick", "foreach_other"))


def _placeholders(text):
    return {f for _, f, _, _ in string.Formatter().parse(text) if f}


async def _run(agent, spec, cond):
    be = FakeBackend()
    init = be.reset(spec)
    async with be.tools(spec, cond) as tc:
        tr = await agent.run(spec["prompt"], tc, SYSTEM_PROMPTS["guided"], spec)
    return score_episode(spec, init, be.finalize(spec), tr, be.events(), cond)


def validate(tasks, cfg, split=None) -> dict:
    exp = cfg["experiment"]
    split = split or exp["split"]
    seeds = max([c.get("seeds", exp["seeds"]) for c in cfg["conditions"]] + [exp["seeds"]])
    sel = [t for t in tasks if t["split"] == split]
    errors, warnings, rows = [], [], []
    moved_colors = Counter()
    oracle, lexical = OracleAgent(), FirstMentionedAgent()

    for t in sel:
        tid = t["id"]
        allowed = set(t["roles"]) | set(t.get("absent_roles", [])) | {"OTHERS"}
        texts = {"indirect": t["text"].get(exp["lang"], [])}
        if t.get("direct"):
            texts["direct"] = t["direct"].get(exp["lang"], [])
        for v, ts in texts.items():
            if not ts:
                errors.append(f"{tid}: sem textos '{v}' no idioma {exp['lang']}")
            for s in ts:
                bad = _placeholders(s) - allowed
                if bad:
                    errors.append(f"{tid} ({v}): lacunas desconhecidas {bad}")
                low = s.lower()
                hits = [w for w in _FORBIDDEN if w.lower() in low]
                if hits:
                    errors.append(f"{tid} ({v}): texto contem nome de ferramenta/dica {hits}")
        if t["expected"] == "act" and not t.get("reference_plan"):
            errors.append(f"{tid}: instrucao 'act' sem reference_plan (o oraculo nao tem o que executar)")
        if split == "test":
            if len(texts["indirect"]) < 3:
                errors.append(f"{tid}: menos de 3 parafrases")
            if set(t.get("paraphrase_axes") or []) != set(PARAPHRASE_AXES):
                errors.append(f"{tid}: paraphrase_axes deve cobrir {PARAPHRASE_AXES}")
            if t["family"] == "F7" and not t.get("abstain_type"):
                errors.append(f"{tid}: F7 sem abstain_type")
            reviewers = [r for r in t.get("reviewed_by", []) if r and r != t.get("author")]
            if not reviewers:
                errors.append(f"{tid}: sem revisor distinto do autor (reviewed_by)")
        if t["family"] in ("F1", "F2", "F3", "F4", "F5", "F6") and not t.get("direct"):
            msg = f"{tid}: sem gemea direta" + (f" (justificativa: {t['no_twin_reason']})"
                                                if t.get("no_twin_reason") else "")
            (errors if split == "test" and not t.get("no_twin_reason") else warnings).append(msg)

        lex_wins = 0
        for v in variants_of(t):
            for s in range(seeds):
                try:
                    spec = instantiate(t, v, s, lang=exp["lang"], global_seed=exp["global_seed"],
                                       perturb=bool(t.get("perturb")))
                except Exception as e:  # noqa: BLE001
                    errors.append(f"{tid} {v} s{s}: nao instancia ({e})")
                    continue
                if v == "indirect" and spec["moved_role_color"]:
                    moved_colors[spec["moved_role_color"]] += 1
                for cond in ({"scaffolding": "guided"}, {"scaffolding": "guided", "perturb": True}):
                    if cond.get("perturb") and not spec.get("perturbation"):
                        continue
                    sp = spec if cond.get("perturb") else {**spec, "perturbation": None}
                    m = asyncio.run(_run(oracle, sp, cond))
                    if not m["S"]:
                        errors.append(f"{tid} {v} s{s}{' perturb' if cond.get('perturb') else ''}: oraculo "
                                      f"falhou (C={m['C']}, relato={m['report']}) -- plano/verificador inconsistentes")
                if v == "indirect":
                    lex_wins += asyncio.run(_run(lexical, {**spec, "perturbation": None}, {}))["S"]
        if lex_wins == seeds and t["expected"] == "act":
            warnings.append(f"{tid}: a heuristica 'primeiro mencionado' resolve todas as sementes (atalho lexical)")
        rows.append({"id": tid, "family": t["family"], "domain": t["domain"], "action": t["action"],
                     "abstain_type": t.get("abstain_type") or "", "moves": _moves(t),
                     "axes": "/".join(t.get("paraphrase_axes") or []),
                     "canonical": t.get("canonical", False), "paraphrases": len(texts["indirect"]),
                     "twin": bool(t.get("direct")), "lexical_wins": f"{lex_wins}/{seeds}"})

    fam = Counter(t["family"] for t in sel)
    dom = Counter(t["domain"] for t in sel)
    act = Counter(t["action"] for t in sel)
    stack = sum(1 for t in sel if _STACK_ACTIONS.search(t["action"]))
    n_act = sum(1 for t in sel if t["expected"] == "act")
    abstain_types = Counter(t["abstain_type"] for t in sel if t["family"] == "F7" and t.get("abstain_type"))
    long_f5 = sum(1 for t in sel if t["family"] == "F5" and _moves(t) >= 4)
    if split == "test":
        if len(abstain_types) < 5:
            warnings.append(f"F7 cobre {len(abstain_types)} categorias de abstencao (plano: >= 5)")
        if long_f5 < 2:
            warnings.append(f"so {long_f5} instrucoes F5 com 4 ou mais movimentos (plano: >= 2)")
        for f in FAMILIES:
            if f != "F0" and fam[f] < 6:
                warnings.append(f"familia {f}: {fam[f]} instrucoes (plano: 6 a 8)")
        if len(dom) < 6:
            warnings.append(f"so {len(dom)} dominios semanticos (plano: >= 6)")
    if n_act and stack / n_act > 0.25:
        warnings.append(f"empilhar/por sobre = {stack}/{n_act} das instrucoes 'act' (plano: <= 25 %)")
    total = sum(moved_colors.values())
    if total:
        for c in PALETTE:
            share = moved_colors[c] / total
            if abs(share - 1 / len(PALETTE)) > 0.10:
                warnings.append(f"cor do objeto movido desbalanceada: {c} = {share:.0%} (uniforme = "
                                f"{1 / len(PALETTE):.0%})")
    return {"split": split, "n": len(sel), "errors": errors, "warnings": warnings, "rows": rows,
            "families": dict(fam), "domains": dict(dom), "abstain_types": dict(abstain_types), "long_f5": long_f5, "actions": dict(act), "moved_colors": dict(moved_colors),
            "stack_share": stack / n_act if n_act else None}


def to_markdown(res) -> str:
    out = [f"# Validacao do conjunto de instrucoes (split {res['split']}, {res['n']} instrucoes)", "",
           f"**Erros: {len(res['errors'])}** | Avisos: {len(res['warnings'])}", ""]
    if res["errors"]:
        out += ["## Erros", ""] + [f"- {e}" for e in res["errors"]] + [""]
    if res["warnings"]:
        out += ["## Avisos", ""] + [f"- {w}" for w in res["warnings"]] + [""]
    out += ["## Cobertura", "", f"- Familias: {res['families']}", f"- Dominios: {res['domains']}",
            f"- Acoes: {res['actions']}", f"- Categorias de abstencao (F7): {res['abstain_types']}",
            f"- F5 com 4 ou mais movimentos: {res['long_f5']}", f"- Cor do objeto movido (indireta, todas as sementes): {res['moved_colors']}",
            f"- Fracao empilhar/por sobre: {res['stack_share']:.0%}" if res["stack_share"] is not None else "", "",
            "## Instrucoes", "",
            "| id | familia | dominio | acao | abstencao | movimentos | canonica | parafrases | eixos | gemea | lexical |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    out += [f"| {r['id']} | {r['family']} | {r['domain']} | {r['action']} | {r['abstain_type']} | {r['moves']} | "
            f"{int(r['canonical'])} | {r['paraphrases']} | {r['axes']} | {int(r['twin'])} | {r['lexical_wins']} |"
            for r in res["rows"]]
    return "\n".join(out)
