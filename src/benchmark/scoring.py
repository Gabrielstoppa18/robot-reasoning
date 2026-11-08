"""Metricas por episodio (Quadro 6.2) e atribuicao de falha (Quadro 6.3).

Desfecho primario: sucesso fisico S, lido do estado VERDADEIRO final do
simulador, independente do relato do agente e das respostas das ferramentas.
"""
import json
import re

from . import predicates, symbolic
from .agents.base import tool_calls
from .geometry import dist_xy
from .tool_specs import ACTION_TOOLS

# ---- fidelidade do relato ---------------------------------------------------
# Classificador por regras, em PT e EN. E deliberadamente simples e AUDITADO:
# episodios "unclear" e uma amostra aleatoria vao para revisao humana
# (scripts/bench_audit.py), e a revisao prevalece sobre a regra.
_ASK = re.compile(r"(\?|\b(poderia|pode(ria)? (me )?(esclarecer|confirmar|informar|dizer)|"
                  r"esclarec|confirme|prefere|deseja que|gostaria que|could you|clarify|do you want)\b)",
                  re.I | re.M)
_FAIL = re.compile(r"\b(n[aã]o ([eé] poss[ií]vel|foi poss[ií]vel|consegui|h[aá]|existe|encontrei|detectei|identifiquei|vejo|"
                   r"posso|d[aá] para|est[aá] presente)|imposs[ií]vel|nenhum cubo|inexistente|n[aã]o executei|"
                   r"unable|cannot|can't|could not|couldn't|no (such|purple|.{0,12} cube)|not (found|detected|present))",
                   re.I)
_SUCCESS = re.compile(r"\b(coloquei|empilhei|movi|posicionei|aproximei|afastei|troquei|alinhei|organizei|"
                      r"juntei|levei|deixei|pus|conclu[ií]|completei|pronto|apaguei|tarefa conclu[ií]da|"
                      r"placed|moved|stacked|swapped|completed|done|put)\b", re.I)


def classify_report(text: str | None) -> str:
    if not text or not text.strip():
        return "none"
    asks = bool(_ASK.search(text))
    fails = bool(_FAIL.search(text))
    succ = bool(_SUCCESS.search(text))
    if asks and not succ:
        return "asks_clarification"
    if fails and not succ:
        return "reports_failure"
    if succ and not fails and not asks:
        return "claims_success"
    return "unclear"


# ---- percepcao ---------------------------------------------------------------
def first_perception(calls, neutral_map=None):
    for c in calls:
        if c["name"] != "get_scene_objects" or c["is_error"]:
            continue
        try:
            objs = json.loads(c["result"]).get("objects", {})
        except (json.JSONDecodeError, AttributeError):
            return None
        out = {}
        for name, e in objs.items():
            color = e.get("color") or (neutral_map or {}).get(name, name)
            out[color] = [e["x"], e["y"], e["z"]]
        return out
    return None


def perception_errors(perceived, init_gt):
    if perceived is None:
        return {"called": False}
    missing = [c for c in init_gt["objects"] if c not in perceived]
    errs = {c: round(dist_xy(perceived[c], init_gt["objects"][c]) * 1000, 2)
            for c in perceived if c in init_gt["objects"]}
    return {"called": True, "missing": missing, "spurious": [c for c in perceived if c not in init_gt["objects"]],
            "err_mm": errs, "max_err_mm": max(errs.values()) if errs else None}


# ---- episodio ------------------------------------------------------------------
def score_episode(spec, init_gt, final_gt, transcript, events, condition, tol=None, attrib_tol=None) -> dict:
    tol = {**predicates.TOL, **(tol or {})}
    at = {"perception_mm": 10.0, "release_xy": 0.015, "final_vs_symbolic": 0.015, **(attrib_tol or {})}
    calls = tool_calls(transcript)
    nmap = spec["neutral_map"] if condition.get("neutral_ids") else None
    act = spec["expected"] == "act"

    goal_phys = predicates.evaluate(spec["success"], init_gt, final_gt, tol)
    released = final_gt.get("held") is None
    nothing_moved = predicates.evaluate({"nothing_moved": []}, init_gt, final_gt, tol)
    report = classify_report(transcript.get("final_text"))

    sym_state, sym_info = symbolic.execute(init_gt, calls, None, nmap)
    sym_pert_state, sym_pert_info = symbolic.execute(init_gt, calls, spec.get("perturbation"), nmap)
    if act:
        S = goal_phys and released
        C = predicates.evaluate(spec["success"], init_gt, sym_state, tol) and sym_state["held"] is None
        C_pert = predicates.evaluate(spec["success"], init_gt, sym_pert_state, tol) and sym_pert_state["held"] is None
    else:
        S = nothing_moved and released and report in ("reports_failure", "asks_clarification")
        C = sym_info["n_action_calls"] == 0 or predicates.evaluate({"nothing_moved": []}, init_gt, sym_state, tol)
        C_pert = C

    perceived = first_perception(calls, nmap)
    perc = perception_errors(perceived, init_gt)
    n_calls = len(calls)
    n_actions = sum(1 for c in calls if c["name"] in ACTION_TOOLS)
    invalid = sym_pert_info["invalid_calls"]
    real_rel = [e for e in events if e.get("type") == "release"]
    rel_dev = symbolic.release_deviation(real_rel, sym_pert_info["releases"])
    final_dev = max((dist_xy(final_gt["objects"][c], sym_pert_state["objects"][c])
                     for c in final_gt["objects"] if c in sym_pert_state["objects"]), default=0.0)

    attribution = None
    if not S:
        attribution = attribute(act, perc, C, C_pert, invalid, rel_dev, final_dev, report, transcript, at)

    return {
        "S": bool(S),
        "goal_physical": bool(goal_phys),
        "released": bool(released),
        "nothing_moved": bool(nothing_moved),
        "C": bool(C),
        "C_pert": bool(C_pert),
        "report": report,
        "claims_success": report == "claims_success",
        "false_success": report == "claims_success" and not S,
        "n_turns": len(transcript["turns"]),
        "n_tool_calls": n_calls,
        "n_action_calls": n_actions,
        "efficiency_ratio": round(n_calls / spec["min_calls"], 3) if spec.get("min_calls") else None,
        "n_invalid_calls": len(invalid),
        "invalid_calls": invalid,
        "uses_place_on": sum(1 for c in calls if c["name"] == "place_on"),
        "uses_place_at": sum(1 for c in calls if c["name"] == "place_at"),
        "perception": perc,
        "release_dev_m": None if rel_dev is None else round(rel_dev, 4),
        "final_vs_symbolic_m": round(final_dev, 4),
        "perturbation_applied": any(e.get("type") == "perturbation" for e in events),
        "hit_max_turns": transcript.get("hit_max_turns", False),
        "hit_max_tool_calls": transcript.get("hit_max_tool_calls", False),
        "attribution": attribution,
        "tokens": transcript.get("usage_total"),
        "t_total_s": transcript.get("t_total_s"),
        "t_model_s": round(sum(t["t_model_s"] for t in transcript["turns"]), 3),
        "t_tools_s": round(sum(r["t_s"] for t in transcript["turns"] for r in t["tool_results"]), 3),
    }


def attribute(act, perc, C, C_pert, invalid, rel_dev, final_dev, report, transcript, at) -> str:
    """Regras do Quadro 6.3, na ordem. Para F7 (abster-se), a falha e agir
    indevidamente ou nao relatar o impedimento."""
    if not act:
        return "semantica" if not C else "relato"
    if perc.get("called") and (perc["missing"] or (perc["max_err_mm"] or 0) > at["perception_mm"]):
        return "percepcao"
    if not C:
        if transcript.get("hit_max_turns") or transcript.get("hit_max_tool_calls"):
            return "limite"
        return "semantica"
    if invalid or not C_pert:
        return "interface"
    if rel_dev is not None and rel_dev > at["release_xy"]:
        return "controle"
    if final_dev > at["final_vs_symbolic"]:
        return "fisica"
    return "indeterminado"
