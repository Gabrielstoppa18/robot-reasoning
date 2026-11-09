"""Executor simbolico (Secao 6.3.3): aplica as chamadas registradas no
transcript ao modelo cinematico do mundo, a partir do estado VERDADEIRO
inicial, e diz se o plano, executado sem erro de percepcao ou de controle,
satisfaria a meta. Separa "plano certo, execucao errada" de "plano errado"
sem enumerar sequencias aceitaveis por instrucao.

Roda duas vezes por episodio: sem a perturbacao (correcao semantica C) e com
ela (C_pert). C verdadeiro e C_pert falso indica primitiva inadequada
(place_at com coordenada congelada onde place_on relia o alvo).
"""
from .geometry import MAX_REACH, dist_xy
from .tool_specs import ACTION_TOOLS
from .world import World


def execute(init_state: dict, calls: list[dict], perturbation=None, neutral_map=None):
    w = World(init_state["objects"], init_state["base_xy"], init_state.get("cube", 0.04), perturbation)
    to_color = (lambda n: neutral_map.get(n, n)) if neutral_map else (lambda n: n)
    invalid = []
    for i, c in enumerate(calls):
        name, args = c["name"], c.get("args") or {}
        if name == "get_scene_objects":
            continue
        try:
            if name == "pick_object":
                color = to_color(args["object_name"])
                if w.held is not None:
                    invalid.append({"call": i, "reason": "pick_while_holding"})
                r = w.pick(color)
                if not r["ok"]:
                    invalid.append({"call": i, "reason": "pick_nonexistent_or_covered", "object": color})
            elif name == "place_on":
                color = to_color(args["object_name"])
                r = w.place_on(color, args.get("clearance_m"))
                if not r["ok"]:
                    invalid.append({"call": i, "reason": f"place_on_{r['reason']}", "object": color})
            elif name == "place_at":
                x, y, z = float(args["x"]), float(args["y"]), float(args["z"])
                if dist_xy([x, y], w.base_xy) > MAX_REACH or z < 0 or z > 0.6:
                    invalid.append({"call": i, "reason": "place_at_unreachable"})
                r = w.place_at(x, y, z)
                if not r["ok"]:
                    invalid.append({"call": i, "reason": "place_at_not_holding"})
            else:
                invalid.append({"call": i, "reason": "unknown_tool"})
        except (KeyError, TypeError, ValueError):
            invalid.append({"call": i, "reason": "bad_arguments"})
    n_actions = sum(1 for c in calls if c["name"] in ACTION_TOOLS)
    return w.state(), {"invalid_calls": invalid, "n_action_calls": n_actions, "releases": [
        e for e in w.events if e["type"] == "release"], "perturbation_applied": bool(
        w.perturbation and w.perturbation.get("applied"))}


def release_deviation(real_releases: list, sym_releases: list) -> float | None:
    """Maior desvio no plano entre a pose verdadeira na soltura e a prevista."""
    if not real_releases or not sym_releases:
        return None
    devs = [dist_xy(r["pose"], s["pose"]) for r, s in zip(real_releases, sym_releases)
            if r.get("object") == s.get("object")]
    return max(devs) if devs else None
