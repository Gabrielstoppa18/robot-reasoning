"""Biblioteca de predicados sobre poses (Quadro 6.1 / Secao 6.3.3).

Um predicado recebe o estado inicial e o final (verdade-terreno do simulador,
ou o estado previsto pelo executor simbolico) e devolve verdadeiro/falso.
O criterio avalia o ESTADO FINAL, entao qualquer solucao valida e aceita.

Formato de estado:
    {"objects": {cor: [x, y, z]}, "held": cor|None, "base_xy": [x, y], "cube": 0.04}

Formato da especificacao (JSON), ja com papeis resolvidos para cores:
    {"on": ["blue", "red"]}
    {"and": [{"on": ["blue", "red"]}, {"unchanged": ["green"]}]}
    {"not": {...}}, {"or": [...]}
"""
import math

from .geometry import base_dist, dist3, dist_xy

# tolerancias (m). Registradas no congelamento do experimento (bench_freeze).
TOL = {
    "on_xy": 0.015,  # mesmo criterio de run_vision_pick_place.py
    "on_z_min": 0.7,  # em arestas, altura relativa minima
    "on_z_max": 1.3,
    "near_max": 0.09,  # distancia entre centros (~2 arestas + folga)
    "beside_min": 0.035,  # nao sobrepostos
    "level_z": 0.01,  # "na mesa": |z - aresta/2|
    "far_gain": 0.10,  # afastar: aumento minimo de distancia
    "base_margin": 0.03,  # mais perto da base: diferenca minima
    "line_dev": 0.03,  # desvio maximo da reta (fila)
    "unchanged": 0.01,  # deslocamento maximo de "nao mexer"
    "moved": 0.03,  # deslocamento minimo de "mexeu"
    "swap": 0.03,  # troca: cada um a menos disso da posicao inicial do outro
}


def _pos(state, c):
    return state["objects"][c]


def _on(init, final, a, b, tol):
    s = final.get("cube", 0.04)
    pa, pb = _pos(final, a), _pos(final, b)
    dz = (pa[2] - pb[2]) / s
    return (abs(pa[0] - pb[0]) < tol["on_xy"] and abs(pa[1] - pb[1]) < tol["on_xy"]
            and tol["on_z_min"] <= dz <= tol["on_z_max"])


def _on_table(final, c, tol):
    s = final.get("cube", 0.04)
    return abs(_pos(final, c)[2] - s / 2) <= tol["level_z"] and final.get("held") != c


def _near(init, final, a, b, tol):
    return dist3(_pos(final, a), _pos(final, b)) <= tol["near_max"]


def _beside(init, final, a, b, tol):
    return (_near(init, final, a, b, tol) and _on_table(final, a, tol) and _on_table(final, b, tol)
            and dist_xy(_pos(final, a), _pos(final, b)) >= tol["beside_min"])


def _far(init, final, a, b, tol):
    return (dist_xy(_pos(final, a), _pos(final, b))
            >= dist_xy(_pos(init, a), _pos(init, b)) + tol["far_gain"])


def _closer_to_base(init, final, a, b, tol):
    base = final["base_xy"]
    return base_dist(_pos(final, a), base) + tol["base_margin"] <= base_dist(_pos(final, b), base)


def _collinear(points, tol):
    if len(points) < 3:
        return True
    p0, p1 = points[0], points[-1]
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    norm = math.hypot(dx, dy)
    if norm < 1e-6:
        return False
    return all(abs((p[0] - p0[0]) * dy - (p[1] - p0[1]) * dx) / norm <= tol["line_dev"] for p in points)


def _line_ordered(init, final, *cs, tol):
    """Fila do mais perto ao mais longe da base, na ordem dada, e alinhada."""
    ok = all(_closer_to_base(init, final, cs[i], cs[i + 1], tol) for i in range(len(cs) - 1))
    return ok and _collinear([_pos(final, c) for c in cs], tol) and all(_on_table(final, c, tol) for c in cs)


def _unchanged(init, final, *cs, tol):
    return all(dist3(_pos(init, c), _pos(final, c)) <= tol["unchanged"] and final.get("held") != c for c in cs)


def _moved(init, final, c, tol):
    return dist3(_pos(init, c), _pos(final, c)) >= tol["moved"]


def _swapped(init, final, a, b, tol):
    return (dist_xy(_pos(final, a), _pos(init, b)) <= tol["swap"]
            and dist_xy(_pos(final, b), _pos(init, a)) <= tol["swap"]
            and _on_table(final, a, tol) and _on_table(final, b, tol))


def _at_initial(init, final, a, b, tol):
    """a ocupa a posicao inicial de b (rodizio, vaga), na mesa."""
    return dist_xy(_pos(final, a), _pos(init, b)) <= tol["swap"] and _on_table(final, a, tol)


def _nothing_moved(init, final, tol):
    return _unchanged(init, final, *init["objects"], tol=tol)


def _released(init, final, tol):
    return final.get("held") is None


_BINARY = {"on": _on, "near": _near, "beside": _beside, "far": _far, "closer_to_base": _closer_to_base,
           "swapped": _swapped, "at_initial": _at_initial}


def evaluate(spec, init, final, tol=None) -> bool:
    tol = {**TOL, **(tol or {})}
    (op, arg), = spec.items()
    if op == "and":
        return all(evaluate(s, init, final, tol) for s in arg)
    if op == "or":
        return any(evaluate(s, init, final, tol) for s in arg)
    if op == "not":
        return not evaluate(arg, init, final, tol)
    if op in _BINARY:
        a, b = arg
        if a not in final["objects"] or b not in final["objects"]:
            return False
        return _BINARY[op](init, final, a, b, tol)
    if op == "all_near":  # {"all_near": [[c1, c2], ref]}
        cs, ref = arg
        return all(_near(init, final, c, ref, tol) for c in cs)
    if op == "line_ordered":
        return _line_ordered(init, final, *arg, tol=tol)
    if op == "base_dist_min":  # {"base_dist_min": [c, d]}: c a pelo menos d m da base (restricao de cena)
        c, d = arg
        return dist_xy(_pos(final, c), final.get("base_xy") or [0.0, 0.0]) >= d
    if op == "within":  # {"within": [a, b, d]}: distancia no plano <= d (restricao de cena)
        a, b, d = arg
        return dist_xy(_pos(final, a), _pos(final, b)) <= d
    if op == "unchanged":
        return _unchanged(init, final, *arg, tol=tol)
    if op == "moved":
        return _moved(init, final, arg[0], tol)
    if op == "on_table":
        return all(_on_table(final, c, tol) for c in arg)
    if op == "nothing_moved":
        return _nothing_moved(init, final, tol)
    if op == "released":
        return _released(init, final, tol)
    if op == "true":
        return True
    raise ValueError(f"predicado desconhecido: {op}")


OPS = set(_BINARY) | {"and", "or", "not", "all_near", "line_ordered", "within", "base_dist_min", "unchanged", "moved",
                      "on_table", "nothing_moved", "released", "true"}


def resolve(spec, roles: dict, others: list):
    """Troca papeis ("A", "B") por cores e "@others" pela lista de distratores."""
    if isinstance(spec, dict):
        return {k: resolve(v, roles, others) for k, v in spec.items()}
    if isinstance(spec, list):
        out = []
        for v in spec:
            if v == "@others":
                out.extend(others)
            else:
                out.append(resolve(v, roles, others))
        return out
    if isinstance(spec, str) and spec in roles:
        return roles[spec]
    return spec


def ops_used(spec) -> set:
    if isinstance(spec, dict):
        out = set()
        for k, v in spec.items():
            if k in OPS:
                out.add(k)
            out |= ops_used(v)
        return out
    if isinstance(spec, list):
        return set().union(*[ops_used(v) for v in spec]) if spec else set()
    return set()
