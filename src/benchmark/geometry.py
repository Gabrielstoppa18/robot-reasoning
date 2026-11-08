"""Geometria da cena: area de trabalho, amostragem de posicoes, pontos livres.

Area de trabalho default: intersecao do campo de visao das duas cameras
(centro (0,375; 0,225), altura 0,6 m, FOV 60 graus, baseline 0,12 m em y)
no plano do topo dos cubos, com margem de ~5 cm, e longe da base do UR5.
Conferir com scripts/bench_check_perception.py antes de usar.
"""
import math

CUBE = 0.04

# r_max: distancia maxima da base (na origem). Com a garra vertical, o UR5
# deixa de alcancar a mesa a ~0,65 m (canto (0,54; 0,42) a 0,68 m: recusado
# pelo planejador; ate ~0,63 m, pega e solta normalmente -- 2026-10-10).
DEFAULT_WORKSPACE = {"x": [0.20, 0.55], "y": [0.02, 0.42], "r_max": 0.62}
DEFAULT_MIN_SEP = 0.10  # entre centros: dedos abertos da RG2 (~11 cm) nao varrem o vizinho
MAX_REACH = 0.80  # UR5 alcanca ~0,85 m; margem


def dist_xy(p, q) -> float:
    return math.hypot(p[0] - q[0], p[1] - q[1])


def dist3(p, q) -> float:
    return math.sqrt(sum((p[k] - q[k]) ** 2 for k in range(3)))


def in_workspace(p, ws, margin=0.0) -> bool:
    return (ws["x"][0] - margin <= p[0] <= ws["x"][1] + margin
            and ws["y"][0] - margin <= p[1] <= ws["y"][1] + margin
            and math.hypot(p[0], p[1]) <= ws.get("r_max", math.inf) + margin)


def sample_positions(rng, n, ws=None, min_sep=DEFAULT_MIN_SEP, max_tries=2000):
    """n posicoes (x, y, z=CUBE/2) uniformes na area, com separacao minima."""
    ws = ws or DEFAULT_WORKSPACE
    for _ in range(max_tries):
        pts = []
        for _ in range(n):
            for _ in range(200):
                p = [rng.uniform(*ws["x"]), rng.uniform(*ws["y"]), CUBE / 2]
                if in_workspace(p, ws) and all(dist_xy(p, q) >= min_sep for q in pts):
                    pts.append(p)
                    break
        if len(pts) == n:
            return pts
    raise RuntimeError(f"nao coube {n} objetos com separacao {min_sep} na area {ws}")


def grid(ws, step=0.02):
    x = ws["x"][0]
    while x <= ws["x"][1] + 1e-9:
        y = ws["y"][0]
        while y <= ws["y"][1] + 1e-9:
            if math.hypot(x, y) <= ws.get("r_max", math.inf):
                yield [round(x, 4), round(y, 4)]
            y += step
        x += step


def free_spots(occupied, ws=None, clearance=0.08, step=0.02):
    """Pontos da grade a pelo menos `clearance` de todos os ocupados."""
    ws = ws or DEFAULT_WORKSPACE
    return [p for p in grid(ws, step) if all(dist_xy(p, q) >= clearance for q in occupied)]


def beside_spot(ref, occupied, ws=None, offset=0.06, clearance=0.07):
    """Ponto livre a `offset` do ref (8 direcoes), ou None."""
    ws = ws or DEFAULT_WORKSPACE
    for k in range(8):
        a = k * math.pi / 4
        p = [ref[0] + offset * math.cos(a), ref[1] + offset * math.sin(a)]
        if in_workspace(p, ws) and all(dist_xy(p, q) >= clearance for q in occupied):
            return p
    return None


def far_spot(ref, occupied, ws=None, clearance=0.08):
    spots = free_spots(occupied, ws, clearance)
    return max(spots, key=lambda p: dist_xy(p, ref)) if spots else None


def base_dist(p, base_xy) -> float:
    return dist_xy(p, base_xy)
