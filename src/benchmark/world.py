"""Modelo cinematico simples do mundo (quem esta sobre quem, onde cada cubo
pousa). Usado em dois lugares:

- executor simbolico (symbolic.py): aplica as chamadas registradas a partir
  do estado verdadeiro inicial e diz se o PLANO, executado sem erro de
  controle, satisfaria a meta (correcao semantica C);
- backend falso (backends/fake.py): simula as ferramentas sem CoppeliaSim,
  para testar o pipeline inteiro sem custo.

As regras imitam o que o servidor real faz (mesmas pre-condicoes e mensagens
de mcp_server.py), sem dinamica: um cubo solto cai reto e pousa no topo do
cubo que estiver embaixo, ou na mesa.
"""
import copy

from .geometry import CUBE, MAX_REACH, dist_xy


class World:
    def __init__(self, objects: dict, base_xy=(0.0, 0.0), cube=CUBE, perturbation=None):
        self.objects = {c: list(p) for c, p in objects.items()}
        self.held = None
        self.base_xy = list(base_xy)
        self.cube = cube
        self.perturbation = dict(perturbation) if perturbation else None
        self._place_calls = 0
        self.events = []
        self.last_xy = None  # xy da ultima pega (onde o objeto cai se a garra abrir)

    # ---- consultas -------------------------------------------------------
    def state(self) -> dict:
        return {"objects": copy.deepcopy(self.objects), "held": self.held,
                "base_xy": list(self.base_xy), "cube": self.cube}

    def covered(self, c) -> bool:
        p = self.objects[c]
        return any(o != c and o != self.held and abs(q[0] - p[0]) < self.cube * 0.75
                   and abs(q[1] - p[1]) < self.cube * 0.75 and q[2] > p[2] + self.cube * 0.5
                   for o, q in self.objects.items())

    def visible(self) -> list:
        return [c for c in self.objects if c != self.held and not self.covered(c)]

    def _landing_z(self, xy, exclude) -> float:
        z = self.cube / 2
        for o, q in self.objects.items():
            if o in (exclude, self.held):
                continue
            if abs(q[0] - xy[0]) < self.cube and abs(q[1] - xy[1]) < self.cube:
                z = max(z, q[2] + self.cube)
        return z

    def _drop_held(self, xy):
        c = self.held
        self.held = None
        z = self._landing_z(xy, c)
        self.objects[c] = [xy[0], xy[1], z]
        self.events.append({"type": "release", "object": c, "pose": list(self.objects[c])})
        return self.objects[c]

    def _maybe_perturb(self):
        pert = self.perturbation
        if not pert or pert.get("applied"):
            return
        if self._place_calls == pert.get("call_index", 0):
            c = pert["object"]
            if c in self.objects and c != self.held:
                p = self.objects[c]
                self.objects[c] = [p[0] + pert["delta"][0], p[1] + pert["delta"][1], p[2]]
                pert["applied"] = True
                self.events.append({"type": "perturbation", "object": c, "pose": list(self.objects[c])})

    # ---- acoes (mesmo contrato das ferramentas) --------------------------
    def pick(self, c) -> dict:
        if c not in self.objects or c not in self.visible():
            return {"ok": False, "reason": "not_detected"}
        if self.held is not None:
            # o servidor real nao checa: abre a garra (o objeto preso cai) e segue
            self._drop_held(self.last_xy or self.objects[self.held][:2])
            self.events.append({"type": "dropped_on_pick"})
        p = self.objects[c]
        self.held = c
        self.last_xy = p[:2]
        self.objects[c] = [p[0], p[1], p[2] + 0.15]
        return {"ok": True, "position": p}

    def place_on(self, target, clearance=None) -> dict:
        if self.held is None:
            return {"ok": False, "reason": "not_holding"}
        if target not in self.objects or target not in self.visible():
            return {"ok": False, "reason": "not_detected"}
        # a perturbacao acontece durante o movimento; place_on rele o alvo depois
        self._maybe_perturb()
        self._place_calls += 1
        t = self.objects[target]
        pose = self._drop_held(t[:2])
        return {"ok": True, "position": [t[0], t[1], t[2] + (clearance if clearance is not None else self.cube)],
                "landed": pose}

    def place_at(self, x, y, z) -> dict:
        if self.held is None:
            return {"ok": False, "reason": "not_holding"}
        self._maybe_perturb()
        self._place_calls += 1
        if dist_xy([x, y], self.base_xy) > MAX_REACH or z > 0.6:
            # o servidor real tentaria a IK e nao chegaria; aqui consideramos que solta onde der
            self.events.append({"type": "unreachable", "target": [x, y, z]})
        pose = self._drop_held([x, y])
        return {"ok": True, "landed": pose}
