"""Backend falso: ferramentas sobre o modelo cinematico (world.World), sem
CoppeliaSim. Serve para testar o pipeline (agentes, registro, metricas,
analise) e para depurar instrucoes sem custo. NAO produz dados para a
dissertacao: nao ha IK, garra nem fisica.

Para nao ser otimista demais, a percepcao falsa reproduz o vies sistematico
medido no Cap. 5 (centroide puxado ~2,5 mm em direcao ao eixo optico).
"""
import json
import math
from contextlib import asynccontextmanager

from ..tool_specs import tool_list
from ..world import World
from . import NameMap

OPTICAL_AXIS_XY = (0.375, 0.225)


class FakeToolClient:
    def __init__(self, world: World, names: NameMap, scaffolding: str, bias_m: float):
        self.world = world
        self.names = names
        self.bias_m = bias_m
        self.tools = tool_list(scaffolding, neutral_ids=names.active)

    def _perceive(self):
        out = {}
        for c in self.world.visible():
            x, y, z = self.world.objects[c]
            dx, dy = OPTICAL_AXIS_XY[0] - x, OPTICAL_AXIS_XY[1] - y
            d = math.hypot(dx, dy) or 1.0
            entry = {"x": round(x + self.bias_m * dx / d, 4), "y": round(y + self.bias_m * dy / d, 4),
                     "z": round(self.world.cube / 2 if z < self.world.cube else z, 4)}
            if self.names.active:
                entry["color"] = c
            out[self.names.ident(c)] = entry
        return out

    def _not_detected(self, name):
        return {"success": False, "error": f"objeto '{name}' nao detectado. "
                                           f"objetos visiveis agora: {list(self._perceive().keys())}"}

    async def call(self, name: str, args: dict):
        w = self.world
        try:
            if name == "get_scene_objects":
                res = {"objects": self._perceive(), "cube_size_m": w.cube, "gripper_holding": w.held is not None}
            elif name == "pick_object":
                obj = args["object_name"]
                r = w.pick(self.names.color(obj))
                res = self._not_detected(obj) if not r["ok"] else {
                    "success": True, "picked_position": [round(v, 4) for v in r["position"]]}
            elif name == "place_on":
                obj = args["object_name"]
                r = w.place_on(self.names.color(obj), args.get("clearance_m"))
                if r["ok"]:
                    res = {"success": True, "still_holding": False,
                           "placed_at": [round(v, 4) for v in r["position"]]}
                elif r["reason"] == "not_holding":
                    res = {"success": False, "error": "a garra nao esta segurando nada"}
                else:
                    res = self._not_detected(obj)
            elif name == "place_at":
                r = w.place_at(float(args["x"]), float(args["y"]), float(args["z"]))
                res = {"success": False, "error": "a garra nao esta segurando nada"} if not r["ok"] else {
                    "success": True, "still_holding": False}
            else:
                return f"ferramenta desconhecida: {name}", True
        except (KeyError, TypeError, ValueError) as e:
            return f"argumentos invalidos para {name}: {e}", True
        return json.dumps(res, indent=2, ensure_ascii=False), False


class FakeBackend:
    kind = "fake"

    def __init__(self, perception_bias_m: float = 0.0025, **_):
        self.bias_m = perception_bias_m
        self.world = None

    def reset(self, spec: dict) -> dict:
        self.world = World(spec["scene"], spec["base_xy"], perturbation=spec.get("perturbation"))
        return self.world.state()

    @asynccontextmanager
    async def tools(self, spec: dict, condition: dict):
        names = NameMap(spec["neutral_map"] if condition.get("neutral_ids") else None)
        yield FakeToolClient(self.world, names, condition.get("scaffolding", "guided"), self.bias_m)

    def finalize(self, spec: dict) -> dict:
        return self.world.state()

    def events(self) -> list:
        return list(self.world.events)

    def capture(self, tag: str, out_dir) -> list:
        return []

    def describe(self) -> dict:
        return {"kind": self.kind, "perception_bias_m": self.bias_m}
