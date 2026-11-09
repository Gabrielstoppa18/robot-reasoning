"""Agentes-baseline, sem LLM e sem custo de API (rodam no simulador real).

- noop: so le a cena e para. Mede o sucesso "de graca" (deve ser 0 nas
  familias F1-F6 por construcao da cena; em F7 mede o piso de abstencao).
- first_mentioned: heuristica lexical -- pega a primeira cor mencionada e
  poe sobre a segunda. Se um LLM nao superar isso com folga numa familia, o
  sucesso ali nao exige inferencia (controle do vies "mover o primeiro
  mencionado", Secao 6.3.1, principio 3).
- oracle: executa o plano de referencia da instrucao, conhecendo os papeis
  (informacao privilegiada), pelas MESMAS ferramentas e com a MESMA
  percepcao do LLM. Mede o teto de execucao: P(S | plano correto) sem LLM.
"""
import json
import re

from ..geometry import CUBE, DEFAULT_WORKSPACE, base_dist, beside_spot, dist_xy, far_spot, free_spots, in_workspace
from ..palette import COLORS
from .base import Agent, Transcript


class _Scripted(Agent):
    def __init__(self, model=None, **kw):
        super().__init__(model or f"baseline-{self.kind}", **kw)

    async def _call(self, tr, tools, name, args, idx):
        turn = tr.new_turn(idx, 0.0, {}, "tool_use", [{"type": "tool_call", "name": name, "args": args}])
        text, is_error = await tr.call_tool(tools, turn, name, args)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"_raw": text, "_is_error": is_error}

    async def _look(self, tr, tools, idx):
        res = await self._call(tr, tools, "get_scene_objects", {}, idx)
        perceived, names = {}, {}
        for name, e in res.get("objects", {}).items():
            color = e.get("color", name)
            perceived[color] = [e["x"], e["y"], e["z"]]
            names[color] = name
        return perceived, names

    def _end(self, tr, idx, text):
        tr.new_turn(idx, 0.0, {}, "end_turn", [{"type": "text", "text": text}])
        return tr.finish(text)


class NoopAgent(_Scripted):
    kind = "noop"

    async def run(self, prompt, tools, system_prompt, spec=None):
        tr = Transcript(self.kind, self.model, system_prompt, prompt)
        await self._look(tr, tools, 0)
        return self._end(tr, 1, "Nao executei nenhuma acao.")


class FirstMentionedAgent(_Scripted):
    kind = "first_mentioned"

    async def run(self, prompt, tools, system_prompt, spec=None):
        tr = Transcript(self.kind, self.model, system_prompt, prompt)
        perceived, names = await self._look(tr, tools, 0)
        lang = (spec or {}).get("lang", "pt")
        low = prompt.lower()
        hits = []
        for color, info in COLORS.items():
            m = re.search(rf"\b{re.escape(info[lang])}\b", low)
            if m:
                hits.append((m.start(), color))
        mentioned = [c for _, c in sorted(hits) if c in perceived]
        if not mentioned:
            return self._end(tr, 1, "Nao encontrei os objetos mencionados.")
        src = mentioned[0]
        await self._call(tr, tools, "pick_object", {"object_name": names[src]}, 1)
        if len(mentioned) >= 2:
            await self._call(tr, tools, "place_on", {"object_name": names[mentioned[1]]}, 2)
        else:
            others = [p for c, p in perceived.items() if c != src]
            spot = far_spot(perceived[src], others) or perceived[src][:2]
            await self._call(tr, tools, "place_at", {"x": spot[0], "y": spot[1], "z": CUBE / 2}, 2)
        return self._end(tr, 3, "Tarefa concluida.")


class OracleAgent(_Scripted):
    kind = "oracle"

    async def run(self, prompt, tools, system_prompt, spec=None):
        tr = Transcript(self.kind, self.model, system_prompt, prompt)
        idx = 0
        perceived, names = await self._look(tr, tools, idx)
        if (spec or {}).get("expected") == "abstain":
            return self._end(tr, 1, "Nao e possivel executar a tarefa como pedida; nada foi movido.")
        initial = dict(perceived)
        held = None
        line = None
        for step in (spec or {}).get("reference_plan", []):
            idx += 1
            op, args = step[0], step[1:]
            if op == "look":
                perceived, names = await self._look(tr, tools, idx)
                continue
            if op == "pick":
                held = args[0]
                await self._call(tr, tools, "pick_object", {"object_name": names.get(held, held)}, idx)
                continue
            if op == "place_on":
                await self._call(tr, tools, "place_on", {"object_name": names.get(args[0], args[0])}, idx)
                held = None
                continue
            # demais passos: calcula uma coordenada a partir de uma leitura nova
            perceived, names = await self._look(tr, tools, idx)
            occupied = [p for c, p in perceived.items()]
            ref = args[0] if args and isinstance(args[0], str) else None
            if ref is not None and op != "place_at_initial" and ref not in perceived:
                return self._end(tr, idx + 1, f"Nao encontrei o cubo {ref} na cena para {op}.")
            if op == "place_beside":
                ref = perceived[args[0]]
                spot = beside_spot(ref, [p for c, p in perceived.items() if c != args[0]])
            elif op == "place_far":
                spot = far_spot(perceived[args[0]], occupied)
            elif op == "place_temp":
                spots = free_spots(occupied + list(initial.values()), clearance=0.09)
                spot = max(spots, key=lambda p: min(dist_xy(p, q) for q in occupied + list(initial.values()))) \
                    if spots else None
            elif op == "place_at_initial":
                spot = initial[args[0]][:2]
            elif op == "place_closer_than":
                ref, base = perceived[args[0]], spec["base_xy"]
                cands = [p for p in free_spots(occupied, clearance=0.08)
                         if base_dist(ref, base) - 0.15 <= base_dist(p, base) <= base_dist(ref, base) - 0.07]
                spot = min(cands, key=lambda p: dist_xy(p, ref)) if cands else None
            elif op == "place_slot":
                if line is None:
                    n_slots = sum(1 for s in spec["reference_plan"] if s[0] == "place_slot")
                    line = _choose_line(perceived, spec["base_xy"], n_slots)
                spot = line[args[0]] if line else None
            else:
                spot = None
            if spot is None:
                return self._end(tr, idx + 1, f"Nao consegui calcular um ponto para {op}.")
            await self._call(tr, tools, "place_at", {"x": round(spot[0], 4), "y": round(spot[1], 4), "z": CUBE / 2},
                             idx)
            held = None
        return self._end(tr, idx + 1, "Tarefa concluida.")


def _choose_line(perceived, base_xy, n_slots=3, ws=DEFAULT_WORKSPACE):
    """Fila radial a partir da base, com as vagas livres (>= 8 cm dos
    objetos atuais), dentro da area de trabalho. Tenta varios angulos e
    pontos de partida, com vagas espacadas de 10 a 8 cm."""
    import math
    for step in (0.10, 0.09, 0.08):
        for r0 in (0.25, 0.22, 0.28, 0.31):
            radii = [r0 + k * step for k in range(n_slots)]
            for deg in range(5, 85, 2):
                a = math.radians(deg)
                slots = [[base_xy[0] + r * math.cos(a), base_xy[1] + r * math.sin(a)] for r in radii]
                if all(in_workspace(s, ws) for s in slots) \
                        and all(dist_xy(s, p) >= 0.08 for s in slots for p in perceived.values()):
                    return slots
    return None
