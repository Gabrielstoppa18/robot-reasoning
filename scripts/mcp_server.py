"""Fase 3 - MCP server: expoe o robo (visao + pick-and-place) como
ferramentas para qualquer cliente MCP (Claude, e via adaptador tambem
Gemini/GPT na Fase 4).

Requisitos antes de rodar:
  - CoppeliaSim aberto com sim/phase1_pick_place.ttt carregado
  - venv do projeto (.venv) com `mcp[cli]` instalado

Uso (stdio, para configurar num cliente MCP):
  .venv/Scripts/python.exe scripts/mcp_server.py
"""
import json
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mcp.server.mcpserver import MCPServer

from robot_control import SimConnection, UR5Arm, RG2Gripper, PickPlaceController, UnreachablePose
from robot_control.pick_place import yaw_away_from
from vision import VisionPipeline
from benchmark.palette import PALETTE, cube_alias
from benchmark.tool_specs import tool_description

CUBE_SIZE = 0.04

# --- modo benchmark (scripts/bench_run.py), configurado por variaveis de
# ambiente; sem elas o servidor se comporta exatamente como antes ---------
#  BENCH_SCAFFOLDING   guided (default) | minimal -- descricoes das ferramentas
#  BENCH_NEUTRAL_MAP   JSON {"obj_1": "red", ...} -- identificadores neutros
#  BENCH_PERTURBATION  JSON {"object": "red", "delta": [dx, dy], "call_index": 0, "step": 50}:
#                      desloca o objeto no passo `step` da `call_index`-esima
#                      chamada place_* que chegou a mover o braco
#  BENCH_EVENT_LOG     JSONL de eventos para o harness (pose verdadeira na
#                      soltura, perturbacao) -- o modelo nunca ve isso
_SCAFFOLDING = os.environ.get("BENCH_SCAFFOLDING", "guided")
_NEUTRAL_MAP = json.loads(os.environ.get("BENCH_NEUTRAL_MAP") or "{}")
_COLOR_TO_ID = {c: i for i, c in _NEUTRAL_MAP.items()}
_PERTURBATION = json.loads(os.environ.get("BENCH_PERTURBATION") or "null")
_EVENT_LOG = os.environ.get("BENCH_EVENT_LOG")

_bench = {"steps_total": 0, "place_calls": 0, "in_place": False, "place_steps": 0, "applied": False}


def _desc(tool_name):
    return tool_description(tool_name, _SCAFFOLDING, neutral_ids=bool(_NEUTRAL_MAP))


def _to_color(name):
    return _NEUTRAL_MAP.get(name, name) if _NEUTRAL_MAP else name


def _to_name(color):
    return _COLOR_TO_ID.get(color, color) if _NEUTRAL_MAP else color


def _log_event(event: dict):
    if not _EVENT_LOG:
        return
    event = {"t": time.time(), "sim_step": _bench["steps_total"], **event}
    with open(_EVENT_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


_GUIDED_INSTRUCTIONS = (
        "Controla um braco UR5 (garra RG2) numa cena do CoppeliaSim com "
        "cubos coloridos sobre uma mesa. Use get_scene_objects() primeiro "
        "para ver o que existe e suas coordenadas (metros, referencial do "
        "mundo, sempre o CENTRO do objeto). Depois pick_object(nome) para "
        "pegar um objeto. Para soltar: se for colocar em cima de outro "
        "objeto que AINDA ESTA VISIVEL na mesa, use place_on(nome) -- ele "
        "re-confirma a posicao do alvo durante o proprio movimento, entao "
        "continua certo mesmo se esse objeto for deslocado depois que voce "
        "o viu. So use place_at(x,y,z) para uma coordenada arbitraria/vazia "
        "da mesa, sem objeto de referencia -- nesse caso a posicao fica "
        "congelada no que voce calculou, sem nenhuma checagem posterior. "
        "Atencao: o objeto que esta na garra PARA DE APARECER em "
        "get_scene_objects() (a camera nao enxerga dentro da garra) -- "
        "guarde o que voce pegou de memoria."
)

mcp = MCPServer(
    name="robot-reasoning",
    instructions=_GUIDED_INSTRUCTIONS if _SCAFFOLDING == "guided" else (
        "Controla um braco robotico numa cena com cubos coloridos sobre uma mesa."
    ),
)

_state = {}


def _live_tracker(vision, object_name, initial_position):
    """Retorna uma funcao que, a cada chamada, tenta reler a posicao atual
    do objeto via visao -- usada para o braco corrigir a rota se o objeto
    se mover entre um estagio do movimento e outro (ver docstring de
    PickPlaceController). So aceita a leitura nova se o objeto aparece
    INTEIRO nas duas cameras (area >= 80 % da leitura limpa do inicio, com o
    braco estacionado): com o braco por cima, o blob fica parcialmente coberto
    e o centroide desloca (~2 cm). Senao, fica com a ultima posicao boa."""
    ref = vision.get_object_detections().get(object_name)
    ref_areas = (ref[1], ref[2]) if ref else (0.0, 0.0)
    last = {"pos": initial_position}

    def getter():
        det = vision.get_object_detections().get(object_name)
        if det and min(ref_areas) > 0 and det[1] >= 0.8 * ref_areas[0] and det[2] >= 0.8 * ref_areas[1]:
            last["pos"] = det[0]
        return last["pos"]

    return getter




PARK_POSITION = [0.35, -0.20, 0.30]  # ao lado da mesa (y < 0), fora da imagem das duas cameras


def _robot():
    """Inicializa (uma vez) e retorna a conexao/robo/visao, reaproveitando
    entre chamadas de ferramentas -- a simulacao fica rodando o tempo todo."""
    if not _state:
        conn = SimConnection()
        sim = conn.sim
        # no benchmark, o harness ja deixou a simulacao rodando com o braco na
        # pose 'ready' e os cubos conferidos: reiniciar aqui moveria o braco
        # com os cubos na mesa (benchmark/backends/coppelia.py, _arm_to_ready)
        bench_running = bool(os.environ.get("BENCH_EVENT_LOG")) and sim.getSimulationState() != sim.simulation_stopped
        if not bench_running:
            conn.stop_simulation()
            while sim.getSimulationState() != sim.simulation_stopped:
                pass
        conn.set_stepping(True)
        if not bench_running:
            conn.start_simulation()
        for _ in range(5):
            conn.step()

        _state["conn"] = conn
        _state["arm"] = UR5Arm(conn)
        _state["gripper"] = RG2Gripper(conn)
        # estaciona fora da vista das cameras depois de cada pick/place
        _state["controller"] = PickPlaceController(_state["arm"], _state["gripper"], park_position=PARK_POSITION)
        _state["vision"] = VisionPipeline(conn)
        _install_bench_hooks(_state)
    return _state


def _install_bench_hooks(state):
    """Conta passos de simulacao (para a perturbacao cair num passo especifico
    do movimento, e nao num instante de relogio) e registra a pose VERDADEIRA
    do objeto no instante em que a garra abre (separa erro de controle de
    erro de fisica na atribuicao de falhas)."""
    conn, sim, gripper = state["conn"], state["conn"].sim, state["gripper"]
    orig_step, orig_open = conn.step, gripper.open
    color_of = {}
    for color in PALETTE:
        h = sim.getObject(cube_alias(color), {"noError": True})
        if h != -1:
            color_of[h] = color

    def step():
        orig_step()
        _bench["steps_total"] += 1
        if not _bench["in_place"] or _PERTURBATION is None or _bench["applied"]:
            return
        _bench["place_steps"] += 1
        if (_bench["place_calls"] == _PERTURBATION.get("call_index", 0)
                and _bench["place_steps"] == _PERTURBATION.get("step", 50)):
            h = sim.getObject(cube_alias(_PERTURBATION["object"]), {"noError": True})
            if h != -1 and h != gripper._attached_shape:
                p = sim.getObjectPosition(h, -1)
                new = [p[0] + _PERTURBATION["delta"][0], p[1] + _PERTURBATION["delta"][1], p[2]]
                sim.setObjectPosition(h, new, -1)
                _bench["applied"] = True
                _log_event({"type": "perturbation", "object": _PERTURBATION["object"], "pose": new})

    def open_():
        h = gripper._attached_shape
        if h is not None:
            _log_event({"type": "release", "object": color_of.get(h, str(h)),
                        "pose": list(sim.getObjectPosition(h, -1))})
        orig_open()

    conn.step = step
    gripper.open = open_


class _PlaceMotion:
    """Marca o trecho de uma chamada place_* em que o braco se move."""

    def __enter__(self):
        _bench["in_place"], _bench["place_steps"] = True, 0

    def __exit__(self, *exc):
        _bench["in_place"] = False
        _bench["place_calls"] += 1


@mcp.tool(description=_desc("get_scene_objects"))
def get_scene_objects() -> dict:
    """Detecta os objetos coloridos na mesa via visao estereo (camera real,
    nao ground truth do simulador). Retorna {nome: {x,y,z}} com o CENTRO de
    cada objeto detectado, em metros, no referencial do mundo. Tambem
    informa o tamanho conhecido dos cubos (todos identicos).
    (O texto que o modelo ve vem de benchmark/tool_specs.py.)"""
    state = _robot()
    positions = state["vision"].get_object_positions()
    objects = {}
    for color, p in positions.items():
        entry = {"x": round(p[0], 4), "y": round(p[1], 4), "z": round(p[2], 4)}
        if _NEUTRAL_MAP:
            entry["color"] = color
        objects[_to_name(color)] = entry
    return {
        "objects": objects,
        "cube_size_m": CUBE_SIZE,
        "gripper_holding": state["gripper"].is_holding,
    }


def _unreachable(e):
    # o braco planeja antes de mover (robot_control.arm.UR5Arm.plan) e nao se
    # mexe se o caminho nao tiver solucao -- antes, um alvo fora do alcance
    # levava o IK a comandar angulos absurdos e o braco "desmontava"
    return {"success": False, "error": f"posicao fora do alcance do braco: {e}"}


def _not_detected(object_name, positions):
    return {
        "success": False,
        "error": f"objeto '{object_name}' nao detectado. "
                 f"objetos visiveis agora: {[_to_name(c) for c in positions]}",
    }


@mcp.tool(description=_desc("pick_object"))
def pick_object(object_name: str) -> dict:
    """Pega um objeto pelo nome (ex: 'red', 'blue' -- os mesmos nomes
    retornados por get_scene_objects). A posicao e re-detectada via visao a
    cada estagio da aproximacao (nao so uma vez no inicio), entao pequenos
    deslocamentos do objeto durante o movimento sao corrigidos. Abre a
    garra, desce por cima do objeto, fecha a garra e sobe. Retorna se
    conseguiu segurar algo."""
    state = _robot()
    positions = state["vision"].get_object_positions()
    color = _to_color(object_name)
    if color not in positions:
        return _not_detected(object_name, positions)
    tracker = _live_tracker(state["vision"], color, positions[color])
    # dedos abrindo perpendiculares ao vizinho mais proximo (nao o empurram)
    yaw = yaw_away_from(positions[color], [p for c, p in positions.items() if c != color])
    try:
        final_pos = state["controller"].pick(tracker, yaw)
    except UnreachablePose as e:
        return _unreachable(e)
    return {"success": state["gripper"].is_holding, "picked_position": final_pos}


@mcp.tool(description=_desc("place_at"))
def place_at(x: float, y: float, z: float) -> dict:
    """Solta o objeto atualmente preso numa coordenada FIXA (x, y, z) em
    metros, referencial do mundo -- z e o CENTRO que o objeto deve ocupar.
    Essa coordenada NAO e re-verificada contra nada: se voce a calculou a
    partir de uma leitura antiga de get_scene_objects() e o cenario mudou
    desde entao (ex: outro objeto se moveu), o robo solta no lugar errado.
    Para empilhar/alinhar com um objeto que ainda esta visivel na mesa,
    prefira place_on(), que reconfirma a posicao do objeto-alvo durante o
    proprio movimento. Precisa estar segurando algo."""
    state = _robot()
    if not state["gripper"].is_holding:
        return {"success": False, "error": "a garra nao esta segurando nada"}
    yaw = yaw_away_from([x, y], list(state["vision"].get_object_positions().values()))
    try:
        with _PlaceMotion():
            state["controller"].place([x, y, z], yaw)
    except UnreachablePose as e:
        return _unreachable(e)
    return {"success": True, "still_holding": state["gripper"].is_holding}


@mcp.tool(description=_desc("place_on"))
def place_on(object_name: str, clearance_m: float = CUBE_SIZE) -> dict:
    """Solta o objeto atualmente preso EM CIMA do objeto `object_name`
    (ex: 'red'), a `clearance_m` acima do centro dele (default = altura de
    um cubo, 4cm -- ajuste se o objeto embaixo for maior/menor). Ao
    contrario de place_at, a posicao do objeto-alvo e RE-LIDA da camera a
    cada estagio do movimento (acima -> hover -> descida), entao se ele for
    deslocado depois que voce chamou get_scene_objects() -- ou ate durante
    o proprio movimento de place -- o braco mira na posicao mais recente,
    nao numa coordenada congelada. Precisa estar segurando algo, e o
    objeto-alvo precisa estar visivel (nao pode ser o que esta na garra)."""
    state = _robot()
    if not state["gripper"].is_holding:
        return {"success": False, "error": "a garra nao esta segurando nada"}
    positions = state["vision"].get_object_positions()
    color = _to_color(object_name)
    if color not in positions:
        return _not_detected(object_name, positions)

    base_tracker = _live_tracker(state["vision"], color, positions[color])

    def target_getter():
        base = base_tracker()
        return [base[0], base[1], base[2] + clearance_m]

    yaw = yaw_away_from(positions[color], [p for c, p in positions.items() if c != color])
    try:
        with _PlaceMotion():
            final_pos = state["controller"].place(target_getter, yaw)
    except UnreachablePose as e:
        return _unreachable(e)
    return {"success": True, "still_holding": state["gripper"].is_holding, "placed_at": final_pos}


def _release_stepping():
    """Desliga o passo a passo deste cliente ao sair. A API conta as
    ativacoes, e com mais de um cliente em modo passo a passo a simulacao so
    avanca quando todos avancam: um servidor que sai sem desligar deixa um
    cliente 'morto' registrado, e o harness (finalize, proximo reset) ficava
    esperando por ele -- episodios de 10 a 16 min ou travados de vez."""
    if not _state:
        return
    try:
        level = _state["conn"].sim.setStepping(False)
        while level and level > 1:
            level = _state["conn"].sim.setStepping(False)
    except Exception:  # noqa: BLE001 -- conexao ja fechada: nada a liberar
        pass


if __name__ == "__main__":
    import atexit
    atexit.register(_release_stepping)
    mcp.run(transport="stdio")
