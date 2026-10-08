"""Fase 3 - MCP server: expoe o robo (visao + pick-and-place) como
ferramentas para qualquer cliente MCP (Claude, e via adaptador tambem
Gemini/GPT na Fase 4).

Requisitos antes de rodar:
  - CoppeliaSim aberto com sim/phase1_pick_place.ttt carregado
  - venv do projeto (.venv) com `mcp[cli]` instalado

Uso (stdio, para configurar num cliente MCP):
  .venv/Scripts/python.exe scripts/mcp_server.py
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mcp.server.mcpserver import MCPServer

from robot_control import SimConnection, UR5Arm, RG2Gripper, PickPlaceController
from vision import VisionPipeline

CUBE_SIZE = 0.04

mcp = MCPServer(
    name="robot-reasoning",
    instructions=(
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
    ),
)

_state = {}


def _live_tracker(vision, object_name, initial_position):
    """Retorna uma funcao que, a cada chamada, tenta reler a posicao atual
    do objeto via visao -- usada para o braco corrigir a rota se o objeto
    se mover entre um estagio do movimento e outro (ver docstring de
    PickPlaceController). Se o objeto some da visao num instante (ex: ficou
    atras da propria garra por um frame), cai pra ultima posicao conhecida
    em vez de quebrar o movimento."""
    last = {"pos": initial_position}

    def getter():
        positions = vision.get_object_positions()
        if object_name in positions:
            last["pos"] = positions[object_name]
        return last["pos"]

    return getter


def _robot():
    """Inicializa (uma vez) e retorna a conexao/robo/visao, reaproveitando
    entre chamadas de ferramentas -- a simulacao fica rodando o tempo todo."""
    if not _state:
        conn = SimConnection()
        sim = conn.sim
        conn.stop_simulation()
        while sim.getSimulationState() != sim.simulation_stopped:
            pass
        conn.set_stepping(True)
        conn.start_simulation()
        for _ in range(5):
            conn.step()

        _state["conn"] = conn
        _state["arm"] = UR5Arm(conn)
        _state["gripper"] = RG2Gripper(conn)
        _state["controller"] = PickPlaceController(_state["arm"], _state["gripper"])
        _state["vision"] = VisionPipeline(conn)
    return _state


@mcp.tool()
def get_scene_objects() -> dict:
    """Detecta os objetos coloridos na mesa via visao estereo (camera real,
    nao ground truth do simulador). Retorna {nome: {x,y,z}} com o CENTRO de
    cada objeto detectado, em metros, no referencial do mundo. Tambem
    informa o tamanho conhecido dos cubos (todos identicos)."""
    state = _robot()
    positions = state["vision"].get_object_positions()
    return {
        "objects": {
            name: {"x": round(p[0], 4), "y": round(p[1], 4), "z": round(p[2], 4)}
            for name, p in positions.items()
        },
        "cube_size_m": CUBE_SIZE,
        "gripper_holding": state["gripper"].is_holding,
    }


@mcp.tool()
def pick_object(object_name: str) -> dict:
    """Pega um objeto pelo nome (ex: 'red', 'blue' -- os mesmos nomes
    retornados por get_scene_objects). A posicao e re-detectada via visao a
    cada estagio da aproximacao (nao so uma vez no inicio), entao pequenos
    deslocamentos do objeto durante o movimento sao corrigidos. Abre a
    garra, desce por cima do objeto, fecha a garra e sobe. Retorna se
    conseguiu segurar algo."""
    state = _robot()
    positions = state["vision"].get_object_positions()
    if object_name not in positions:
        return {
            "success": False,
            "error": f"objeto '{object_name}' nao detectado. "
                     f"objetos visiveis agora: {list(positions.keys())}",
        }
    tracker = _live_tracker(state["vision"], object_name, positions[object_name])
    final_pos = state["controller"].pick(tracker)
    return {"success": state["gripper"].is_holding, "picked_position": final_pos}


@mcp.tool()
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
    state["controller"].place([x, y, z])
    return {"success": True, "still_holding": state["gripper"].is_holding}


@mcp.tool()
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
    if object_name not in positions:
        return {
            "success": False,
            "error": f"objeto '{object_name}' nao detectado. "
                     f"objetos visiveis agora: {list(positions.keys())}",
        }

    base_tracker = _live_tracker(state["vision"], object_name, positions[object_name])

    def target_getter():
        base = base_tracker()
        return [base[0], base[1], base[2] + clearance_m]

    final_pos = state["controller"].place(target_getter)
    return {"success": True, "still_holding": state["gripper"].is_holding, "placed_at": final_pos}


if __name__ == "__main__":
    mcp.run(transport="stdio")
