"""Backend real: CoppeliaSim + servidor MCP (scripts/mcp_server.py).

O harness tem conexao ZMQ PROPRIA com o simulador e e o unico que mexe na
cena entre episodios (o agente nao tem ferramenta de reinicio):

  reset     simulacao parada -> remove os cubos do episodio anterior -> cria
            os cubos da semente -> roda alguns passos para assentar -> le a
            verdade-terreno inicial e grava as imagens "antes" -> para.
  tools     sobe um servidor MCP NOVO por episodio (processo stdio), com o
            modo benchmark nas variaveis de ambiente; o servidor reinicia a
            simulacao no primeiro uso, como sempre fez.
  finalize  depois que o servidor fecha: alguns passos para a fisica
            assentar (cubo solto cai, pilha torta tomba) -> verdade-terreno
            final -> imagens "depois" -> para a simulacao.

Cuidado: com stepping ligado num cliente ZMQ, o simulador espera esse
cliente; por isso o harness DESLIGA o proprio stepping antes de entregar a
cena ao servidor.
"""
import json
import os
import sys
import tempfile
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path

import numpy as np

from ..geometry import CUBE
from ..palette import COLORS, PALETTE, cube_alias

_CAMERAS = ("/cam_left", "/cam_right", "/cam_overview")
_LEGACY_ALIASES = ("/red_cube", "/blue_cube")


class MCPToolClient:
    def __init__(self, session, tools):
        self.session = session
        self.tools = tools

    async def call(self, name: str, args: dict):
        result = await self.session.call_tool(name, args)
        parts = [c.text for c in result.content if getattr(c, "type", None) == "text"]
        return ("\n".join(parts) if parts else "(sem conteudo)"), bool(getattr(result, "is_error", False))


class CoppeliaBackend:
    kind = "coppelia"

    def __init__(self, repo_root, python_executable=None, settle_steps=20, host="localhost", port=23000, **_):
        self.repo_root = Path(repo_root)
        self.python_executable = python_executable or sys.executable
        self.server_script = str(self.repo_root / "scripts" / "mcp_server.py")
        self.settle_steps = settle_steps
        self.host, self.port = host, port
        self.image_dir = None  # o runner define por episodio
        self._client = None
        self._sim = None
        self._event_log = None
        self.base_xy = None

    # ---- conexao ----------------------------------------------------------
    @property
    def sim(self):
        if self._sim is None:
            from coppeliasim_zmqremoteapi_client import RemoteAPIClient
            self._client = RemoteAPIClient(self.host, self.port)
            self._sim = self._client.require("sim")
        return self._sim

    def _stop(self):
        sim = self.sim
        sim.stopSimulation()
        while sim.getSimulationState() != sim.simulation_stopped:
            pass

    def _run_steps(self, n):
        sim = self.sim
        sim.setStepping(True)
        if sim.getSimulationState() == sim.simulation_stopped:
            sim.startSimulation()
        for _ in range(n):
            sim.step()

    # ---- cena -------------------------------------------------------------
    def _remove_cubes(self):
        sim = self.sim
        for alias in {cube_alias(c) for c in PALETTE} | set(_LEGACY_ALIASES):
            h = sim.getObject(alias, {"noError": True})
            if h != -1:
                sim.removeObject(h)

    def _create_cube(self, color, pos):
        sim = self.sim
        h = sim.createPrimitiveShape(sim.primitiveshape_cuboid, [CUBE, CUBE, CUBE], 1)
        sim.setObjectAlias(h, cube_alias(color)[1:])
        sim.setObjectPose(h, [pos[0], pos[1], CUBE / 2, 0, 0, 0, 1], -1)
        sim.setObjectInt32Param(h, sim.shapeintparam_static, 0)
        sim.setObjectInt32Param(h, sim.shapeintparam_respondable, 1)
        sim.setShapeColor(h, None, sim.colorcomponent_ambient_diffuse, COLORS[color]["rgb"])
        sim.computeMassAndInertia(h, 500)
        # o cubo nasce na origem (dentro da base do robo); se a fisica rodar
        # antes do setObjectPose, ele e expulso e guarda a velocidade
        sim.resetDynamicObject(h)

    def snapshot(self, colors) -> dict:
        sim = self.sim
        objects, held = {}, None
        for c in colors:
            h = sim.getObject(cube_alias(c), {"noError": True})
            if h == -1:
                continue
            objects[c] = [round(v, 5) for v in sim.getObjectPosition(h, -1)]
            if sim.getObjectParent(h) != -1:  # anexado a garra
                held = c
        return {"objects": objects, "held": held, "base_xy": self.base_xy, "cube": CUBE}

    def capture(self, tag: str, out_dir=None) -> list:
        out_dir = out_dir or self.image_dir
        if not out_dir:
            return []
        import cv2
        sim, paths = self.sim, []
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        for alias in _CAMERAS:
            h = sim.getObject(alias, {"noError": True})
            if h == -1:
                continue
            img, res = sim.getVisionSensorImg(h)
            arr = np.flipud(np.frombuffer(img, dtype=np.uint8).reshape(res[1], res[0], 3))
            p = Path(out_dir) / f"{tag}_{alias.strip('/')}.png"
            cv2.imwrite(str(p), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
            paths.append(str(p))
        return paths

    def _arm_to_ready(self):
        """Leva o braco a pose 'ready' ANTES de criar os cubos. A transicao
        varre o espaco (os elos nao teletransportam na fisica) e chegava a
        derrubar cubos -- depois da conferencia da cena inicial, o que faria a
        falha cair na conta do modelo. O servidor MCP, em modo benchmark,
        encontra a simulacao rodando e o braco ja na pose, e nao o move."""
        from robot_control import SimConnection, UR5Arm
        conn = SimConnection(self.host, self.port)
        conn.set_stepping(True)
        if conn.sim.getSimulationState() == conn.sim.simulation_stopped:
            conn.start_simulation()
        for _ in range(5):
            conn.step()
        UR5Arm(conn)
        conn.set_stepping(False)

    def reset(self, spec: dict) -> dict:
        sim = self.sim
        self._stop()
        self._remove_cubes()
        self._arm_to_ready()
        sim.setStepping(True)  # nenhum passo de fisica entre criar e posicionar cada cubo
        for color, pos in spec["scene"].items():
            self._create_cube(color, pos)
        self.base_xy = [round(v, 4) for v in sim.getObjectPosition(sim.getObject("/UR5"), -1)[:2]]
        # o passo a passo ja esta ligado: _run_steps ligaria de novo, e a API
        # conta as ativacoes -- um setStepping(False) so nao desligaria, e o
        # servidor MCP ficava travado esperando este cliente avancar
        for _ in range(5):
            sim.step()
        init = self.snapshot(spec["scene"])
        self.capture("before")
        sim.setStepping(False)  # a simulacao segue rodando; o servidor MCP assume o passo a passo
        self._event_log = Path(tempfile.mkstemp(prefix="bench_events_", suffix=".jsonl")[1])
        return init

    @asynccontextmanager
    async def tools(self, spec: dict, condition: dict):
        from mcp import ClientSession
        from mcp.client.stdio import StdioServerParameters, get_default_environment, stdio_client

        env = get_default_environment()  # sem as chaves de API: o servidor nao precisa delas
        env["BENCH_SCAFFOLDING"] = condition.get("scaffolding", "guided")
        env["BENCH_EVENT_LOG"] = str(self._event_log)
        if condition.get("neutral_ids"):
            env["BENCH_NEUTRAL_MAP"] = json.dumps(spec["neutral_map"])
        if spec.get("perturbation"):
            env["BENCH_PERTURBATION"] = json.dumps(spec["perturbation"])
        params = StdioServerParameters(command=self.python_executable, args=[self.server_script],
                                       env=env, cwd=str(self.repo_root))
        async with AsyncExitStack() as stack:
            read, write = await stack.enter_async_context(stdio_client(params))
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            listed = await session.list_tools()
            tools = [{"name": t.name, "description": t.description or "", "input_schema": t.input_schema}
                     for t in listed.tools]
            yield MCPToolClient(session, tools)

    def finalize(self, spec: dict) -> dict:
        self._run_steps(self.settle_steps)
        final = self.snapshot(spec["scene"])
        self.capture("after")
        self._stop()
        self.sim.setStepping(False)
        return final

    def events(self) -> list:
        if not self._event_log or not self._event_log.exists():
            return []
        lines = self._event_log.read_text(encoding="utf-8").splitlines()
        try:
            os.unlink(self._event_log)
        except OSError:
            pass
        self._event_log = None
        return [json.loads(l) for l in lines if l.strip()]

    def describe(self) -> dict:
        sim = self.sim
        return {"kind": self.kind, "coppeliasim_version": sim.getInt32Param(sim.intparam_program_full_version),
                "settle_steps": self.settle_steps, "base_xy": self.base_xy}
