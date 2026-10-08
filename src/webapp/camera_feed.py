"""Conexao leve e independente com o CoppeliaSim, so para capturar frames
das cameras para o dashboard (nao passa pelo MCP -- o MCP so expoe
posicoes numericas, nao imagens cruas)."""
import cv2

from robot_control import SimConnection
from vision.camera import SimCamera

_CAMERA_ALIASES = ["cam_left", "cam_right", "cam_overview"]


class CameraFeed:
    def __init__(self):
        self.conn = SimConnection()
        sim = self.conn.sim
        # so garante que a simulacao esta rodando -- nunca para/reinicia
        # (isso e responsabilidade de quem possui a sessao do agente; aqui
        # a gente so precisa que os vision sensors estejam renderizando).
        if sim.getSimulationState() == sim.simulation_stopped:
            self.conn.set_stepping(False)
            self.conn.start_simulation()

        self.cameras = {}
        for alias in _CAMERA_ALIASES:
            handle = sim.getObject(f"/{alias}", {"noError": True})
            if handle != -1:
                self.cameras[alias] = SimCamera(self.conn, f"/{alias}")

    def available_cameras(self) -> list[str]:
        return list(self.cameras.keys())

    def snapshot_jpeg(self, camera_name: str) -> bytes | None:
        cam = self.cameras.get(camera_name)
        if cam is None:
            return None
        rgb = cam.capture_rgb()
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            return None
        return buf.tobytes()
