"""Wrapper de um vision sensor do CoppeliaSim: captura de imagem + intrinsecos."""
import math
import numpy as np


class SimCamera:
    def __init__(self, conn, alias):
        self.sim = conn.sim
        self.conn = conn
        self.handle = self.sim.getObject(alias)

        # le resolucao/FOV direto do sensor (em vez de supor um valor fixo
        # aqui, que facilmente desalinha do que foi configurado na cena)
        res_x = self.sim.getObjectInt32Param(self.handle, self.sim.visionintparam_resolution_x)
        res_y = self.sim.getObjectInt32Param(self.handle, self.sim.visionintparam_resolution_y)
        fov = self.sim.getObjectFloatParam(self.handle, self.sim.visionfloatparam_perspective_angle)
        self.resolution = (res_x, res_y)

        # sensor quadrado -> mesma FOV nos dois eixos, fx == fy
        f = (res_x / 2) / math.tan(fov / 2)
        self.fx = f
        self.fy = f
        self.cx = res_x / 2
        self.cy = res_y / 2

    @property
    def intrinsics(self):
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1],
        ])

    def capture_rgb(self):
        img, res = self.sim.getVisionSensorImg(self.handle)
        arr = np.frombuffer(img, dtype=np.uint8).reshape(res[1], res[0], 3)
        return np.flipud(arr)  # a imagem do CoppeliaSim vem invertida no eixo Y

    def world_pose(self):
        """[x, y, z, qx, qy, qz, qw] do sensor no referencial do mundo."""
        return self.sim.getObjectPose(self.handle, -1)
