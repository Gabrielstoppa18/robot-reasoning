"""Triangulacao estereo: 2 pontos 2D (um por camera) -> 1 ponto 3D no mundo.

As poses das cameras vem direto do simulador (extrinsecos conhecidos/"ground
truth"), o que e razoavel nesta fase -- calibracao de camera real fica para
quando o pipeline for portado para hardware fisico.
"""
import numpy as np
import cv2


def _quat_to_rotmat(qx, qy, qz, qw):
    n = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx ** 2 + qy ** 2)],
    ])


# Convencao do vision sensor do CoppeliaSim determinada empiricamente
# (projetando um ponto de ground truth e comparando com o pixel detectado):
# local +Z aponta "para frente" (igual OpenCV), mas X e Y vem invertidos em
# relacao ao referencial de camera do OpenCV.
_CS_TO_CV = np.diag([-1.0, -1.0, 1.0])


def _projection_matrix(camera):
    x, y, z, qx, qy, qz, qw = camera.world_pose()
    r_wc = _quat_to_rotmat(qx, qy, qz, qw)  # camera -> mundo
    t_wc = np.array([x, y, z])

    r_cv = _CS_TO_CV @ r_wc.T
    t_cv = _CS_TO_CV @ (-r_wc.T @ t_wc)

    extrinsics = np.hstack([r_cv, t_cv.reshape(3, 1)])
    return camera.intrinsics @ extrinsics


def backproject_to_z(camera, pixel, z_world):
    """Interseca o raio da camera (passando pelo pixel) com o plano
    horizontal z=z_world. Mais robusto que triangulacao estereo pura aqui:
    o objeto detectado e o topo de um cubo (nao um ponto), e a visao
    levemente obliqua de cada camera faz o centroide do topo nao coincidir
    com o centro do objeto -- isso desloca muito a profundidade triangulada
    (baseline pequeno/distancia grande amplifica esse erro), mas quase nao
    afeta X/Y. Como sabemos a altura da mesa (ou do objeto apoiado nela),
    interceptar o raio nesse plano conhecido da uma estimativa muito mais
    precisa do que confiar na triangulacao de profundidade.
    """
    x, y, z, qx, qy, qz, qw = camera.world_pose()
    r_wc = _quat_to_rotmat(qx, qy, qz, qw)
    cam_pos = np.array([x, y, z])

    u, v = pixel
    d_cv = np.array([(u - camera.cx) / camera.fx, (v - camera.cy) / camera.fy, 1.0])
    d_local = _CS_TO_CV @ d_cv  # _CS_TO_CV e sua propria inversa (diagonal +-1)
    d_world = r_wc @ d_local

    if abs(d_world[2]) < 1e-9:
        return None
    t = (z_world - cam_pos[2]) / d_world[2]
    point = cam_pos + t * d_world
    return tuple(point)


class StereoRig:
    def __init__(self, cam_left, cam_right):
        self.cam_left = cam_left
        self.cam_right = cam_right
        self.p_left = _projection_matrix(cam_left)
        self.p_right = _projection_matrix(cam_right)

    def triangulate(self, pt_left, pt_right):
        """pt_left/pt_right: (px, py). Retorna (x, y, z) no mundo.

        Triangulacao estereo "pura" (sem assumir plano/altura conhecida).
        Ver `backproject_to_z` para a alternativa usada no pipeline, mais
        precisa para objetos pequenos apoiados numa superficie conhecida.
        """
        pts_l = np.array([[pt_left[0]], [pt_left[1]]], dtype=np.float64)
        pts_r = np.array([[pt_right[0]], [pt_right[1]]], dtype=np.float64)
        homogeneous = cv2.triangulatePoints(self.p_left, self.p_right, pts_l, pts_r)
        point = (homogeneous[:3] / homogeneous[3]).flatten()
        return tuple(point)

    def backproject_to_z(self, pt_left, pt_right, z_world):
        """Media das intersecoes (raio de cada camera) com o plano z=z_world."""
        p_l = backproject_to_z(self.cam_left, pt_left, z_world)
        p_r = backproject_to_z(self.cam_right, pt_right, z_world)
        return tuple((np.array(p_l) + np.array(p_r)) / 2.0)
