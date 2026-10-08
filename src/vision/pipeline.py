"""Pipeline de visao: duas imagens -> posicoes 3D dos cubos coloridos."""
from .camera import SimCamera
from .color_detect import detect_color_centroid, COLOR_RANGES
from .triangulate import StereoRig


class VisionPipeline:
    def __init__(self, conn, left_alias="/cam_left", right_alias="/cam_right",
                 object_height=0.04):
        """`object_height`: altura dos objetos a detectar (cubos de 4cm).
        As cameras, de cima, enxergam a FACE SUPERIOR do cubo -- interceptar
        o raio da camera no plano dessa face (z=object_height) da X/Y quase
        exatos (e onde o raio realmente bate no objeto); o centro do objeto
        fica so meio-tamanho abaixo (z=object_height/2), entao usamos o X/Y
        dessa intersecao com esse Z conhecido. Interceptar direto no plano do
        CENTRO (em vez do topo) enviesa X/Y, porque o raio nunca passa por
        ali de verdade. Ver `vision.triangulate.backproject_to_z`."""
        self.cam_left = SimCamera(conn, left_alias)
        self.cam_right = SimCamera(conn, right_alias)
        self.rig = StereoRig(self.cam_left, self.cam_right)
        self.object_height = object_height

    def get_object_positions(self):
        """Retorna {cor: (x, y, z)} para cada cor detectada nas DUAS cameras."""
        img_left = self.cam_left.capture_rgb()
        img_right = self.cam_right.capture_rgb()

        positions = {}
        for color in COLOR_RANGES:
            pt_left = detect_color_centroid(img_left, color)
            pt_right = detect_color_centroid(img_right, color)
            if pt_left is None or pt_right is None:
                continue
            top_x, top_y, _ = self.rig.backproject_to_z(
                pt_left, pt_right, self.object_height
            )
            positions[color] = (top_x, top_y, self.object_height / 2)
        return positions
