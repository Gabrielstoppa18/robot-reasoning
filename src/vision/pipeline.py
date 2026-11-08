"""Pipeline de visao: duas imagens -> posicoes 3D dos cubos coloridos."""
from .camera import SimCamera
from .color_detect import detect_color_blob, COLOR_RANGES
from .triangulate import StereoRig, backproject_to_z


_TRI_Z_TABLE = 0.034  # z triangulado de um cubo de 4 cm apoiado na mesa
_TRI_Z_PER_LEVEL = 0.033  # acrescimo por andar de pilha
_MAX_LEVEL = 3


class VisionPipeline:
    def __init__(self, conn, left_alias="/cam_left", right_alias="/cam_right",
                 object_height=0.04, ray_plane_z=0.0254):
        """`object_height`: altura dos objetos a detectar (cubos de 4cm).
        As cameras, de cima, enxergam a FACE SUPERIOR do cubo -- interceptar
        o raio da camera no plano dessa face (z=object_height) da X/Y quase
        exatos (e onde o raio realmente bate no objeto); o centro do objeto
        fica so meio-tamanho abaixo (z=object_height/2), entao usamos o X/Y
        dessa intersecao com esse Z conhecido. Interceptar direto no plano do
        CENTRO (em vez do topo) enviesa X/Y, porque o raio nunca passa por
        ali de verdade. Ver `vision.triangulate.backproject_to_z`.

        `ray_plane_z`: altura do plano onde o raio e interceptado. Na pratica
        a mancha de cor inclui parte das faces laterais (as cameras veem o
        cubo um pouco de lado), e o centroide fica abaixo do topo. Calibrado
        no portao de percepcao (2026-10-09, 136 cubos em 40 cenas): no topo
        (4 cm) o erro era radial, ~2,5 % em torno do nadir das cameras (media
        3,8 mm, max 7,7 mm); com 2,54 cm, media 1,0 mm e max 2,0 mm."""
        self.cam_left = SimCamera(conn, left_alias)
        self.cam_right = SimCamera(conn, right_alias)
        self.rig = StereoRig(self.cam_left, self.cam_right)
        self.object_height = object_height
        self.ray_plane_z = ray_plane_z

    def get_object_detections(self):
        """{cor: ((x, y, z), area_esq, area_dir)}; area 0 = nao visto naquela camera."""
        img_left = self.cam_left.capture_rgb()
        img_right = self.cam_right.capture_rgb()
        out = {}
        for color in COLOR_RANGES:
            bl, br = detect_color_blob(img_left, color), detect_color_blob(img_right, color)
            pos = self._locate(bl[:2] if bl else None, br[:2] if br else None)
            if pos is not None:
                out[color] = (pos, bl[2] if bl else 0.0, br[2] if br else 0.0)
        return out

    def _level(self, pt_left, pt_right):
        """Andar do cubo (0 = na mesa) pelo z da triangulacao estereo. O z
        triangulado e ruidoso para medir posicao, mas separa bem os andares:
        medido em 2026-10-10, ~0,034 m na mesa e ~0,100 m no topo de uma
        pilha de 3, ou ~0,033 m por andar."""
        z = self.rig.triangulate(pt_left, pt_right)[2]
        return min(range(_MAX_LEVEL + 1), key=lambda k: abs(z - (_TRI_Z_TABLE + k * _TRI_Z_PER_LEVEL)))

    def _locate(self, pt_left, pt_right):
        if pt_left is None and pt_right is None:
            return None
        if pt_left is not None and pt_right is not None:
            k = self._level(pt_left, pt_right)
            top_x, top_y, _ = self.rig.backproject_to_z(pt_left, pt_right, self.ray_plane_z + k * self.object_height)
        else:
            # uma camera so (a outra tapada pelo braco): sem estereo, assume a mesa
            k = 0
            cam, pt = (self.cam_left, pt_left) if pt_left is not None else (self.cam_right, pt_right)
            top_x, top_y, _ = backproject_to_z(cam, pt, self.ray_plane_z)
        return (top_x, top_y, self.object_height / 2 + k * self.object_height)

    def get_object_positions(self):
        """Retorna {cor: (x, y, z)} para cada cor detectada. Com as duas
        cameras, usa a media das duas intersecoes; se o braco tapar uma delas,
        usa so a outra (a intersecao com o plano conhecido nao precisa de par
        estereo)."""
        return {c: d[0] for c, d in self.get_object_detections().items()}
