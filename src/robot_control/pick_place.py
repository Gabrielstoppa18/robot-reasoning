"""Sequencia de alto nivel: pick(posicao) / place(posicao).

`position` e sempre o CENTRO do objeto (ex: centro do cubo). O tip (ponto
attachPoint da garra) e levado ate essa mesma altura para um grasp
centralizado nos dedos.
"""

import math

# orientacao com a garra apontando para baixo (180 graus em torno de X)
DOWN_ORIENTATION = [1, 0, 0, 0]


def down_with_yaw(yaw):
    """Garra para baixo, girada de `yaw` rad em torno do proprio eixo. Medido
    na cena: o eixo de abertura dos dedos fica a (180 graus - yaw) no mundo."""
    return [math.cos(yaw / 2), -math.sin(yaw / 2), 0.0, 0.0]  # DOWN * Rz(yaw), [x, y, z, w]


def yaw_away_from(target_xy, neighbors_xy, radius=0.15):
    """Yaw que deixa os dedos abrindo PERPENDICULARES a direcao do vizinho
    mais proximo (num raio de `radius`). Abertos, os dedos ocupam ~10,5 cm:
    alinhados com um cubo a 6 cm, empurravam o vizinho ao pegar e ao soltar.
    Sem vizinho perto, 0. Resultado em (-pi/2, pi/2] (menor giro do punho)."""
    near = [q for q in neighbors_xy if math.dist(q[:2], target_xy[:2]) <= radius]
    if not near:
        return 0.0
    q = min(near, key=lambda n: math.dist(n[:2], target_xy[:2]))
    theta = math.atan2(q[1] - target_xy[1], q[0] - target_xy[0])
    yaw = math.pi / 2 - theta
    while yaw > math.pi / 2:
        yaw -= math.pi
    while yaw <= -math.pi / 2:
        yaw += math.pi
    return yaw

_APPROACH_HEIGHT = 0.15  # altura acima do objeto no waypoint de aproximacao
_HOVER_HEIGHT = 0.05  # waypoint intermediario logo acima do objeto
_FINAL_DESCEND_STEPS = 10  # pontos do plano no trecho final (hover->objeto): 5 mm entre eles


class PickPlaceController:
    """`pick`/`place` aceitam tanto uma posicao fixa [x,y,z] quanto uma
    funcao sem argumentos que RETORNA [x,y,z] (ex: uma nova leitura de
    visao). Com uma funcao, a posicao e re-resolvida antes de CADA estagio
    do movimento (acima -> hover -> descida final), entao se o objeto-alvo
    se mexer entre uma chamada e outra, o braco corrige a rota em vez de
    mirar numa coordenada congelada no inicio do pick/place. Isso NAO e
    rastreamento continuo durante o movimento (so re-mede entre estagios),
    mas fecha a maior parte da janela de erro: o pior caso e o objeto se
    mover durante os ~1-2s de UM estagio, nao durante toda a tarefa."""

    def __init__(self, arm, gripper, down_orientation=DOWN_ORIENTATION, park_position=None):
        """`park_position`: se dado, o braco vai para la depois de cada pick e
        de cada place -- fora da vista das cameras. Pairando sobre a mesa, o
        braco tapava parte da imagem e get_scene_objects 'perdia' cubos."""
        self.arm = arm
        self.gripper = gripper
        self.down_orientation = down_orientation
        self.park_position = park_position

    def _park(self):
        if self.park_position is not None:
            self.arm.move_to_pose(self.park_position, self.down_orientation)

    @staticmethod
    def _resolve(position):
        return list(position()) if callable(position) else list(position)

    def _orientation(self, yaw):
        return self.down_orientation if not yaw else down_with_yaw(yaw)

    def _approach(self, position, yaw=0.0):
        q = self._orientation(yaw)
        p = self._resolve(position)
        above = [p[0], p[1], p[2] + _APPROACH_HEIGHT]
        self.arm.move_to_pose(above, q)

        p = self._resolve(position)
        hover = [p[0], p[1], p[2] + _HOVER_HEIGHT]
        self.arm.move_to_pose(hover, q)

        # sobe/alinha em XY bem acima primeiro, so entao desce devagar e reto
        # por cima do alvo -- evita que a garra (aberta) varra lateralmente
        # o objeto enquanto ainda esta convergindo em X/Y.
        p = self._resolve(position)
        self.arm.move_to_pose(p, q, steps=_FINAL_DESCEND_STEPS)
        return p

    def pick(self, position, yaw=0.0):
        """Retorna a posicao (ultima resolvida) onde de fato tentou pegar."""
        self.gripper.open()
        p = self._approach(position, yaw)
        self.gripper.close()
        above = [p[0], p[1], p[2] + _APPROACH_HEIGHT]
        self.arm.move_to_pose(above, self._orientation(yaw))
        self._park()
        return p

    def place(self, position, yaw=0.0):
        """Retorna a posicao (ultima resolvida) onde de fato soltou."""
        p = self._approach(position, yaw)
        self.gripper.open()
        above = [p[0], p[1], p[2] + _APPROACH_HEIGHT]
        self.arm.move_to_pose(above, self._orientation(yaw))
        self._park()
        return p
