"""Sequencia de alto nivel: pick(posicao) / place(posicao).

`position` e sempre o CENTRO do objeto (ex: centro do cubo). O tip (ponto
attachPoint da garra) e levado ate essa mesma altura para um grasp
centralizado nos dedos.
"""

# orientacao com a garra apontando para baixo (180 graus em torno de X)
DOWN_ORIENTATION = [1, 0, 0, 0]

_APPROACH_HEIGHT = 0.15  # altura acima do objeto no waypoint de aproximacao
_HOVER_HEIGHT = 0.05  # waypoint intermediario logo acima do objeto
_FINAL_DESCEND_STEPS = 150  # passos extra-finos so no trecho final (hover->objeto)


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

    def __init__(self, arm, gripper, down_orientation=DOWN_ORIENTATION):
        self.arm = arm
        self.gripper = gripper
        self.down_orientation = down_orientation

    @staticmethod
    def _resolve(position):
        return list(position()) if callable(position) else list(position)

    def _approach(self, position):
        p = self._resolve(position)
        above = [p[0], p[1], p[2] + _APPROACH_HEIGHT]
        self.arm.move_to_pose(above, self.down_orientation)

        p = self._resolve(position)
        hover = [p[0], p[1], p[2] + _HOVER_HEIGHT]
        self.arm.move_to_pose(hover, self.down_orientation)

        # sobe/alinha em XY bem acima primeiro, so entao desce devagar e reto
        # por cima do alvo -- evita que a garra (aberta) varra lateralmente
        # o objeto enquanto ainda esta convergindo em X/Y.
        p = self._resolve(position)
        self.arm.move_to_pose(p, self.down_orientation, steps=_FINAL_DESCEND_STEPS)
        return p

    def pick(self, position):
        """Retorna a posicao (ultima resolvida) onde de fato tentou pegar."""
        self.gripper.open()
        p = self._approach(position)
        self.gripper.close()
        above = [p[0], p[1], p[2] + _APPROACH_HEIGHT]
        self.arm.move_to_pose(above, self.down_orientation)
        return p

    def place(self, position):
        """Retorna a posicao (ultima resolvida) onde de fato soltou."""
        p = self._approach(position)
        self.gripper.open()
        above = [p[0], p[1], p[2] + _APPROACH_HEIGHT]
        self.arm.move_to_pose(above, self.down_orientation)
        return p
