"""Controle do UR5 via IK (plugin simIK), dirigindo o dummy '/UR5_target'."""
import itertools
import math

_STEPS_PER_MOVE = 10  # numero minimo de pontos do plano por movimento
_WAYPOINT_SPACING = 0.02  # m entre pontos do plano cartesiano
_IK_POS_TOL = 0.001  # m: erro maximo do IK em cada ponto planejado
_MAX_JOINT_JUMP = 0.25  # rad entre pontos vizinhos; acima disso, salto de configuracao (singularidade)
_SETTLE_MAX_STEPS = 120  # passos extras ate a ponta alcancar o ultimo ponto
_SETTLE_TOL = 0.005  # rad (pose ready)
_SETTLE_TIP_TOL = 0.0015  # m: ponta no ultimo ponto do plano
_TRACK_TIP_TOL = 0.01  # m: atraso maximo da ponta em relacao ao ponto atual do plano
_TRACK_MAX_STEPS = 20
_POLAR_MIN_SWING = 0.2  # rad de giro da base a partir do qual o transporte e polar
_POLAR_MIN_RADIUS = 0.15  # m: mais perto do eixo que isso, a reta e usada
_CHECKED_JOINTS = (2, 3, 4)  # cotovelo, punho 1, punho 2 (indices em self.joints)
_JOINT_SPACE_MIN_DIST = 0.2  # m: acima disso o movimento e planejado em espaco de juntas
_JOINT_SPACE_STEP = 0.04  # rad maximos por ponto no espaco de juntas
_ELBOW_UP_SEED = [0.1, -2.0, 0.3, 1.57, 0.0]  # juntas 2..6: cotovelo para cima (junta 1 = direcao do alvo)
_ELBOW_UP_Z = 0.15  # m: altura minima do eixo do cotovelo no ramo cotovelo-para-cima
_TIP_SAG_TOL = 0.03  # m: quanto a ponta pode passar abaixo dos dois extremos no transporte
_MAX_TURN = 4.5  # rad: maior giro de uma junta aceito entre a pose atual e a final
_MAX_COMBOS = 12  # voltas equivalentes testadas por solucao de IK
_MIN_JOINT_Z = 0.06  # m: altura minima dos eixos (meia espessura dos elos ~6 cm; a pose ready tem o ombro a 9 cm)

# pose "ready" do UR5, longe da singularidade de braco totalmente esticado
# (todos os joints em 0), que deixa a IK instavel/presa. Usada so na
# inicializacao, antes de qualquer movimento via IK.
_READY_CONFIG = [0, -math.pi / 2, math.pi / 2, -math.pi / 2, -math.pi / 2, 0]


def _lerp(a, b, t):
    return a + (b - a) * t


def _slerp(q1, q2, t):
    # quaternion [x, y, z, w]
    dot = sum(a * b for a, b in zip(q1, q2))
    if dot < 0.0:
        q2 = [-c for c in q2]
        dot = -dot
    if dot > 0.9995:
        result = [_lerp(a, b, t) for a, b in zip(q1, q2)]
        n = math.sqrt(sum(c * c for c in result))
        return [c / n for c in result]
    theta_0 = math.acos(max(-1.0, min(1.0, dot)))
    theta = theta_0 * t
    sin_theta = math.sin(theta)
    sin_theta_0 = math.sin(theta_0)
    s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return [s0 * a + s1 * b for a, b in zip(q1, q2)]


class UnreachablePose(Exception):
    """O plano de movimento nao tem solucao de IK precisa e continua: alvo
    fora do alcance ou caminho por uma singularidade. O braco nao se move."""


def _unwrap(q, ref):
    """Angulo equivalente (q + 2*pi*k) mais perto de ref (juntas de volta multipla)."""
    return q + 2 * math.pi * round((ref - q) / (2 * math.pi))


class UR5Arm:
    def __init__(self, conn):
        self.conn = conn
        self.sim = conn.sim
        self.simIK = conn.simIK

        self.base = self.sim.getObject("/UR5")
        self.tip = self.sim.getObject("/UR5_tip")
        self.target = self.sim.getObject("/UR5_target")

        # os 6 joints do braco sao os 6 primeiros encontrados na subarvore do
        # UR5 (a garra RG2, montada no final da cadeia, tem seus proprios
        # joints internos que NAO entram aqui). Mantidos no modo dinamico
        # original (motor PID do proprio modelo) -- colocar em modo kinematic
        # faz os elos "cairem" sob gravidade a cada passo sem reforco
        # continuo, e o simIK (com syncWorlds=True) ja sabe comandar joints
        # dinamicos via setJointTargetPosition.
        self.joints = self._collect_joints(self.base)[:6]
        self._disable_link_collisions()

        # ambiente de IK usado so para PLANEJAR: resolve cada ponto sem mexer no
        # braco da cena (sem syncWorlds). O braco so recebe alvos de junta
        # depois que o plano inteiro tem solucao precisa e continua -- um alvo
        # inalcancavel ou uma singularidade no meio do caminho fazia o IK
        # devolver angulos absurdos, os motores saturavam e o braco
        # "desmontava" (e arremessava os cubos).
        self.ik_env = self.simIK.createEnvironment()
        self.ik_group = self.simIK.createGroup(self.ik_env)
        self.simIK.setGroupCalculation(
            self.ik_env, self.ik_group,
            self.simIK.method_damped_least_squares, 0.1, 50,
        )
        _, sim_to_ik, _ = self.simIK.addElementFromScene(
            self.ik_env, self.ik_group,
            self.base, self.tip, self.target,
            self.simIK.constraint_pose,
        )
        self._ik_target, self._ik_tip = sim_to_ik[self.target], sim_to_ik[self.tip]
        self._ik_joints = [sim_to_ik[j] for j in self.joints]

        self._go_to_ready_pose()

    def _disable_link_collisions(self):
        """Elos do UR5 (nao a garra) deixam de colidir. Montado na altura da
        mesa, o braco encostava o elo superior na mesa e o antebraco no chao;
        os motores da base e do ombro saturavam em 200 N.m, a base travava e a
        ponta ficava de 4 a 16 cm fora do alvo (2026-10-10). A garra RG2 e os
        cubos mantem a fisica completa (agarre, empurroes, pilhas). So vale
        durante a execucao -- a cena salva nao muda."""
        rg2 = self.sim.getObject("/UR5/RG2", {"noError": True})
        rg2_tree = set(self.sim.getObjectsInTree(rg2, self.sim.handle_all, 0)) if rg2 != -1 else set()
        for h in self.sim.getObjectsInTree(self.base, self.sim.object_shape_type, 0):
            if h == self.base or h in rg2_tree:
                continue
            if self.sim.getObjectInt32Param(h, self.sim.shapeintparam_respondable):
                self.sim.setObjectInt32Param(h, self.sim.shapeintparam_respondable, 0)
                self.sim.resetDynamicObject(h)

    def _go_to_ready_pose(self):
        # as juntas sao dinamicas (motor PID): e preciso fixar tambem o ALVO,
        # senao o motor as puxa de volta para 0 -- a configuracao esticada e
        # singular que esta pose existe para evitar
        # (e setJointPosition nao teletransporta uma junta dinamica: quem leva o
        # braco ate la e o motor, a ~1,57 rad/s -- espera chegar de verdade,
        # senao o planejamento parte de um braco ainda em movimento)
        # Na fisica a "teletransportacao" nao e instantanea: os elos varrem o
        # espaco ate a pose nova, e a garra (colidivel) arremessava cubos da
        # mesa no caminho. Durante a transicao a garra tambem fica sem colisao.
        rg2 = self.sim.getObject("/UR5/RG2", {"noError": True})
        rg2_shapes = [h for h in (self.sim.getObjectsInTree(rg2, self.sim.object_shape_type, 0) if rg2 != -1 else [])
                      if self.sim.getObjectInt32Param(h, self.sim.shapeintparam_respondable)]
        for h in rg2_shapes:
            self.sim.setObjectInt32Param(h, self.sim.shapeintparam_respondable, 0)
        for j, angle in zip(self.joints, _READY_CONFIG):
            self.sim.setJointPosition(j, angle)
            self.sim.setJointTargetPosition(j, angle)
        for _ in range(200):
            self.conn.step()
            if max(abs(self.sim.getJointPosition(j) - a) for j, a in zip(self.joints, _READY_CONFIG)) <= _SETTLE_TOL:
                break
        for _ in range(10):  # elos dinamicos assentam depois das juntas
            self.conn.step()
        for h in rg2_shapes:
            self.sim.setObjectInt32Param(h, self.sim.shapeintparam_respondable, 1)
            self.sim.resetDynamicObject(h)
        self._cmd_pose = list(self.sim.getObjectPose(self.tip, -1))
        self.sim.setObjectPose(self.target, self._cmd_pose, -1)

    def _collect_joints(self, base):
        all_objs = self.sim.getObjectsInTree(base, self.sim.object_joint_type, 0)
        return list(all_objs)

    def get_tip_pose(self):
        return self.sim.getObjectPose(self.tip, -1)

    def _fk(self, q):
        """Pose da ponta e altura dos eixos checados para uma configuracao,
        no ambiente de IK (nao mexe no braco da cena)."""
        ik, env, w = self.simIK, self.ik_env, self.simIK.handle_world
        for h, v in zip(self._ik_joints, q):
            ik.setJointPosition(env, h, v)
        tip = ik.getObjectPose(env, self._ik_tip, w)
        low = min(ik.getObjectPose(env, self._ik_joints[i], w)[2] for i in _CHECKED_JOINTS)
        return tip, low

    def _plan_joint_space(self, position, end_quat, start_pos, steps):
        """Interpola as juntas da configuracao atual ate uma solucao de IK do
        alvo no ramo cotovelo-para-cima. Devolve None (e o chamador cai na
        reta cartesiana) se nao houver solucao precisa ou se o caminho
        deixar eixos abaixo de _MIN_JOINT_Z ou a ponta abaixo de ambos os
        extremos (varreria a mesa/cubos)."""
        ik, env, grp, w = self.simIK, self.ik_env, self.ik_group, self.simIK.handle_world
        q_cur = [self.sim.getJointPosition(j) for j in self.joints]
        th = math.atan2(position[1], position[0])
        best = None
        for dth in (0.0, 0.15, -0.15, 0.3, -0.3):
            seed = [th + math.pi / 2 + dth] + _ELBOW_UP_SEED
            for h, v in zip(self._ik_joints, seed):
                ik.setJointPosition(env, h, v)
            ik.setObjectPose(env, self._ik_target, list(position) + list(end_quat), w)
            for _ in range(4):
                ik.handleGroup(env, grp)
            tip = ik.getObjectPose(env, self._ik_tip, w)
            if math.dist(tip[:3], position) > _IK_POS_TOL:
                continue
            q_raw = [ik.getJointPosition(env, h) for h in self._ik_joints]
            _, low = self._fk(q_raw)
            elbow_z = ik.getObjectPose(env, self._ik_joints[2], w)[2]
            if low < _MIN_JOINT_Z or elbow_z < _ELBOW_UP_Z:
                continue
            # a mesma pose tem varias voltas equivalentes (q + 2*pi*k) por
            # junta; a mais proxima nem sempre faz o caminho seguro (o cotovelo
            # girando "por dentro" derruba o punho), entao testa as voltas das
            # tres primeiras juntas em ordem de custo
            cands = []
            for i, (v, r) in enumerate(zip(q_raw, q_cur)):
                opts = [v + 2 * math.pi * k for k in (-1, 0, 1)] if i in (1, 2, 3) else [_unwrap(v, r)]
                cands.append([o for o in opts if abs(o - r) <= _MAX_TURN] or [_unwrap(v, r)])
            combos = sorted(itertools.product(*cands),
                            key=lambda c: max(abs(a - b) for a, b in zip(c, q_cur)))
            for q_end in combos[:_MAX_COMBOS]:
                path = self._joint_path(q_cur, list(q_end), position, start_pos, steps)
                if path is not None:
                    cost = max(abs(a - b) for a, b in zip(q_end, q_cur))
                    if best is None or cost < best[0]:
                        best = (cost, path)
                    break
        if best is None:
            return None
        return best[1], list(position) + list(end_quat)

    def _joint_path(self, q_cur, q_end, position, start_pos, steps):
        """Pontos (q, posicao da ponta) entre duas configuracoes, ou None se
        algum eixo checado ficar baixo demais ou a ponta afundar."""
        n = max(steps, math.ceil(max(abs(a - b) for a, b in zip(q_end, q_cur)) / _JOINT_SPACE_STEP))
        floor_z = min(start_pos[2], position[2]) - _TIP_SAG_TOL
        path = []
        for i in range(1, n + 1):
            t = i / n
            q = [a + (b - a) * t for a, b in zip(q_cur, q_end)]
            tip, low = self._fk(q)
            if low < _MIN_JOINT_Z or tip[2] < floor_z:
                return None
            path.append((q, list(tip[:3])))
        return path

    def plan(self, position, orientation_quat=None, steps=_STEPS_PER_MOVE):
        """Configuracoes de junta ao longo da reta cartesiana ate a pose dada,
        cada uma resolvida a partir da anterior. Se as juntas saltarem num
        trecho, o trecho e dividido ao meio (ate 1/16): um giro continuo do
        punho diminui com a subdivisao; uma troca de ramo da solucao
        (singularidade) nao diminui. Levanta UnreachablePose se algum ponto
        nao tiver solucao precisa ou se o salto persistir."""
        start_pos, start_quat = self._cmd_pose[:3], self._cmd_pose[3:]
        end_quat = list(orientation_quat) if orientation_quat is not None else start_quat

        # Movimentos longos: espaco de juntas, no ramo "cotovelo para cima".
        # A IK local (DLS) a partir da pose atual cai muito no ramo cotovelo
        # para baixo (eixo do cotovelo a ~0 cm do chao) para alvos que tem
        # solucao com o cotovelo a ~50 cm; o ramo certo e escolhido aqui.
        if math.dist(start_pos, position) > _JOINT_SPACE_MIN_DIST:
            planned = self._plan_joint_space(position, end_quat, start_pos, steps)
            if planned is not None:
                return planned

        # Transporte: uma reta cartesiana entre dois pontos de lados opostos da
        # base passa perto do eixo do ombro; ali o braco se dobra e os elos
        # afundam no chao (medido: ate -21 cm). Quando o giro da base e
        # grande, interpola em coordenadas polares (raio, angulo, z) em torno
        # da base: o raio fica entre o inicial e o final e o braco so gira e
        # estende. Em movimentos verticais/curtos coincide com a reta.
        sr, er = math.hypot(*start_pos[:2]), math.hypot(*position[:2])
        s_th, e_th = math.atan2(start_pos[1], start_pos[0]), math.atan2(position[1], position[0])
        d_th = (e_th - s_th + math.pi) % (2 * math.pi) - math.pi
        polar = abs(d_th) > _POLAR_MIN_SWING and min(sr, er) >= _POLAR_MIN_RADIUS

        def path_point(t):
            if not polar:
                return [_lerp(start_pos[k], position[k], t) for k in range(3)]
            r, th = _lerp(sr, er, t), s_th + d_th * t
            return [r * math.cos(th), r * math.sin(th), _lerp(start_pos[2], position[2], t)]

        if polar:
            arc = 0.5 * (sr + er) * abs(d_th)
            length = math.sqrt(arc ** 2 + (er - sr) ** 2 + (position[2] - start_pos[2]) ** 2)
        else:
            length = math.dist(start_pos, position)
        n = max(steps, math.ceil(length / _WAYPOINT_SPACING))
        q_prev = [self.sim.getJointPosition(j) for j in self.joints]
        for h, q in zip(self._ik_joints, q_prev):
            self.simIK.setJointPosition(self.ik_env, h, q)

        def solve(t, ref):
            pos = path_point(t)
            quat = _slerp(start_quat, end_quat, t)
            for h, q in zip(self._ik_joints, ref):
                self.simIK.setJointPosition(self.ik_env, h, q)
            self.simIK.setObjectPose(self.ik_env, self._ik_target, pos + quat, self.simIK.handle_world)
            self.simIK.handleGroup(self.ik_env, self.ik_group)  # ate 50 iteracoes de DLS por chamada
            tip = self.simIK.getObjectPose(self.ik_env, self._ik_tip, self.simIK.handle_world)
            q = [_unwrap(self.simIK.getJointPosition(self.ik_env, h), r) for h, r in zip(self._ik_joints, ref)]
            # altura dos eixos do cotovelo e dos dois primeiros do punho: os
            # elos tem ~6 cm de meia espessura, entao abaixo de _MIN_JOINT_Z
            # o elo atravessa o chao (os elos nao colidem, so se veria no video)
            low = min(self.simIK.getObjectPose(self.ik_env, self._ik_joints[i], self.simIK.handle_world)[2]
                      for i in _CHECKED_JOINTS)
            return q, pos, math.dist(tip[:3], pos), max(abs(a - b) for a, b in zip(q, ref)), low

        path, t, dt = [], 0.0, 1.0 / n
        while t < 1.0 - 1e-9:
            step = min(dt, 1.0 - t)
            q, pos, err, jump, low = solve(t + step, q_prev)
            if err <= _IK_POS_TOL and low < _MIN_JOINT_Z:
                raise UnreachablePose(
                    f"o braco atravessaria o chao para ir a {[round(float(v), 3) for v in position]} "
                    f"(eixo a {low * 100:.0f} cm de altura em {t * 100:.0f} % do caminho)")
            if err <= _IK_POS_TOL and jump <= _MAX_JOINT_JUMP:
                path.append((q, pos))
                q_prev, t = q, t + step
                dt = 1.0 / n
                continue
            if step > 1.0 / (16 * n) and err <= _IK_POS_TOL:
                dt = step / 2
                continue
            raise UnreachablePose(
                f"sem solucao de IK para {[round(float(v), 3) for v in position]} "
                f"(em {t * 100:.0f} % do caminho: erro {err * 1000:.1f} mm, salto de junta {jump:.2f} rad)")
        return path, list(position) + end_quat

    def move_to_pose(self, position, orientation_quat=None, steps=_STEPS_PER_MOVE):
        """Move o braco ate que a ponta (tip) alcance a pose dada, seguindo a
        reta cartesiana. `orientation_quat` em [x,y,z,w]; se None, mantem a
        orientacao atual. Planeja antes de mover: levanta UnreachablePose (e o
        braco nao se mexe) se o caminho nao tiver solucao."""
        path, final_pose = self.plan(position, orientation_quat, steps)
        for q, pos in path:
            for j, a in zip(self.joints, q):
                self.sim.setJointTargetPosition(j, a)
            # as juntas giram a no maximo ~1,57 rad/s: sem esperar cada ponto,
            # num movimento grande o braco ficava para tras do plano e a
            # trajetoria real saia da reta (cortava a mesa e varria cubos)
            for _ in range(_TRACK_MAX_STEPS):
                self.conn.step()
                if math.dist(self.sim.getObjectPosition(self.tip, -1), pos) <= _TRACK_TIP_TOL:
                    break
        # as juntas (PID) chegam atrasadas: espera a ponta assentar no fim
        for _ in range(_SETTLE_MAX_STEPS):
            if math.dist(self.sim.getObjectPosition(self.tip, -1), position) <= _SETTLE_TIP_TOL:
                break
            self.conn.step()
        self._cmd_pose = final_pose
        self.sim.setObjectPose(self.target, final_pose, -1)

    def current_tip_position(self):
        return self.sim.getObjectPosition(self.tip, -1)
