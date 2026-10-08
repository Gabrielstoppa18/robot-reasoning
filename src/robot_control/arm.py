"""Controle do UR5 via IK (plugin simIK), dirigindo o dummy '/UR5_target'."""
import math

_STEPS_PER_MOVE = 100

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

        self.ik_env = self.simIK.createEnvironment()
        self.ik_group = self.simIK.createGroup(self.ik_env)
        self.simIK.setGroupCalculation(
            self.ik_env, self.ik_group,
            self.simIK.method_damped_least_squares, 0.3, 20,
        )
        self.simIK.addElementFromScene(
            self.ik_env, self.ik_group,
            self.base, self.tip, self.target,
            self.simIK.constraint_pose,
        )

        self._go_to_ready_pose()

    def _go_to_ready_pose(self):
        for j, angle in zip(self.joints, _READY_CONFIG):
            self.sim.setJointPosition(j, angle)
        for _ in range(10):
            self.conn.step()
        self.sim.setObjectPose(self.target, self.sim.getObjectPose(self.tip, -1), -1)

    def _collect_joints(self, base):
        all_objs = self.sim.getObjectsInTree(base, self.sim.object_joint_type, 0)
        return list(all_objs)

    def get_tip_pose(self):
        return self.sim.getObjectPose(self.tip, -1)

    def move_to_pose(self, position, orientation_quat=None, steps=_STEPS_PER_MOVE):
        """Move o braco (via IK) ate que a ponta (tip) alcance a pose dada.
        `orientation_quat` em [x,y,z,w]; se None, mantem a orientacao atual."""
        start_pose = self.sim.getObjectPose(self.target, -1)
        start_pos, start_quat = start_pose[:3], start_pose[3:]
        end_quat = orientation_quat if orientation_quat is not None else start_quat

        for i in range(1, steps + 1):
            t = i / steps
            pos = [_lerp(start_pos[k], position[k], t) for k in range(3)]
            quat = _slerp(start_quat, end_quat, t)
            self.sim.setObjectPose(self.target, pos + quat, -1)
            self.simIK.handleGroup(self.ik_env, self.ik_group, {"syncWorlds": True})
            self.conn.step()

    def current_tip_position(self):
        return self.sim.getObjectPosition(self.tip, -1)
