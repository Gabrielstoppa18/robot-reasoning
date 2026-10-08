"""Fase 1 - demo completo: pega o cubo vermelho e coloca em outro ponto da mesa.
Roda com o CoppeliaSim aberto e a cena sim/phase1_pick_place.ttt carregada."""
import sys
sys.path.insert(0, 'src')
from robot_control import SimConnection, UR5Arm, RG2Gripper, PickPlaceController

PLACE_TARGET = [0.30, -0.25, 0.02]

conn = SimConnection()
sim = conn.sim

conn.stop_simulation()
while sim.getSimulationState() != sim.simulation_stopped:
    pass
conn.set_stepping(True)
conn.start_simulation()

arm = UR5Arm(conn)
gripper = RG2Gripper(conn)
controller = PickPlaceController(arm, gripper)

cube = sim.getObject('/red_cube')
cube_start = sim.getObjectPosition(cube, -1)
print('cubo antes:', [round(v, 3) for v in cube_start])

controller.pick(cube_start)
print('segurando o cubo apos pick?', gripper.is_holding)
controller.place(PLACE_TARGET)

cube_end = sim.getObjectPosition(cube, -1)
err = sum((cube_end[k] - PLACE_TARGET[k]) ** 2 for k in range(3)) ** 0.5
print('cubo depois:', [round(v, 3) for v in cube_end])
print('erro em relacao ao alvo (m):', round(err, 4))
print('segurando algo ao final?', gripper.is_holding)
