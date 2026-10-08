"""Fase 2 - pick-and-place usando coordenadas vindas da VISAO (par estereo +
segmentacao de cor), nao mais do ground truth do simulador. Empilha o cubo
azul sobre o vermelho -- o movimento fisico da tarefa final ("apagar o
fogo"), ainda sem raciocinio de LLM (isso fica para a Fase 4)."""
import sys
sys.path.insert(0, 'src')
from robot_control import SimConnection, UR5Arm, RG2Gripper, PickPlaceController
from vision import VisionPipeline

CUBE_SIZE = 0.04

conn = SimConnection()
sim = conn.sim

conn.stop_simulation()
while sim.getSimulationState() != sim.simulation_stopped:
    pass
conn.set_stepping(True)
conn.start_simulation()
for _ in range(5):
    conn.step()

vision = VisionPipeline(conn)
detected = vision.get_object_positions()
print('Objetos detectados pela visao:', detected)

if 'red' not in detected or 'blue' not in detected:
    print('ERRO: nao detectou os dois cubos. Abortando.')
    sys.exit(1)

red_pos = detected['red']
blue_pos = detected['blue']
stack_target = [red_pos[0], red_pos[1], red_pos[2] + CUBE_SIZE]

arm = UR5Arm(conn)
gripper = RG2Gripper(conn)
controller = PickPlaceController(arm, gripper)

controller.pick(blue_pos)
print('segurando o cubo azul?', gripper.is_holding)
controller.place(stack_target)

blue_cube = sim.getObject('/blue_cube')
red_cube = sim.getObject('/red_cube')
final_blue = sim.getObjectPosition(blue_cube, -1)
final_red = sim.getObjectPosition(red_cube, -1)
print('posicao final do cubo azul:', [round(v, 3) for v in final_blue])
print('posicao final do cubo vermelho:', [round(v, 3) for v in final_red])
stacked = (
    abs(final_blue[0] - final_red[0]) < 0.015
    and abs(final_blue[1] - final_red[1]) < 0.015
    and final_blue[2] > final_red[2] + CUBE_SIZE * 0.7
)
print('cubo azul empilhado sobre o vermelho?', stacked)
