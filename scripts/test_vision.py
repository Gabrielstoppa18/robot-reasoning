"""Fase 2 - valida a triangulacao estereo contra a posicao real (ground truth)
dos cubos no simulador. Roda com sim/phase1_pick_place.ttt aberto."""
import sys
sys.path.insert(0, 'src')
from robot_control import SimConnection
from vision import VisionPipeline

conn = SimConnection()
sim = conn.sim
conn.stop_simulation()
while sim.getSimulationState() != sim.simulation_stopped:
    pass
conn.set_stepping(True)
conn.start_simulation()
for _ in range(5):
    conn.step()

pipeline = VisionPipeline(conn)
detected = pipeline.get_object_positions()

ground_truth = {
    'red': sim.getObjectPosition(sim.getObject('/red_cube'), -1),
    'blue': sim.getObjectPosition(sim.getObject('/blue_cube'), -1),
}

for color, gt in ground_truth.items():
    est = detected.get(color)
    print(f'{color}: ground truth={[round(v,3) for v in gt]}', end='  ')
    if est is None:
        print('-> NAO DETECTADO')
        continue
    err = sum((est[k] - gt[k]) ** 2 for k in range(3)) ** 0.5
    print(f'estimado={[round(v,3) for v in est]}  erro={round(err*100,2)}cm')
