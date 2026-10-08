"""
Fase 0/1 - monta a cena: UR5 + garra RG2, mesa, cubo vermelho, 2 sensores de
visao (estereo, usados só na Fase 2), e salva como sim/phase1_pick_place.ttt.

Rodar com o CoppeliaSim aberto (cena em branco) e a simulacao PARADA.
"""
import math
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

MODELS_DIR = 'C:/Program Files/CoppeliaRobotics/CoppeliaSimEdu/models'

client = RemoteAPIClient()
sim = client.require('sim')

sim.stopSimulation()
while sim.getSimulationState() != sim.simulation_stopped:
    pass

# --- limpa objetos de uma eventual execucao anterior deste script ---
# (importa quando o CoppeliaSim ja estava aberto -- scripts/start_all.py
# nao reinicia o processo se ele ja estiver respondendo). So /UR5 e /RG2
# vieram de loadModel (precisam de removeModel); o resto foi criado direto
# (createPrimitiveShape/createVisionSensor/createDummy) e usa removeObject
# -- usar removeModel neles da erro "object is not tagged as model base".
_MODEL_ALIASES = {'/UR5', '/RG2'}
for alias in ['/UR5', '/RG2', '/table', '/red_cube', '/blue_cube',
              '/cam_left', '/cam_right', '/cam_overview', '/UR5_tip', '/UR5_target']:
    h = sim.getObject(alias, {'noError': True})
    if h != -1:
        if alias in _MODEL_ALIASES:
            sim.removeModel(h)
        else:
            sim.removeObject(h)

# --- robo ---
ur5 = sim.loadModel(f'{MODELS_DIR}/robots/non-mobile/UR5.ttm')
sim.setObjectAlias(ur5, 'UR5')

# o .ttm vem com um script de demo (threaded) que move os joints sozinho
# assim que a simulacao comeca; removemos para controlar o braco via IK/API.
demo_script = sim.getObject('/UR5/Script', {'noError': True})
if demo_script != -1:
    sim.removeObjects([demo_script])

connection = sim.getObject('/UR5/connection')

rg2 = sim.loadModel(f'{MODELS_DIR}/components/grippers/RG2.ttm')
sim.setObjectAlias(rg2, 'RG2')
# alinha a garra exatamente no ponto de conexao do UR5
sim.setObjectParent(rg2, connection, False)
sim.setObjectPose(rg2, [0, 0, 0, 0, 0, 0, 1], connection)
# a base da RG2 vem com corpo dinamico (cai por gravidade, independente do
# pai na hierarquia); torna-la estatica faz com que siga rigidamente o braco.
# os joints dos dedos continuam dinamicos (herdam dessa base agora fixa).
sim.setObjectInt32Param(rg2, sim.shapeintparam_static, 1)

attach_point = sim.getObject('/RG2/attachPoint')

# dummy "tip" (TCP) acompanhando a garra, usado como ponta da cadeia de IK
tip = sim.createDummy(0.01)
sim.setObjectAlias(tip, 'UR5_tip')
sim.setObjectParent(tip, attach_point, False)
sim.setObjectPose(tip, [0, 0, 0, 0, 0, 0, 1], attach_point)

# dummy "target" livre no mundo: e o que vamos mover para comandar o braco via IK
target = sim.createDummy(0.02)
sim.setObjectAlias(target, 'UR5_target')
world_tip_pose = sim.getObjectPose(tip, -1)
sim.setObjectPose(target, world_tip_pose, -1)

# --- mesa (bloco estatico) ---
table = sim.createPrimitiveShape(sim.primitiveshape_cuboid, [0.8, 0.8, 0.05])
sim.setObjectAlias(table, 'table')
sim.setObjectPose(table, [0, 0, -0.025, 0, 0, 0, 1], -1)
sim.setObjectInt32Param(table, sim.shapeintparam_static, 1)
sim.setObjectInt32Param(table, sim.shapeintparam_respondable, 1)
sim.setShapeColor(table, None, sim.colorcomponent_ambient_diffuse, [0.55, 0.4, 0.25])

# --- cubo vermelho (dinamico, respondable) ---
cube_size = 0.04
red_cube = sim.createPrimitiveShape(sim.primitiveshape_cuboid,
                                     [cube_size, cube_size, cube_size], 1)
sim.setObjectAlias(red_cube, 'red_cube')
sim.setObjectPose(red_cube, [0.45, 0.15, cube_size / 2, 0, 0, 0, 1], -1)
sim.setObjectInt32Param(red_cube, sim.shapeintparam_static, 0)
sim.setObjectInt32Param(red_cube, sim.shapeintparam_respondable, 1)
sim.setShapeColor(red_cube, None, sim.colorcomponent_ambient_diffuse, [0.8, 0.05, 0.05])
sim.computeMassAndInertia(red_cube, 500)  # densidade kg/m3 -> massa/inercia

# --- cubo azul (dinamico, respondable) ---
blue_cube = sim.createPrimitiveShape(sim.primitiveshape_cuboid,
                                      [cube_size, cube_size, cube_size], 1)
sim.setObjectAlias(blue_cube, 'blue_cube')
sim.setObjectPose(blue_cube, [0.30, 0.30, cube_size / 2, 0, 0, 0, 1], -1)
sim.setObjectInt32Param(blue_cube, sim.shapeintparam_static, 0)
sim.setObjectInt32Param(blue_cube, sim.shapeintparam_respondable, 1)
sim.setShapeColor(blue_cube, None, sim.colorcomponent_ambient_diffuse, [0.05, 0.05, 0.85])
sim.computeMassAndInertia(blue_cube, 500)

# --- par estereo: olha de cima para o ponto medio entre os dois cubos ---
CAM_CENTER = [(0.45 + 0.30) / 2, (0.15 + 0.30) / 2]
CAM_HEIGHT = 0.6
CAM_BASELINE = 0.12
CAM_FOV_DEG = 60

def make_camera(alias, x, y, z, orientation=(math.pi, 0, 0), fov_deg=CAM_FOV_DEG, resolution=512):
    # bit 1 (valor 2) = modo perspectiva. Sem isso o sensor cria em modo
    # ORTOGRAFICO (default), e floatParams[2] vira "tamanho ortografico" em
    # vez de angulo de visada -- o que quebra todo o modelo de camera
    # pinhole usado em vision/camera.py e vision/triangulate.py.
    options = 2
    int_params = [resolution, resolution, 0, 0]
    float_params = [0.01, 5.0, fov_deg * math.pi / 180, 0.1, 0.1, 0.1, 0, 0, 0, 0, 0]
    cam = sim.createVisionSensor(options, int_params, float_params)
    sim.setObjectFloatParam(cam, sim.visionfloatparam_perspective_angle, fov_deg * math.pi / 180)
    actual_fov = sim.getObjectFloatParam(cam, sim.visionfloatparam_perspective_angle)
    assert abs(math.degrees(actual_fov) - fov_deg) < 0.01, f'FOV nao aplicado: {math.degrees(actual_fov)}'
    sim.setObjectAlias(cam, alias)
    sim.setObjectPose(cam, [x, y, z, 0, 0, 0, 1], -1)
    sim.setObjectOrientation(cam, list(orientation), -1)
    return cam

cam_left = make_camera('cam_left', CAM_CENTER[0], CAM_CENTER[1] - CAM_BASELINE / 2, CAM_HEIGHT)
cam_right = make_camera('cam_right', CAM_CENTER[0], CAM_CENTER[1] + CAM_BASELINE / 2, CAM_HEIGHT)

# camera "de monitoramento": vista larga de cima de toda a cena (braco +
# mesa), so para a pessoa acompanhar a simulacao no dashboard -- nao e
# usada pelo pipeline de visao/triangulacao.
cam_overview = make_camera('cam_overview', 0.2, 0.1, 1.8, fov_deg=85, resolution=640)

sim.saveScene('D:/Documentos/Dev/robot-reasoning/sim/phase1_pick_place.ttt')
print('Cena salva em sim/phase1_pick_place.ttt')
print('UR5:', ur5, 'RG2:', rg2, 'tip:', tip, 'target:', target,
      'red_cube:', red_cube, 'blue_cube:', blue_cube, 'table:', table)
