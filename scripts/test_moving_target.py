"""Simula o cenario: pega o cubo azul, e ENQUANTO o braco esta a caminho do
lugar de soltar, o cubo vermelho e deslocado (ex: alguem mexeu nele).
Verifica se place_on() corrige a mira em vez de mirar na posicao antiga."""
import asyncio
import importlib.util
import threading
import time

spec = importlib.util.spec_from_file_location('mcp_server', 'scripts/mcp_server.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

async def main():
    state = mod._robot()
    sim = state['conn'].sim

    scene = await mod.mcp.call_tool('get_scene_objects', {})
    print('cena inicial:', scene.content[0].text)

    r = await mod.mcp.call_tool('pick_object', {'object_name': 'blue'})
    print('pick:', r.content[0].text)

    red_handle = sim.getObject('/red_cube')
    original_red = sim.getObjectPosition(red_handle, -1)
    moved_red = [original_red[0] - 0.12, original_red[1] + 0.10, original_red[2]]

    def mover_vermelho_no_meio_do_movimento():
        # espera um pouco (o place_on ja comecou a se mover) e desloca o
        # cubo vermelho fisicamente ENQUANTO o braco esta em transito.
        # usa uma conexao ZMQ PROPRIA (socket REQ nao e thread-safe/nao da
        # pra compartilhar com a conexao principal usada pelo braco).
        time.sleep(1.2)
        from coppeliasim_zmqremoteapi_client import RemoteAPIClient
        client2 = RemoteAPIClient()
        sim2 = client2.require('sim')
        sim2.setObjectPosition(red_handle, moved_red, -1)
        print(f'>>> cubo vermelho deslocado NO MEIO DO MOVIMENTO para {[round(v,3) for v in moved_red]}')

    mover_thread = threading.Thread(target=mover_vermelho_no_meio_do_movimento)
    mover_thread.start()

    r = await mod.mcp.call_tool('place_on', {'object_name': 'red'})
    print('place_on:', r.content[0].text)
    mover_thread.join()

    blue_final = sim.getObjectPosition(sim.getObject('/blue_cube'), -1)
    red_final = sim.getObjectPosition(red_handle, -1)
    print('vermelho (posicao original):', [round(v,3) for v in original_red])
    print('vermelho (posicao final, depois do deslocamento):', [round(v,3) for v in red_final])
    print('azul (posicao final):', [round(v,3) for v in blue_final])

    err_vs_new = sum((blue_final[k]-red_final[k])**2 for k in range(2)) ** 0.5
    err_vs_old = sum((blue_final[k]-original_red[k])**2 for k in range(2)) ** 0.5
    print(f'distancia XY do azul ate o vermelho NOVO: {err_vs_new*100:.1f}cm')
    print(f'distancia XY do azul ate o vermelho ANTIGO: {err_vs_old*100:.1f}cm')
    print('empilhou sobre a posicao NOVA (corrigiu em voo)?', err_vs_new < 0.02)

asyncio.run(main())
