"""Fase 4 - liga tudo do zero: abre o CoppeliaSim (se nao estiver aberto),
monta a cena, sobe o dashboard web e abre ele no navegador.

Depois de reiniciar o PC (ou fechar tudo manualmente), e so rodar este
script -- nao precisa fazer mais nada a mao.

Uso:
  .venv/Scripts/python.exe scripts/start_all.py

Pra parar: Ctrl+C nesta janela (derruba o dashboard; o CoppeliaSim fica
aberto, feche ele direto se quiser).
"""
import os
import subprocess
import sys
import time
import webbrowser

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
COPPELIA_EXE = r"C:\Program Files\CoppeliaRobotics\CoppeliaSimEdu\coppeliaSim.exe"
DASHBOARD_URL = "http://localhost:8000"


def _coppelia_actually_running() -> bool:
    """So conta um processo que esta de fato vivo (CPU>0 em algum momento
    ou responde), nao entradas fantasmas que o Windows as vezes deixa para
    tras depois de um fechamento abrupto."""
    try:
        import zmq
        ctx = zmq.Context()
        sock = ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.LINGER, 0)
        sock.RCVTIMEO = 1000
        sock.SNDTIMEO = 1000
        sock.connect("tcp://127.0.0.1:23000")
        sock.close()
        ctx.term()
        # conectar nao garante que alguem esta ouvindo (TCP so confirma na
        # troca de dados) -- tenta de fato via o client oficial, que da erro
        # rapido se nao houver servidor.
        from coppeliasim_zmqremoteapi_client import RemoteAPIClient
        client = RemoteAPIClient()
        client.require("sim")
        return True
    except Exception:
        return False


def main():
    if not os.path.exists(PYTHON):
        print(f"ERRO: venv nao encontrado em {PYTHON}. Rode a partir da raiz do projeto.")
        sys.exit(1)

    print("Verificando se o CoppeliaSim ja esta respondendo...")
    if _coppelia_actually_running():
        print("  -> ja esta rodando e respondendo, não precisa abrir de novo.")
    else:
        print("  -> nao respondeu. Abrindo o CoppeliaSim...")
        subprocess.Popen(["powershell", "-Command", f"Start-Process '{COPPELIA_EXE}'"])
        print("  Aguardando ele inicializar (~8s)...")
        time.sleep(8)

    print("Montando a cena (sim/phase1_pick_place.ttt)...")
    subprocess.run(
        [PYTHON, os.path.join(ROOT, "scripts", "build_scene_phase1.py")],
        check=True,
    )

    print("Subindo o dashboard...")
    proc = subprocess.Popen([PYTHON, os.path.join(ROOT, "scripts", "run_webapp.py")])
    time.sleep(2.5)  # da tempo do uvicorn subir antes de abrir o navegador
    print(f"Abrindo {DASHBOARD_URL} no navegador...")
    webbrowser.open(DASHBOARD_URL)

    print("\nTudo pronto. Ctrl+C aqui derruba o dashboard (o CoppeliaSim continua aberto).\n")
    try:
        proc.wait()
    except KeyboardInterrupt:
        print("Parando o dashboard...")
        proc.terminate()


if __name__ == "__main__":
    main()
