"""Fase 4 - sobe o dashboard web local (painel de controle do agente +
cameras do robo).

Requisitos:
  - CoppeliaSim aberto com sim/phase1_pick_place.ttt carregado
  - .env preenchido com a(s) chave(s) de API que for usar

Uso:
  .venv/Scripts/python.exe scripts/run_webapp.py
  (abre em http://localhost:8000)
"""
import os
import sys

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, os.path.abspath(SRC_DIR))

import uvicorn

if __name__ == "__main__":
    print("Dashboard em http://localhost:8000")
    uvicorn.run("webapp.app:app", host="127.0.0.1", port=8000, reload=False)
