"""Dashboard web da Fase 4: manda uma tarefa em linguagem natural pro
Claude ou Gemini, ve o raciocinio/chamadas de ferramenta chegando em tempo
real, e acompanha as cameras do robo. Roda localmente (uvicorn).

Uso:
  .venv/Scripts/python.exe -m uvicorn webapp.app:app --reload --port 8000
  (ou scripts/run_webapp.py, que ja chama isso)
"""
import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response

# este arquivo vive em src/webapp/app.py; roda-lo assume que "src" ja esta
# no sys.path (scripts/run_webapp.py cuida disso via uvicorn --app-dir).
ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

from llm_agent.claude_agent import ClaudeAgentSession
from llm_agent.gemini_agent import GeminiAgentSession
from webapp.camera_feed import CameraFeed

@asynccontextmanager
async def _lifespan(app: FastAPI):
    yield
    for session in _sessions.values():
        try:
            await session.__aexit__(None, None, None)
        except Exception:
            pass


app = FastAPI(title="robot-reasoning dashboard", lifespan=_lifespan)

_PYTHON_EXE = sys.executable
_MCP_SERVER_SCRIPT = str(ROOT / "scripts" / "mcp_server.py")

_sessions: dict[str, object] = {}
_session_locks: dict[str, asyncio.Lock] = {"claude": asyncio.Lock(), "gemini": asyncio.Lock()}
_camera_feed: CameraFeed | None = None


def _provider_available(provider: str) -> bool:
    if provider == "claude":
        return bool(os.environ.get("ANTHROPIC_API_KEY"))
    if provider == "gemini":
        return bool(os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"))
    return False


async def _get_session(provider: str, model: str | None):
    if provider not in _sessions:
        if provider == "claude":
            session = ClaudeAgentSession(_PYTHON_EXE, _MCP_SERVER_SCRIPT, model=model or "claude-opus-5")
        elif provider == "gemini":
            session = GeminiAgentSession(_PYTHON_EXE, _MCP_SERVER_SCRIPT, model=model or "gemini-3.7-flash")
        else:
            raise ValueError(f"provider desconhecido: {provider}")
        await session.__aenter__()
        _sessions[provider] = session
    return _sessions[provider]


@app.get("/")
async def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/api/status")
async def status():
    return JSONResponse({
        "providers": {
            "claude": {"available": _provider_available("claude"), "active": "claude" in _sessions},
            "gemini": {"available": _provider_available("gemini"), "active": "gemini" in _sessions},
        },
    })


@app.get("/api/camera/{name}")
async def camera_snapshot(name: str):
    global _camera_feed
    if _camera_feed is None:
        _camera_feed = CameraFeed()
    jpeg = _camera_feed.snapshot_jpeg(name)
    if jpeg is None:
        return JSONResponse({"error": f"camera '{name}' nao encontrada"}, status_code=404)
    return Response(content=jpeg, media_type="image/jpeg")


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "message": "JSON invalido"})
                continue

            provider = msg.get("provider", "gemini")
            model = msg.get("model") or None
            task = msg.get("task", "").strip()

            if not task:
                await websocket.send_json({"type": "error", "message": "tarefa vazia"})
                continue
            if not _provider_available(provider):
                await websocket.send_json({
                    "type": "error",
                    "message": f"provider '{provider}' sem chave de API configurada (.env)",
                })
                continue

            lock = _session_locks[provider]
            if lock.locked():
                await websocket.send_json({
                    "type": "error",
                    "message": f"ja ha uma tarefa rodando em '{provider}', espere terminar",
                })
                continue

            async with lock:
                try:
                    await websocket.send_json({"type": "session_starting", "provider": provider})
                    session = await _get_session(provider, model)
                    await websocket.send_json({"type": "task_started", "provider": provider, "task": task})

                    async def on_event(event: dict, ws=websocket):
                        await ws.send_json({**event, "provider": provider})

                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    transcript_path = str(ROOT / "docs" / "transcripts" / f"{provider}_{ts}.json")

                    await session.run_task(
                        task,
                        model=model,
                        transcript_path=transcript_path,
                        on_event=on_event,
                        verbose=False,
                    )
                    await websocket.send_json({
                        "type": "task_complete", "provider": provider, "transcript_path": transcript_path,
                    })
                except Exception as e:
                    await websocket.send_json({"type": "error", "message": f"{type(e).__name__}: {e}"})
    except WebSocketDisconnect:
        pass
