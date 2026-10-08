"""Fase 4 - agente LLM (Gemini) controlando o robo via MCP.

Mesma ideia e mesmo formato de transcript do claude_agent.py (loop
agentico manual, registrando entrada/saida de cada chamada de ferramenta)
para que os dois sejam diretamente comparaveis na dissertacao. A API da
Gemini e deliberadamente diferente da da Claude (function_call/
function_response em vez de tool_use/tool_result, roles "model"/"user" em
vez de "assistant"/"user"), entao o loop foi escrito do zero pra essa SDK
em vez de reaproveitar o da Claude.

`GeminiAgentSession` mantem a conexao MCP (e o processo do mcp_server.py)
viva entre varias chamadas a run_task -- usado pelo dashboard web, que
dispara varias tarefas em sequencia sem reconectar/reiniciar a simulacao a
cada uma. A funcao `run_task` solta no final do modulo e um atalho "uma
tarefa so" pra scripts simples (CLI, testes).
"""
import asyncio
import json
import time
from contextlib import AsyncExitStack
from pathlib import Path

from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

SYSTEM_PROMPT = (
    "Voce controla um braco robotico (UR5 com garra) numa simulacao, "
    "atraves das ferramentas disponiveis. Antes de agir, use "
    "get_scene_objects() para ver o que existe na mesa. Pense em voz alta "
    "sobre o que a tarefa pede e so entao chame as ferramentas necessarias, "
    "uma de cada vez, conferindo o resultado antes da proxima chamada. "
    "Quando a tarefa estiver completa, escreva uma frase final resumindo o "
    "que voce fez e por que -- sem chamar mais nenhuma ferramenta."
)

_DEFAULT_MODEL = "gemini-3.7-flash"  # 3.8-flash tem cota free-tier bem apertada (20 req/dia)
_MAX_TURNS = 15
_RETRYABLE_STATUS = {429, 500, 503, 504}
_MAX_RETRIES = 5


def _mcp_tools_to_gemini(mcp_tools) -> types.Tool:
    return types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name=t.name,
                description=t.description or "",
                parameters_json_schema=t.input_schema,
            )
            for t in mcp_tools
        ]
    )


def _parts_to_log(parts) -> list[dict]:
    out = []
    for p in parts or []:
        if p.function_call is not None:
            out.append({
                "type": "function_call",
                "name": p.function_call.name,
                "args": dict(p.function_call.args or {}),
            })
        elif p.thought:
            out.append({"type": "thinking", "text": p.text or ""})
        elif p.text is not None:
            out.append({"type": "text", "text": p.text})
    return out


async def _emit(on_event, event: dict):
    if on_event is None:
        return
    result = on_event(event)
    if asyncio.iscoroutine(result):
        await result


class GeminiAgentSession:
    """Conexao persistente com o MCP server + client do Gemini. Use como
    context manager async; `run_task` pode ser chamado varias vezes."""

    def __init__(self, python_executable: str, mcp_server_script: str, model: str = _DEFAULT_MODEL):
        self.python_executable = python_executable
        self.mcp_server_script = mcp_server_script
        self.model = model
        self._stack: AsyncExitStack | None = None
        self.mcp_session: ClientSession | None = None
        self.client: genai.Client | None = None
        self._gemini_tool: types.Tool | None = None

    async def __aenter__(self) -> "GeminiAgentSession":
        self._stack = AsyncExitStack()
        await self._stack.__aenter__()
        params = StdioServerParameters(
            command=self.python_executable, args=[self.mcp_server_script]
        )
        read, write = await self._stack.enter_async_context(stdio_client(params))
        self.mcp_session = await self._stack.enter_async_context(ClientSession(read, write))
        await self.mcp_session.initialize()
        tools_result = await self.mcp_session.list_tools()
        self._gemini_tool = _mcp_tools_to_gemini(tools_result.tools)
        self.client = genai.Client()
        return self

    async def __aexit__(self, *exc_info):
        await self._stack.__aexit__(*exc_info)
        self._stack = None

    async def _generate_with_retry(self, *, model, contents, config, on_event, verbose):
        delay = 2.0
        for attempt in range(_MAX_RETRIES):
            try:
                return await self.client.aio.models.generate_content(
                    model=model, contents=contents, config=config
                )
            except genai_errors.ClientError as e:
                # 429 de cota diaria esgotada nao adianta tentar de novo (o
                # retry sugerido pela API costuma ser de horas) -- falha
                # rapido com uma mensagem legivel em vez de martelar a API
                # e devolver o stack trace cru pro usuario.
                if e.code == 429:
                    raise RuntimeError(
                        f"Cota da API do Gemini esgotada para o modelo '{model}' "
                        f"(tier gratuito). Detalhe da API: {e.message}"
                    ) from e
                raise
            except genai_errors.ServerError as e:
                status = e.code
                if status not in _RETRYABLE_STATUS or attempt == _MAX_RETRIES - 1:
                    raise
                msg = (f"erro {status} da API (tentativa {attempt + 1}/{_MAX_RETRIES}), "
                       f"esperando {delay:.0f}s e tentando de novo...")
                if verbose:
                    print("  [aviso]", msg)
                await _emit(on_event, {"type": "warning", "message": msg})
                await asyncio.sleep(delay)
                delay *= 2

    async def run_task(
        self,
        task_prompt: str,
        *,
        model: str | None = None,
        max_turns: int = _MAX_TURNS,
        transcript_path: str | None = None,
        on_event=None,
        verbose: bool = True,
    ) -> dict:
        """Roda `task_prompt` ate o Gemini terminar (ou max_turns). Chama
        `on_event(dict)` (sync ou async) a cada evento -- turno, pensamento,
        texto, chamada/resultado de ferramenta, aviso, final."""
        model = model or self.model
        transcript = {
            "model": model,
            "task_prompt": task_prompt,
            "system_prompt": SYSTEM_PROMPT,
            "started_at": time.time(),
            "turns": [],
        }

        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            tools=[self._gemini_tool],
            thinking_config=types.ThinkingConfig(include_thoughts=True),
        )

        contents = [types.Content(role="user", parts=[types.Part.from_text(text=task_prompt)])]
        final_text = None
        finish_reason = None

        for turn_idx in range(max_turns):
            await _emit(on_event, {"type": "turn_start", "turn": turn_idx})
            response = await self._generate_with_retry(
                model=model, contents=contents, config=config, on_event=on_event, verbose=verbose
            )
            candidate = response.candidates[0]
            finish_reason = candidate.finish_reason.value if candidate.finish_reason else None
            parts = candidate.content.parts or []
            function_calls = [p.function_call for p in parts if p.function_call is not None]

            turn_log = {
                "turn": turn_idx,
                "finish_reason": finish_reason,
                "blocks": _parts_to_log(parts),
                "tool_results": [],
            }

            for b in turn_log["blocks"]:
                if verbose:
                    if b["type"] == "thinking":
                        print("  [pensando]", b["text"][:300])
                    elif b["type"] == "text":
                        print("  [texto]", b["text"])
                    elif b["type"] == "function_call":
                        print(f"  [chamada] {b['name']}({b['args']})")
                if b["type"] == "thinking":
                    await _emit(on_event, {"type": "thinking", "turn": turn_idx, "text": b["text"]})
                elif b["type"] == "text":
                    await _emit(on_event, {"type": "text", "turn": turn_idx, "text": b["text"]})
                elif b["type"] == "function_call":
                    await _emit(on_event, {
                        "type": "tool_call", "turn": turn_idx, "name": b["name"], "args": b["args"],
                    })

            contents.append(candidate.content)
            transcript["turns"].append(turn_log)

            if not function_calls:
                final_text = "\n".join(
                    p.text for p in parts if p.text and not p.thought
                ) or None
                break

            function_response_parts = []
            for fc in function_calls:
                result = await self.mcp_session.call_tool(fc.name, dict(fc.args or {}))
                text_parts = [
                    c.text for c in result.content if getattr(c, "type", None) == "text"
                ]
                result_text = "\n".join(text_parts) if text_parts else "(sem conteudo)"
                is_error = bool(getattr(result, "is_error", False))

                if verbose:
                    tag = "ERRO" if is_error else "resultado"
                    print(f"  [{tag}] {fc.name} -> {result_text}")
                await _emit(on_event, {
                    "type": "tool_result", "turn": turn_idx, "name": fc.name,
                    "args": dict(fc.args or {}), "result": result_text, "is_error": is_error,
                })

                turn_log["tool_results"].append({
                    "name": fc.name, "args": dict(fc.args or {}),
                    "result": result_text, "is_error": is_error,
                })
                response_payload = {"error": result_text} if is_error else {"result": result_text}
                function_response_parts.append(
                    types.Part.from_function_response(name=fc.name, response=response_payload)
                )

            contents.append(types.Content(role="user", parts=function_response_parts))

        transcript["finished_at"] = time.time()
        transcript["final_finish_reason"] = finish_reason
        transcript["final_text"] = final_text
        transcript["hit_max_turns"] = final_text is None
        await _emit(on_event, {
            "type": "final", "text": final_text, "hit_max_turns": transcript["hit_max_turns"],
        })

        if transcript_path:
            Path(transcript_path).parent.mkdir(parents=True, exist_ok=True)
            Path(transcript_path).write_text(
                json.dumps(transcript, indent=2, ensure_ascii=False), encoding="utf-8"
            )

        return transcript


async def run_task(
    task_prompt: str,
    *,
    model: str = _DEFAULT_MODEL,
    python_executable: str,
    mcp_server_script: str,
    max_turns: int = _MAX_TURNS,
    transcript_path: str | None = None,
    verbose: bool = True,
) -> dict:
    """Atalho p/ rodar uma unica tarefa (abre a sessao, roda, fecha).
    Para varias tarefas na mesma conexao, use GeminiAgentSession direto."""
    async with GeminiAgentSession(python_executable, mcp_server_script, model=model) as s:
        return await s.run_task(
            task_prompt,
            max_turns=max_turns,
            transcript_path=transcript_path,
            verbose=verbose,
        )
