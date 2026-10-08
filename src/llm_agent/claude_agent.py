"""Fase 4 - agente LLM (Claude) controlando o robo via MCP.

Loop agentico manual (nao o tool_runner da SDK): queremos registrar tanto a
ENTRADA quanto a SAIDA de cada chamada de ferramenta no transcript (pra
analise na dissertacao), e manter o mesmo formato de log reutilizavel do
que o do gemini_agent.py.

`ClaudeAgentSession` mantem a conexao MCP (e o processo do mcp_server.py)
viva entre varias chamadas a run_task -- usado pelo dashboard web. A
funcao `run_task` solta no final do modulo e um atalho "uma tarefa so".
"""
import asyncio
import json
import time
from contextlib import AsyncExitStack
from pathlib import Path

import anthropic
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

_DEFAULT_MODEL = "claude-opus-5"
_MAX_TOKENS = 16000  # teto generoso pra nao truncar raciocinio+resposta no meio
_MAX_TURNS = 15


def _mcp_tools_to_claude(mcp_tools) -> list[dict]:
    return [
        {"name": t.name, "description": t.description or "", "input_schema": t.input_schema}
        for t in mcp_tools
    ]


def _blocks_to_log(content_blocks) -> list[dict]:
    out = []
    for b in content_blocks:
        if b.type == "text":
            out.append({"type": "text", "text": b.text})
        elif b.type == "thinking":
            out.append({"type": "thinking", "text": b.thinking})
        elif b.type == "tool_use":
            out.append({"type": "tool_use", "id": b.id, "name": b.name, "input": b.input})
    return out


async def _emit(on_event, event: dict):
    if on_event is None:
        return
    result = on_event(event)
    if asyncio.iscoroutine(result):
        await result


class ClaudeAgentSession:
    """Conexao persistente com o MCP server + client da Anthropic. Use
    como context manager async; `run_task` pode ser chamado varias vezes."""

    def __init__(self, python_executable: str, mcp_server_script: str, model: str = _DEFAULT_MODEL):
        self.python_executable = python_executable
        self.mcp_server_script = mcp_server_script
        self.model = model
        self._stack: AsyncExitStack | None = None
        self.mcp_session: ClientSession | None = None
        self.client: anthropic.AsyncAnthropic | None = None
        self._claude_tools: list[dict] | None = None

    async def __aenter__(self) -> "ClaudeAgentSession":
        self._stack = AsyncExitStack()
        await self._stack.__aenter__()
        params = StdioServerParameters(
            command=self.python_executable, args=[self.mcp_server_script]
        )
        read, write = await self._stack.enter_async_context(stdio_client(params))
        self.mcp_session = await self._stack.enter_async_context(ClientSession(read, write))
        await self.mcp_session.initialize()
        tools_result = await self.mcp_session.list_tools()
        self._claude_tools = _mcp_tools_to_claude(tools_result.tools)
        self.client = anthropic.AsyncAnthropic()
        return self

    async def __aexit__(self, *exc_info):
        await self._stack.__aexit__(*exc_info)
        self._stack = None

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
        """Roda `task_prompt` ate o Claude terminar (ou max_turns). Chama
        `on_event(dict)` (sync ou async) a cada evento -- turno, pensamento,
        texto, chamada/resultado de ferramenta, final."""
        model = model or self.model
        transcript = {
            "model": model,
            "task_prompt": task_prompt,
            "system_prompt": SYSTEM_PROMPT,
            "started_at": time.time(),
            "turns": [],
        }

        messages = [{"role": "user", "content": task_prompt}]
        final_text = None
        stop_reason = None

        for turn_idx in range(max_turns):
            await _emit(on_event, {"type": "turn_start", "turn": turn_idx})
            response = await self.client.messages.create(
                model=model,
                max_tokens=_MAX_TOKENS,
                system=SYSTEM_PROMPT,
                tools=self._claude_tools,
                thinking={"type": "adaptive", "display": "summarized"},
                messages=messages,
            )
            stop_reason = response.stop_reason
            turn_log = {
                "turn": turn_idx,
                "stop_reason": stop_reason,
                "blocks": _blocks_to_log(response.content),
                "tool_results": [],
            }

            for b in turn_log["blocks"]:
                if verbose:
                    if b["type"] == "thinking":
                        print("  [pensando]", b["text"][:300])
                    elif b["type"] == "text":
                        print("  [texto]", b["text"])
                    elif b["type"] == "tool_use":
                        print(f"  [chamada] {b['name']}({b['input']})")
                if b["type"] == "thinking":
                    await _emit(on_event, {"type": "thinking", "turn": turn_idx, "text": b["text"]})
                elif b["type"] == "text":
                    await _emit(on_event, {"type": "text", "turn": turn_idx, "text": b["text"]})
                elif b["type"] == "tool_use":
                    await _emit(on_event, {
                        "type": "tool_call", "turn": turn_idx, "name": b["name"], "args": b["input"],
                    })

            messages.append({"role": "assistant", "content": response.content})
            transcript["turns"].append(turn_log)

            if stop_reason != "tool_use":
                final_text = next(
                    (b.text for b in response.content if b.type == "text"), None
                )
                break

            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
            tool_results = []
            for tb in tool_use_blocks:
                result = await self.mcp_session.call_tool(tb.name, tb.input)
                text_parts = [
                    c.text for c in result.content if getattr(c, "type", None) == "text"
                ]
                result_text = "\n".join(text_parts) if text_parts else "(sem conteudo)"
                is_error = bool(getattr(result, "is_error", False))

                if verbose:
                    tag = "ERRO" if is_error else "resultado"
                    print(f"  [{tag}] {tb.name} -> {result_text}")
                await _emit(on_event, {
                    "type": "tool_result", "turn": turn_idx, "name": tb.name,
                    "args": tb.input, "result": result_text, "is_error": is_error,
                })

                turn_log["tool_results"].append(
                    {"name": tb.name, "input": tb.input, "result": result_text, "is_error": is_error}
                )
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tb.id,
                    "content": result_text,
                    "is_error": is_error,
                })

            messages.append({"role": "user", "content": tool_results})

        transcript["finished_at"] = time.time()
        transcript["final_stop_reason"] = stop_reason
        transcript["final_text"] = final_text
        transcript["hit_max_turns"] = final_text is None and stop_reason == "tool_use"
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
    Para varias tarefas na mesma conexao, use ClaudeAgentSession direto."""
    async with ClaudeAgentSession(python_executable, mcp_server_script, model=model) as s:
        return await s.run_task(
            task_prompt,
            max_turns=max_turns,
            transcript_path=transcript_path,
            verbose=verbose,
        )
