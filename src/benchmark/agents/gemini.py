"""Agente Gemini (google-genai SDK), loop agentico manual como em
llm_agent/gemini_agent.py (inclusive o retry de 5xx, que aqui fica
registrado no transcript)."""
import asyncio
import time

from .base import Agent, ProviderError, Transcript

_RETRYABLE_STATUS = {429, 500, 503, 504}
_MAX_RETRIES = 6
_HTTP_TIMEOUT_MS = 120_000  # sem isso, uma requisicao presa trava o episodio ate o timeout de 900 s


class GeminiAgent(Agent):
    kind = "gemini"

    def __init__(self, model, **kw):
        super().__init__(model, **kw)
        import httpx
        from google import genai
        from google.genai import errors, types
        self._errors, self._types, self._timeouts = errors, types, (httpx.TimeoutException,)
        self.client = genai.Client(http_options=types.HttpOptions(timeout=_HTTP_TIMEOUT_MS))

    async def _generate(self, tr, **kw):
        delay = 4.0
        for attempt in range(_MAX_RETRIES):
            try:
                return await self.client.aio.models.generate_content(**kw)
            except self._errors.APIError as e:
                retryable = e.code in _RETRYABLE_STATUS and attempt < _MAX_RETRIES - 1
                tr.setdefault("api_retries", []).append({"code": e.code, "attempt": attempt})
                if not retryable:
                    raise ProviderError(f"gemini {e.code}: {e.message}") from e
                await asyncio.sleep(delay)
                delay *= 2
            except self._timeouts as e:
                tr.setdefault("api_retries", []).append({"code": "timeout", "attempt": attempt})
                if attempt == _MAX_RETRIES - 1:
                    raise ProviderError(f"gemini timeout depois de {_MAX_RETRIES} tentativas") from e

    async def run(self, prompt, tools, system_prompt, spec=None):
        types = self._types
        tr = Transcript(self.kind, self.model, system_prompt, prompt)
        gtool = types.Tool(function_declarations=[
            types.FunctionDeclaration(name=t["name"], description=t["description"],
                                      parameters_json_schema=t["input_schema"])
            for t in tools.tools
        ])
        thinking = dict(self.options.get("thinking", {}))
        thinking.setdefault("include_thoughts", True)
        config = types.GenerateContentConfig(system_instruction=system_prompt, tools=[gtool],
                                             thinking_config=types.ThinkingConfig(**thinking))
        contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]
        final_text = None
        for turn_idx in range(self.max_turns):
            t0 = time.perf_counter()
            response = await self._generate(tr, model=self.model, contents=contents, config=config)
            tr["model_version"] = getattr(response, "model_version", None) or self.model
            um = response.usage_metadata
            usage = {"input_tokens": getattr(um, "prompt_token_count", 0),
                     "output_tokens": getattr(um, "candidates_token_count", 0),
                     "thinking_tokens": getattr(um, "thoughts_token_count", 0),
                     "cache_read_tokens": getattr(um, "cached_content_token_count", 0)}
            if not response.candidates:
                tr.new_turn(turn_idx, time.perf_counter() - t0, usage, "NO_CANDIDATE", [])
                break
            cand = response.candidates[0]
            parts = (cand.content.parts if cand.content else None) or []
            blocks = []
            for p in parts:
                if p.function_call is not None:
                    blocks.append({"type": "tool_call", "name": p.function_call.name,
                                   "args": dict(p.function_call.args or {})})
                elif p.thought:
                    blocks.append({"type": "thinking", "text": p.text or ""})
                elif p.text is not None:
                    blocks.append({"type": "text", "text": p.text})
            stop = cand.finish_reason.value if cand.finish_reason else None
            turn = tr.new_turn(turn_idx, time.perf_counter() - t0, usage, stop, blocks)
            contents.append(cand.content)

            calls = [p.function_call for p in parts if p.function_call is not None]
            if not calls:
                final_text = "\n".join(p.text for p in parts if p.text and not p.thought) or None
                break
            resp_parts = []
            for fc in calls:
                if tr.n_tool_calls >= self.max_tool_calls:
                    tr["hit_max_tool_calls"] = True
                    break
                text, is_error = await tr.call_tool(tools, turn, fc.name, dict(fc.args or {}))
                payload = {"error": text} if is_error else {"result": text}
                resp_parts.append(types.Part.from_function_response(name=fc.name, response=payload))
            if tr["hit_max_tool_calls"]:
                break
            contents.append(types.Content(role="user", parts=resp_parts))
        else:
            tr["hit_max_turns"] = True
        return tr.finish(final_text)
