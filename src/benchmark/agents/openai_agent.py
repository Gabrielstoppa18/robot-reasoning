"""Agente GPT (OpenAI SDK, Responses API), mesmo contrato de ferramentas.

O historico fica no servidor da OpenAI (previous_response_id), o que tambem
preserva os itens de raciocinio entre turnos. O resumo do raciocinio
(reasoning.summary) pode exigir verificacao da organizacao; se a API
recusar, configure `reasoning = {}` para o modelo no experiment.toml.
"""
import json
import time

from .base import Agent, ProviderError, Transcript


class OpenAIAgent(Agent):
    kind = "openai"

    def __init__(self, model, **kw):
        super().__init__(model, **kw)
        import openai
        self._openai = openai
        self.client = openai.AsyncOpenAI(max_retries=6)
        self.reasoning = self.options.get("reasoning", {"summary": "auto"})

    async def _create(self, **kw):
        try:
            return await self.client.responses.create(**kw)
        except (self._openai.APIStatusError, self._openai.APIConnectionError) as e:
            raise ProviderError(f"openai: {e}") from e

    async def run(self, prompt, tools, system_prompt, spec=None):
        tr = Transcript(self.kind, self.model, system_prompt, prompt)
        fn_tools = [{"type": "function", "name": t["name"], "description": t["description"],
                     "parameters": t["input_schema"], "strict": False} for t in tools.tools]
        common = dict(model=self.model, instructions=system_prompt, tools=fn_tools)
        if self.reasoning:
            common["reasoning"] = self.reasoning
        next_input = [{"role": "user", "content": prompt}]
        prev_id = None
        final_text = None
        for turn_idx in range(self.max_turns):
            t0 = time.perf_counter()
            kw = dict(common, input=next_input)
            if prev_id:
                kw["previous_response_id"] = prev_id
            resp = await self._create(**kw)
            prev_id = resp.id
            tr["model_version"] = resp.model
            u = resp.usage
            details = getattr(u, "output_tokens_details", None)
            in_details = getattr(u, "input_tokens_details", None)
            usage = {"input_tokens": u.input_tokens, "output_tokens": u.output_tokens,
                     "thinking_tokens": getattr(details, "reasoning_tokens", 0),
                     "cache_read_tokens": getattr(in_details, "cached_tokens", 0)}
            blocks, calls = [], []
            for item in resp.output:
                if item.type == "reasoning":
                    text = "\n".join(s.text for s in (item.summary or []))
                    blocks.append({"type": "thinking", "text": text})
                elif item.type == "message":
                    text = "".join(c.text for c in item.content if getattr(c, "type", "") == "output_text")
                    blocks.append({"type": "text", "text": text})
                elif item.type == "function_call":
                    try:
                        args = json.loads(item.arguments or "{}")
                    except json.JSONDecodeError:
                        args = {"_raw": item.arguments}
                    blocks.append({"type": "tool_call", "id": item.call_id, "name": item.name, "args": args})
                    calls.append((item.call_id, item.name, args))
            turn = tr.new_turn(turn_idx, time.perf_counter() - t0, usage, resp.status, blocks)
            if not calls:
                final_text = resp.output_text or None
                break
            next_input = []
            for call_id, name, args in calls:
                if tr.n_tool_calls >= self.max_tool_calls:
                    tr["hit_max_tool_calls"] = True
                    break
                text, is_error = await tr.call_tool(tools, turn, name, args)
                next_input.append({"type": "function_call_output", "call_id": call_id, "output": text})
            if tr["hit_max_tool_calls"]:
                break
        else:
            tr["hit_max_turns"] = True
        return tr.finish(final_text)
