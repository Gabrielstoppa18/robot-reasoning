"""Agente Claude (Anthropic SDK), loop agentico manual como em
llm_agent/claude_agent.py."""
import time

from .base import Agent, ProviderError, Transcript

_MAX_TOKENS = 16000


class ClaudeAgent(Agent):
    kind = "claude"

    def __init__(self, model, **kw):
        super().__init__(model, **kw)
        import anthropic
        self._anthropic = anthropic
        self.client = anthropic.AsyncAnthropic(max_retries=6)
        # pensamento: default igual ao do painel; sobrescrever por modelo no experiment.toml
        self.thinking = self.options.get("thinking", {"type": "adaptive", "display": "summarized"})
        self.max_tokens = self.options.get("max_tokens", _MAX_TOKENS)

    async def run(self, prompt, tools, system_prompt, spec=None):
        tr = Transcript(self.kind, self.model, system_prompt, prompt)
        claude_tools = [{"name": t["name"], "description": t["description"], "input_schema": t["input_schema"]}
                        for t in tools.tools]
        messages = [{"role": "user", "content": prompt}]
        final_text = None
        for turn_idx in range(self.max_turns):
            t0 = time.perf_counter()
            kwargs = dict(model=self.model, max_tokens=self.max_tokens, system=system_prompt,
                          tools=claude_tools, messages=messages)
            if self.thinking:
                kwargs["thinking"] = self.thinking
            try:
                response = await self.client.messages.create(**kwargs)
            except (self._anthropic.APIStatusError, self._anthropic.APIConnectionError) as e:
                raise ProviderError(f"anthropic: {e}") from e
            tr["model_version"] = response.model
            u = response.usage
            usage = {"input_tokens": u.input_tokens, "output_tokens": u.output_tokens,
                     "cache_read_tokens": getattr(u, "cache_read_input_tokens", 0)}
            blocks = []
            for b in response.content:
                if b.type == "text":
                    blocks.append({"type": "text", "text": b.text})
                elif b.type == "thinking":
                    blocks.append({"type": "thinking", "text": b.thinking})
                elif b.type == "tool_use":
                    blocks.append({"type": "tool_call", "id": b.id, "name": b.name, "args": b.input})
            turn = tr.new_turn(turn_idx, time.perf_counter() - t0, usage, response.stop_reason, blocks)
            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason != "tool_use":
                final_text = "\n".join(b.text for b in response.content if b.type == "text") or None
                break
            results = []
            for b in response.content:
                if b.type != "tool_use":
                    continue
                if tr.n_tool_calls >= self.max_tool_calls:
                    tr["hit_max_tool_calls"] = True
                    break
                text, is_error = await tr.call_tool(tools, turn, b.name, b.input)
                results.append({"type": "tool_result", "tool_use_id": b.id, "content": text, "is_error": is_error})
            if tr["hit_max_tool_calls"]:
                break
            messages.append({"role": "user", "content": results})
        else:
            tr["hit_max_turns"] = True
        return tr.finish(final_text)
