"""Interface comum dos agentes e formato do transcript do benchmark.

Diferente de llm_agent/*.py (usados pelo painel), os agentes daqui recebem
um ToolClient ja aberto pelo backend -- o mesmo agente roda contra o
simulador real ou contra o backend falso -- e registram, por turno, tempo de
resposta do modelo, tokens e tempo de cada chamada de ferramenta.

Transcript (um por episodio):
    {"agent", "model", "model_version", "system_prompt", "task_prompt",
     "turns": [{"turn", "t_model_s", "usage", "stop_reason",
                "blocks": [{"type": "thinking"|"text"|"tool_call", ...}],
                "tool_results": [{"name", "args", "result", "is_error", "t_s"}]}],
     "final_text", "hit_max_turns", "hit_max_tool_calls", "t_total_s", "usage_total"}
"""
import time

USAGE_KEYS = ("input_tokens", "output_tokens", "thinking_tokens", "cache_read_tokens")


class ProviderError(Exception):
    """Falha da API do provedor (cota, 5xx, rede) depois das tentativas.
    O episodio e marcado como api_error e reexecutado depois -- nao conta
    como falha do modelo (criterio de exclusao pre-definido)."""


class Transcript(dict):
    def __init__(self, agent, model, system_prompt, task_prompt):
        super().__init__(agent=agent, model=model, model_version=None, system_prompt=system_prompt,
                         task_prompt=task_prompt, turns=[], final_text=None, hit_max_turns=False,
                         hit_max_tool_calls=False, t_total_s=None,
                         usage_total={k: 0 for k in USAGE_KEYS})
        self._t0 = time.perf_counter()
        self.n_tool_calls = 0

    def new_turn(self, idx, t_model_s, usage, stop_reason, blocks):
        usage = {k: int(usage.get(k) or 0) for k in USAGE_KEYS}
        for k in USAGE_KEYS:
            self["usage_total"][k] += usage[k]
        turn = {"turn": idx, "t_model_s": round(t_model_s, 3), "usage": usage,
                "stop_reason": stop_reason, "blocks": blocks, "tool_results": []}
        self["turns"].append(turn)
        return turn

    async def call_tool(self, tools, turn, name, args):
        t0 = time.perf_counter()
        text, is_error = await tools.call(name, args)
        turn["tool_results"].append({"name": name, "args": args, "result": text, "is_error": is_error,
                                     "t_s": round(time.perf_counter() - t0, 3)})
        self.n_tool_calls += 1
        return text, is_error

    def finish(self, final_text):
        self["final_text"] = final_text
        self["t_total_s"] = round(time.perf_counter() - self._t0, 3)
        return self


def tool_calls(transcript) -> list[dict]:
    """Chamadas executadas, em ordem, com o resultado."""
    out = []
    for t in transcript["turns"]:
        for r in t["tool_results"]:
            out.append({**r, "turn": t["turn"]})
    return out


class Agent:
    kind = "base"

    def __init__(self, model: str, max_turns: int = 25, max_tool_calls: int = 40, **options):
        self.model = model
        self.max_turns = max_turns
        self.max_tool_calls = max_tool_calls
        self.options = options

    async def run(self, prompt: str, tools, system_prompt: str, spec: dict | None = None) -> Transcript:
        raise NotImplementedError

    def describe(self) -> dict:
        return {"kind": self.kind, "model": self.model, "max_turns": self.max_turns,
                "max_tool_calls": self.max_tool_calls, "options": self.options}
