"""Backends de execucao: o simulador real (CoppeliaSim + servidor MCP) e um
backend falso cinematico, com a mesma interface:

    backend.reset(spec)                    -> estado verdadeiro inicial
    async with backend.tools(spec, cond)   -> ToolClient (tools, call)
    backend.finalize(spec)                 -> estado verdadeiro final
    backend.events()                       -> eventos do servidor (soltura, perturbacao)
"""


class NameMap:
    """Traduz identificadores neutros (obj_1...) <-> cores. Sem mapa, identidade."""

    def __init__(self, neutral_map: dict | None):
        self.to_color = dict(neutral_map or {})
        self.to_id = {c: i for i, c in self.to_color.items()}

    @property
    def active(self) -> bool:
        return bool(self.to_color)

    def color(self, name):
        return self.to_color.get(name, name) if self.active else name

    def ident(self, color):
        return self.to_id.get(color, color) if self.active else color


def make_backend(kind: str, **kw):
    if kind == "fake":
        from .fake import FakeBackend
        return FakeBackend(**kw)
    if kind == "coppelia":
        from .coppelia import CoppeliaBackend
        return CoppeliaBackend(**kw)
    raise ValueError(f"backend desconhecido: {kind}")
