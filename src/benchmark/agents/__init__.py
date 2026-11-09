"""Fabrica de agentes: provedores LLM e baselines."""


def make_agent(provider: str, model: str | None = None, **options):
    if provider == "claude":
        from .claude import ClaudeAgent
        return ClaudeAgent(model, **options)
    if provider == "gemini":
        from .gemini import GeminiAgent
        return GeminiAgent(model, **options)
    if provider == "openai":
        from .openai_agent import OpenAIAgent
        return OpenAIAgent(model, **options)
    if provider == "baseline":
        from .scripted import FirstMentionedAgent, NoopAgent, OracleAgent
        cls = {"noop": NoopAgent, "first_mentioned": FirstMentionedAgent, "oracle": OracleAgent}[model]
        return cls(**options)
    raise ValueError(f"provedor desconhecido: {provider}")
