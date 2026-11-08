"""Benchmark de raciocinio semantico (Cap. 6 da qualificacao).

Pipeline: conjunto de instrucoes (benchmark/tasks) -> plano de episodios
(schedule) -> reinicio da cena por semente (backends) -> agente LLM ou
baseline (agents) -> estado verdadeiro final -> metricas (scoring) ->
analise estatistica (analysis). Ver benchmark/PROTOCOLO.md.
"""
