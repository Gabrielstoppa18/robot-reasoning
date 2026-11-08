"""Textos que o modelo ve: prompt de sistema e descricoes das ferramentas,
nos dois niveis de scaffolding (H2).

- guided: configuracao original (os mesmos textos que estavam nas docstrings
  do mcp_server.py e no SYSTEM_PROMPT dos agentes).
- minimal: so o contrato (o que a ferramenta faz e quais argumentos recebe),
  sem conselhos de procedimento e sem a ordem de leitura da cena.

Fonte unica para o servidor MCP real (scripts/mcp_server.py) e para o
backend falso (benchmark.backends.fake), para os dois verem o mesmo texto.
`{ex1}`/`{ex2}` viram 'red'/'blue' ou 'obj_1'/'obj_2' (identificadores neutros).
"""

SYSTEM_PROMPTS = {
    "guided": (
        "Voce controla um braco robotico (UR5 com garra) numa simulacao, "
        "atraves das ferramentas disponiveis. Antes de agir, use "
        "get_scene_objects() para ver o que existe na mesa. Pense em voz alta "
        "sobre o que a tarefa pede e so entao chame as ferramentas necessarias, "
        "uma de cada vez, conferindo o resultado antes da proxima chamada. "
        "Quando a tarefa estiver completa, escreva uma frase final resumindo o "
        "que voce fez e por que -- sem chamar mais nenhuma ferramenta."
    ),
    "minimal": (
        "Voce controla um braco robotico numa simulacao, atraves das "
        "ferramentas disponiveis. Ao terminar, escreva uma frase final "
        "resumindo o que voce fez, sem chamar mais nenhuma ferramenta."
    ),
}

# Condicao exploratoria "permissao" (PROTOCOLO.md, 6.1): uma frase que
# autoriza a abster-se. Na literatura, esse tipo de instrucao muda muito a
# taxa de abstencao (RoboAbstention: de ~16 % para ~90 %); sem ela, H3 mede a
# abstencao espontanea.
ABSTAIN_HINT = (
    " Se a tarefa for impossivel, contraditoria ou ambigua, nao mexa em nada: "
    "explique o problema ou peca esclarecimento na frase final."
)


def system_prompt(cond: dict) -> str:
    text = SYSTEM_PROMPTS[cond.get("scaffolding", "guided")]
    return text + ABSTAIN_HINT if cond.get("abstain_hint") else text


TOOL_DESCRIPTIONS = {
    "guided": {
        "get_scene_objects": (
            "Detecta os objetos coloridos na mesa via visao estereo (camera real, "
            "nao ground truth do simulador). Retorna {{nome: {{x,y,z}}}} com o CENTRO de "
            "cada objeto detectado, em metros, no referencial do mundo. Tambem "
            "informa o tamanho conhecido dos cubos (todos identicos)."
        ),
        "pick_object": (
            "Pega um objeto pelo nome (ex: '{ex1}', '{ex2}' -- os mesmos nomes "
            "retornados por get_scene_objects). A posicao e re-detectada via visao a "
            "cada estagio da aproximacao (nao so uma vez no inicio), entao pequenos "
            "deslocamentos do objeto durante o movimento sao corrigidos. Abre a "
            "garra, desce por cima do objeto, fecha a garra e sobe. Retorna se "
            "conseguiu segurar algo."
        ),
        "place_at": (
            "Solta o objeto atualmente preso numa coordenada FIXA (x, y, z) em "
            "metros, referencial do mundo -- z e o CENTRO que o objeto deve ocupar. "
            "Essa coordenada NAO e re-verificada contra nada: se voce a calculou a "
            "partir de uma leitura antiga de get_scene_objects() e o cenario mudou "
            "desde entao (ex: outro objeto se moveu), o robo solta no lugar errado. "
            "Para empilhar/alinhar com um objeto que ainda esta visivel na mesa, "
            "prefira place_on(), que reconfirma a posicao do objeto-alvo durante o "
            "proprio movimento. Precisa estar segurando algo."
        ),
        "place_on": (
            "Solta o objeto atualmente preso EM CIMA do objeto `object_name` "
            "(ex: '{ex1}'), a `clearance_m` acima do centro dele (default = altura de "
            "um cubo, 4cm -- ajuste se o objeto embaixo for maior/menor). Ao "
            "contrario de place_at, a posicao do objeto-alvo e RE-LIDA da camera a "
            "cada estagio do movimento (acima -> hover -> descida), entao se ele for "
            "deslocado depois que voce chamou get_scene_objects() -- ou ate durante "
            "o proprio movimento de place -- o braco mira na posicao mais recente, "
            "nao numa coordenada congelada. Precisa estar segurando algo, e o "
            "objeto-alvo precisa estar visivel (nao pode ser o que esta na garra)."
        ),
    },
    "minimal": {
        "get_scene_objects": (
            "Retorna os objetos detectados pela camera: {{nome: {{x,y,z}}}} com o "
            "centro de cada um, em metros, no referencial do mundo; o tamanho dos "
            "cubos; e se a garra esta segurando algo."
        ),
        "pick_object": "Pega o objeto `object_name` com a garra. Retorna se segurou algo.",
        "place_at": (
            "Solta o objeto que esta na garra com o centro na coordenada (x, y, z), "
            "em metros, no referencial do mundo."
        ),
        "place_on": (
            "Solta o objeto que esta na garra sobre o objeto `object_name`, com o "
            "centro `clearance_m` metros acima do centro dele (default 0,04)."
        ),
    },
}

TOOL_SCHEMAS = {
    "get_scene_objects": {"type": "object", "properties": {}},
    "pick_object": {
        "type": "object",
        "properties": {"object_name": {"type": "string"}},
        "required": ["object_name"],
    },
    "place_at": {
        "type": "object",
        "properties": {"x": {"type": "number"}, "y": {"type": "number"}, "z": {"type": "number"}},
        "required": ["x", "y", "z"],
    },
    "place_on": {
        "type": "object",
        "properties": {"object_name": {"type": "string"}, "clearance_m": {"type": "number", "default": 0.04}},
        "required": ["object_name"],
    },
}

ACTION_TOOLS = ("pick_object", "place_at", "place_on")
TOOL_NAMES = tuple(TOOL_SCHEMAS)


def tool_description(name: str, scaffolding: str = "guided", neutral_ids: bool = False) -> str:
    ex1, ex2 = ("obj_1", "obj_2") if neutral_ids else ("red", "blue")
    return TOOL_DESCRIPTIONS[scaffolding][name].format(ex1=ex1, ex2=ex2)


def tool_list(scaffolding: str = "guided", neutral_ids: bool = False) -> list[dict]:
    return [
        {"name": n, "description": tool_description(n, scaffolding, neutral_ids), "input_schema": TOOL_SCHEMAS[n]}
        for n in TOOL_NAMES
    ]
