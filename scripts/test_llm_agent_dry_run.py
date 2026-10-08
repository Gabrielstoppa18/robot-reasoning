"""Testa o loop do agente Claude SEM gastar chamada de API real: substitui
client.messages.create por respostas falsas simulando uma trajetoria
plausivel (ver cena -> pegar azul -> empilhar sobre vermelho -> resumo),
mas executa as tool calls DE VERDADE contra o CoppeliaSim, validando a
ponte MCP inteira."""
import sys, os, asyncio, json
sys.path.insert(0, 'src')

from unittest.mock import AsyncMock, patch
import anthropic
from llm_agent import claude_agent


class FakeBlock:
    def __init__(self, type, **kw):
        self.type = type
        for k, v in kw.items():
            setattr(self, k, v)


class FakeResponse:
    def __init__(self, content, stop_reason):
        self.content = content
        self.stop_reason = stop_reason


# trajetoria falsa: le a cena, pega o azul, empilha sobre o vermelho, resume
FAKE_TURNS = [
    FakeResponse(
        [
            FakeBlock('thinking', thinking='Preciso ver o que tem na mesa primeiro.'),
            FakeBlock('tool_use', id='t1', name='get_scene_objects', input={}),
        ],
        'tool_use',
    ),
    FakeResponse(
        [
            FakeBlock('thinking', thinking='Vermelho=fogo, azul=agua. Vou pegar o azul.'),
            FakeBlock('tool_use', id='t2', name='pick_object', input={'object_name': 'blue'}),
        ],
        'tool_use',
    ),
    FakeResponse(
        [
            FakeBlock('tool_use', id='t3', name='place_on', input={'object_name': 'red'}),
        ],
        'tool_use',
    ),
    FakeResponse(
        [FakeBlock('text', text='Coloquei o cubo azul (agua) sobre o vermelho (fogo), apagando-o.')],
        'end_turn',
    ),
]


async def main():
    call_count = {'n': 0}

    async def fake_create(**kwargs):
        i = call_count['n']
        call_count['n'] += 1
        print(f'(mock) chamada {i} -- {len(kwargs["messages"])} mensagens no historico')
        return FAKE_TURNS[i]

    with patch.object(anthropic.AsyncAnthropic, '__init__', lambda self, **kw: None):
        client_instance = anthropic.AsyncAnthropic.__new__(anthropic.AsyncAnthropic)
        # patch so client.messages.create aponta pro fake, preservando o resto
        import types
        client_instance.messages = types.SimpleNamespace(create=fake_create)
        with patch('llm_agent.claude_agent.anthropic.AsyncAnthropic', return_value=client_instance):
            transcript = await claude_agent.run_task(
                'O cubo vermelho representa lava e o azul representa agua. Apague o fogo.',
                model='FAKE-dry-run',
                python_executable=sys.executable,
                mcp_server_script='scripts/mcp_server.py',
                max_turns=10,
                transcript_path='docs/transcripts/_dry_run.json',
            )

    print('\n=== RESULTADO ===')
    print('final_text:', transcript['final_text'])
    print('num turnos:', len(transcript['turns']))
    print('hit_max_turns:', transcript['hit_max_turns'])

asyncio.run(main())
