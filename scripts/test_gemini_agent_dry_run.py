"""Testa o loop do agente Gemini SEM gastar chamada de API real -- mesma
ideia do test_llm_agent_dry_run.py (Claude): substitui
client.aio.models.generate_content por respostas falsas simulando uma
trajetoria plausivel, mas executa as tool calls DE VERDADE contra o
CoppeliaSim, validando a ponte MCP + conversao de tipos da SDK do Gemini."""
import sys, asyncio
sys.path.insert(0, 'src')

from unittest.mock import patch
from google.genai import types
from llm_agent import gemini_agent


def _resp(parts, finish_reason=types.FinishReason.STOP):
    content = types.Content(role="model", parts=parts)
    candidate = types.Candidate(content=content, finish_reason=finish_reason)
    return types.GenerateContentResponse(candidates=[candidate])


FAKE_TURNS = [
    _resp([
        types.Part(text='Preciso ver o que tem na mesa primeiro.', thought=True),
        types.Part(function_call=types.FunctionCall(name='get_scene_objects', args={})),
    ]),
    _resp([
        types.Part(text='Vermelho=fogo, azul=agua. Vou pegar o azul.', thought=True),
        types.Part(function_call=types.FunctionCall(name='pick_object', args={'object_name': 'blue'})),
    ]),
    _resp([
        types.Part(function_call=types.FunctionCall(name='place_on', args={'object_name': 'red'})),
    ]),
    _resp([
        types.Part(text='Coloquei o cubo azul (agua) sobre o vermelho (fogo), apagando-o.'),
    ]),
]


async def main():
    call_count = {'n': 0}

    async def fake_generate_content(*, model, contents, config):
        i = call_count['n']
        call_count['n'] += 1
        print(f'(mock) chamada {i} -- {len(contents)} itens no historico')
        return FAKE_TURNS[i]

    with patch('llm_agent.gemini_agent.genai.Client') as MockClient:
        instance = MockClient.return_value
        instance.aio.models.generate_content = fake_generate_content

        transcript = await gemini_agent.run_task(
            'O cubo vermelho representa lava e o azul representa agua. Apague o fogo.',
            model='FAKE-dry-run',
            python_executable=sys.executable,
            mcp_server_script='scripts/mcp_server.py',
            max_turns=10,
            transcript_path='docs/transcripts/_dry_run_gemini.json',
        )

    print('\n=== RESULTADO ===')
    print('final_text:', transcript['final_text'])
    print('num turnos:', len(transcript['turns']))
    print('hit_max_turns:', transcript['hit_max_turns'])

asyncio.run(main())
