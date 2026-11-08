# Validacao do conjunto de instrucoes (split test, 56 instrucoes)

**Erros: 0** | Avisos: 4

## Avisos

- test-F3-fila-base: sem gemea direta (justificativa: A instrucao ja e explicita; a inferencia e espacial (referencial da base).)
- test-F3-mais-perto-que: sem gemea direta (justificativa: A instrucao ja e explicita; a inferencia e metrica (distancia a base).)
- test-F3-fila-inversa: sem gemea direta (justificativa: A instrucao ja e explicita; a ordem e citada do mais longe para o mais perto.)
- test-F4-longe-de-C: a heuristica 'primeiro mencionado' resolve todas as sementes (atalho lexical)

## Cobertura

- Familias: {'F1': 8, 'F2': 8, 'F3': 8, 'F4': 8, 'F5': 8, 'F6': 8, 'F7': 8}
- Dominios: {'elementos': 1, 'mecânica': 1, 'geografia': 1, 'natureza': 3, 'trânsito': 3, 'espacial': 8, 'restrição': 7, 'militar': 2, 'construção': 2, 'calendário': 1, 'seleção': 3, 'quantificação': 5, 'inviável': 5, 'ambígua': 3, 'animais': 1, 'astronomia': 1, 'vestuário': 1, 'cotidiano': 1, 'emergência': 1, 'alimentos': 1, 'símbolos nacionais': 1, 'óptica': 1, 'negação': 1, 'jogos': 1, 'correio': 1}
- Acoes: {'pôr sobre': 10, 'aproximar': 14, 'afastar': 6, 'ordenar': 6, 'trocar': 5, 'empilhar 3': 1, 'ordenar 4': 1, 'agrupar': 4, 'abster-se': 8, 'rodízio': 1}
- Categorias de abstencao (F7): {'referente_ausente': 2, 'contraditoria': 1, 'capacidade_ausente': 1, 'premissa_falsa': 1, 'inviabilidade_fisica': 1, 'intencao_subespecificada': 1, 'intencao_subjetiva': 1}
- F5 com 4 ou mais movimentos: 3
- Cor do objeto movido (indireta, todas as sementes): {'yellow': 33, 'blue': 23, 'purple': 24, 'red': 30, 'green': 22}
- Fracao empilhar/por sobre: 23%

## Instrucoes

| id | familia | dominio | acao | abstencao | movimentos | canonica | parafrases | eixos | gemea | lexical |
|---|---|---|---|---|---|---|---|---|---|---|
| test-F1-lava-agua | F1 | elementos | pôr sobre |  | 1 | 1 | 3 | base/lexical/contexto | 1 | 0/3 |
| test-F1-ima-prego | F1 | mecânica | aproximar |  | 1 | 1 | 3 | base/lexical/contexto | 1 | 0/3 |
| test-F1-vulcao-aldeia | F1 | geografia | afastar |  | 1 | 0 | 3 | base/lexical/contexto | 1 | 0/3 |
| test-F2-ceu-grama | F2 | natureza | pôr sobre |  | 1 | 1 | 3 | base/lexical/contexto | 1 | 0/3 |
| test-F2-semaforo | F2 | trânsito | pôr sobre |  | 1 | 0 | 3 | base/lexical/contexto | 1 | 0/3 |
| test-F3-longe-perto | F3 | espacial | aproximar |  | 1 | 0 | 3 | base/lexical/contexto | 1 | 0/3 |
| test-F3-fila-base | F3 | espacial | ordenar |  | 3 | 0 | 3 | base/lexical/contexto | 0 | 0/3 |
| test-F4-sobre-sem-mexer | F4 | restrição | pôr sobre |  | 1 | 0 | 3 | base/lexical/contexto | 1 | 2/3 |
| test-F4-fragil | F4 | restrição | aproximar |  | 1 | 0 | 3 | base/lexical/contexto | 1 | 0/3 |
| test-F5-troca-guarda | F5 | militar | trocar |  | 3 | 0 | 3 | base/lexical/contexto | 1 | 0/3 |
| test-F5-casa | F5 | construção | empilhar 3 |  | 2 | 1 | 3 | base/lexical/contexto | 1 | 0/3 |
| test-F5-estacoes | F5 | calendário | ordenar 4 |  | 4 | 0 | 3 | base/lexical/contexto | 1 | 0/3 |
| test-F6-afastar-vizinho | F6 | seleção | afastar |  | 1 | 0 | 3 | base/lexical/contexto | 1 | 0/3 |
| test-F6-todos-perto | F6 | quantificação | agrupar |  | 1 | 0 | 3 | base/lexical/contexto | 1 | 0/3 |
| test-F7-cor-ausente | F7 | inviável | abster-se | referente_ausente | 0 | 0 | 3 | base/lexical/contexto | 0 | 0/3 |
| test-F7-contraditoria | F7 | inviável | abster-se | contraditoria | 0 | 0 | 3 | base/lexical/contexto | 0 | 0/3 |
| test-F7-mais-pesado | F7 | ambígua | abster-se | capacidade_ausente | 0 | 0 | 3 | base/lexical/contexto | 0 | 0/3 |
| test-F7-premissa-falsa | F7 | inviável | abster-se | premissa_falsa | 0 | 0 | 3 | base/lexical/contexto | 0 | 0/3 |
| test-F7-dentro | F7 | inviável | abster-se | inviabilidade_fisica | 0 | 0 | 3 | base/lexical/contexto | 0 | 0/3 |
| test-F7-aquele-ali | F7 | ambígua | abster-se | intencao_subespecificada | 0 | 0 | 3 | base/lexical/contexto | 0 | 3/3 |
| test-F7-cor-favorita | F7 | ambígua | abster-se | intencao_subjetiva | 0 | 0 | 3 | base/lexical/contexto | 0 | 0/3 |
| test-F1-gato-rato | F1 | animais | afastar |  | 1 | 1 | 3 | base/lexical/contexto | 1 | 0/3 |
| test-F1-terra-lua | F1 | astronomia | aproximar |  | 1 | 1 | 3 | base/lexical/contexto | 1 | 0/3 |
| test-F1-boneco-chapeu | F1 | vestuário | pôr sobre |  | 1 | 0 | 3 | base/lexical/contexto | 1 | 1/3 |
| test-F1-fila-onibus | F1 | cotidiano | ordenar |  | 1 | 0 | 3 | base/lexical/contexto | 1 | 0/3 |
| test-F1-bombeiros | F1 | emergência | aproximar |  | 1 | 0 | 3 | base/lexical/contexto | 1 | 0/3 |
| test-F2-fruteira | F2 | alimentos | aproximar |  | 1 | 0 | 3 | base/lexical/contexto | 1 | 0/3 |
| test-F2-mar-morango | F2 | natureza | afastar |  | 1 | 0 | 3 | base/lexical/contexto | 1 | 0/3 |
| test-F2-bandeira | F2 | símbolos nacionais | pôr sobre |  | 1 | 1 | 3 | base/lexical/contexto | 1 | 0/3 |
| test-F2-sinal-atencao | F2 | trânsito | ordenar |  | 1 | 0 | 3 | base/lexical/contexto | 1 | 0/3 |
| test-F2-sangue-folha | F2 | natureza | trocar |  | 3 | 0 | 3 | base/lexical/contexto | 1 | 0/3 |
| test-F2-arco-iris | F2 | óptica | ordenar |  | 3 | 1 | 3 | base/lexical/contexto | 1 | 0/3 |
| test-F3-mais-perto-que | F3 | espacial | ordenar |  | 1 | 0 | 3 | base/lexical/contexto | 0 | 0/3 |
| test-F3-longe-sobre-perto | F3 | espacial | pôr sobre |  | 1 | 0 | 3 | base/lexical/contexto | 1 | 0/3 |
| test-F3-vizinho-base | F3 | espacial | aproximar |  | 1 | 0 | 3 | base/lexical/contexto | 1 | 0/3 |
| test-F3-fila-inversa | F3 | espacial | ordenar |  | 3 | 0 | 3 | base/lexical/contexto | 0 | 0/3 |
| test-F3-junto-do-mais-longe | F3 | espacial | aproximar |  | 1 | 0 | 3 | base/lexical/contexto | 1 | 0/3 |
| test-F3-troca-extremos | F3 | espacial | trocar |  | 3 | 0 | 3 | base/lexical/contexto | 1 | 0/3 |
| test-F4-sem-empilhar | F4 | restrição | aproximar |  | 1 | 0 | 3 | base/lexical/contexto | 1 | 0/3 |
| test-F4-so-move-B | F4 | restrição | aproximar |  | 1 | 0 | 3 | base/lexical/contexto | 1 | 0/3 |
| test-F4-sem-encostar | F4 | restrição | afastar |  | 1 | 0 | 3 | base/lexical/contexto | 1 | 0/3 |
| test-F4-negacao | F4 | negação | pôr sobre |  | 1 | 0 | 3 | base/lexical/contexto | 1 | 0/3 |
| test-F4-longe-de-C | F4 | restrição | aproximar |  | 1 | 0 | 3 | base/lexical/contexto | 1 | 3/3 |
| test-F4-todos-menos | F4 | restrição | agrupar |  | 1 | 0 | 3 | base/lexical/contexto | 1 | 0/3 |
| test-F5-danca-cadeiras | F5 | jogos | rodízio |  | 4 | 0 | 3 | base/lexical/contexto | 1 | 0/3 |
| test-F5-revista | F5 | militar | trocar |  | 4 | 0 | 3 | base/lexical/contexto | 1 | 0/3 |
| test-F5-inverter-pilha | F5 | construção | pôr sobre |  | 1 | 0 | 3 | base/lexical/contexto | 1 | 0/3 |
| test-F5-estacionamento | F5 | trânsito | trocar |  | 3 | 0 | 3 | base/lexical/contexto | 1 | 0/3 |
| test-F5-carteiro | F5 | correio | aproximar |  | 1 | 0 | 3 | base/lexical/contexto | 1 | 0/3 |
| test-F6-dois-mais-perto | F6 | seleção | agrupar |  | 2 | 0 | 3 | base/lexical/contexto | 1 | 0/3 |
| test-F6-exceto-dois | F6 | quantificação | agrupar |  | 1 | 0 | 3 | base/lexical/contexto | 1 | 0/3 |
| test-F6-contar | F6 | quantificação | pôr sobre |  | 1 | 0 | 3 | base/lexical/contexto | 1 | 0/3 |
| test-F6-o-mais-perto-de-B | F6 | seleção | aproximar |  | 1 | 0 | 3 | base/lexical/contexto | 1 | 0/3 |
| test-F6-afastar-todos | F6 | quantificação | afastar |  | 2 | 0 | 3 | base/lexical/contexto | 1 | 0/3 |
| test-F6-qualquer-um | F6 | quantificação | aproximar |  | 1 | 0 | 3 | base/lexical/contexto | 1 | 0/3 |
| test-F7-caixa | F7 | inviável | abster-se | referente_ausente | 0 | 0 | 3 | base/lexical/contexto | 0 | 0/3 |