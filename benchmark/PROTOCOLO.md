# Protocolo de validação e avaliação

Este documento fixa os parâmetros do experimento proposto no Capítulo 6 da qualificação: as tarefas, os modelos, o número de execuções, os critérios de acerto, as regras de exclusão e a análise. O código que implementa cada item está indicado entre parênteses. Tudo o que for decidido depois de ver os dados do conjunto de teste é **exploratório** e deve ser relatado como tal.

**Objeto de estudo.** A classe das instruções semanticamente indiretas, e não um exemplo. A instrução-piloto (lava/água) é uma instância do conjunto de desenvolvimento e não entra nas conclusões. A **instrução** é a unidade de generalização; o episódio, não.

---

## 1. Hipóteses e o que as testa

| Hipótese | Contraste | Condições e dados | Teste principal |
|---|---|---|---|
| H1, custo semântico | indireta × gêmea direta (mesma meta, mesma cena) | `guided`, F1–F6 e suas gêmeas F0 | GLMM `S ~ modelo * indireta + (1 + indireta \| instrução) + (1 \| célula)` |
| H2, dependência de *scaffolding* | prompt e descrições guiados × mínimos | `guided` × `minimal`, indiretas | GLMM `S ~ modelo * mínima + …` |
| H3, inviáveis ou ambíguas | agir indevidamente × abster-se × relato | F7 em `guided` e `minimal` | proporções com Wilson; Fisher exato entre modelos (Holm) |
| H4, perturbação | `place_on` (relê o alvo) × `place_at` (coordenada congelada) | `perturb` | S por primitiva; taxa de escolha de `place_on` por modelo |
| H5, forma superficial | paráfrase, cor do objeto movido, canônica × nova | `guided`, indiretas | razão de verossimilhança no GLMM e componente de variância da instrução |

Exploratório, sem hipótese direcional: distribuição das falhas pelos elos da cadeia (atribuição, Seção 7.4), desempenho por família e por categoria de abstenção (F7), efeito dos identificadores neutros (`neutral`) e efeito de uma frase que autoriza a abster-se (`permissao`, Seção 6.1). H3 mede a abstenção **espontânea**: nenhum dos prompts das condições principais menciona a possibilidade de recusar ou pedir esclarecimento.

---

## 2. Portões de validação do sistema (antes de qualquer episódio com LLM)

Um LLM só é avaliado se a camada abaixo dele for confiável; do contrário, falha de percepção ou de controle seria contada como falha de raciocínio. Os portões rodam no simulador real, nesta ordem, e o resultado de cada um é arquivado em `benchmark/results/`.

| Portão | Comando | Critério de aprovação | Se reprovar |
|---|---|---|---|
| G0 – testes offline | `pytest tests -q` | 100 % dos testes | corrigir o código |
| G1 – percepção | `bench_check_perception.py --scenes 40` | 100 % dos cubos detectados, nenhuma cor espúria, erro no plano ≤ 5 mm (máx.) e ≤ 3 mm (média), por cor | ajustar faixas HSV (`vision/color_detect.py`) ou a área de trabalho; repetir |
| G2 – reinício da cena | 2 execuções do oráculo com a mesma semente | estado inicial verdadeiro idêntico (≤ 1 mm) e o runner nunca acusa "cena inicial não bate" | corrigir o *harness* |
| G3 – execução (oráculo) | `bench_run.py --config benchmark/pilot.toml --models base-oraculo` | S do oráculo ≥ 95 % em cada família F1–F6 e 100 % em `perturb` com `place_on` | ajustar controle/predicado/tolerância **antes** do congelamento |
| G4 – baselines | idem, `base-noop,base-primeiro-mencionado` | `noop`: S = 0 em F1–F6 (cena não trivial por construção) | revisar a instrução ou a restrição de cena |

O oráculo conhece os papéis (informação privilegiada), mas usa as mesmas ferramentas e a mesma percepção do LLM. Sua taxa de sucesso é o **teto de execução**: diferenças entre um LLM e o oráculo vêm do raciocínio ou do uso da interface, não do braço.

---

## 3. Tarefas: o conjunto de instruções

### 3.1 Famílias e tamanho

Famílias F1–F7 do Quadro 6.1 (F0 existe só como gêmea direta). Cada instrução indireta de F1–F6 tem gêmea direta com o mesmo predicado de sucesso; F3 pode dispensar a gêmea quando a instrução já é explícita, com o campo `no_twin_reason`.

| Conjunto | F1–F6 (por família) | F7 | Total | Gêmeas diretas | Situação |
|---|---|---|---|---|---|
| Desenvolvimento (`dev`) | 1 | 1 | 7 | 6 | escrito, descartável (só piloto e depuração) |
| Teste (`test`) | **8** | **8** | **56** | 45 | escrito (`tasks/instructions.json`, versão test-1.0-rascunho); 56 revisadas por Gabriel Stoppa |

**Origem do conjunto de teste (registrar no Cap. 6).** As 21 instruções da versão dev-0.2, já revisadas, foram promovidas para o teste **antes de qualquer piloto com LLM**; o único contato de LLM com elas foram 5 episódios de calibração de custo (2026-10-08: `lava-agua` ×2, `cor-ausente`, `mais-pesado`, `vulcao-aldeia`, todos no backend falso), usados só para medir tokens, sem nenhuma alteração motivada pelo resultado. Uma delas (`test-F1-ima-prego`) teve o predicado endurecido depois, para remover um atalho lexical, e voltou para revisão. As outras 35 de teste e as 7 de desenvolvimento foram rascunhadas com auxílio de LLM (Claude Opus 5.5, que não é avaliado; `author` registra isso) e revisadas pelo pesquisador. Os ids do teste começam com `test-`, e as cenas de cada instrução dependem do id.

**Por que 56.** A simulação de poder (`scripts/bench_power.py`, Seção 8.4) indica que, com 3 sementes e cerca de 48 instruções "act", uma diferença pareada de 15 p.p. (H1, H2) é detectada com poder ≈ 0,85 a α = 0,05, e uma de 20 p.p. com poder ≈ 0,9 mesmo com correção de Holm para até 6 contrastes. Com 2 ou 3 LLMs há 1 a 3 comparações par a par, o que reduz a correção e aumenta o poder por comparação. Com as 40 instruções citadas no Cap. 6, o poder para 15 p.p. cai para ≈ 0,75. **Recomenda-se atualizar o Cap. 6 para 56 instruções de teste.**

**Ancoragem na literatura.** O levantamento em [`LITERATURA.md`](LITERATURA.md) liga cada família a trabalhos anteriores: F2 a *symbol understanding* (RT-2) e *Common Sense* (EmbodiedBench); F3 a *Spatial Awareness* (EmbodiedBench) e LoHoRavens; F5 a *Long-Horizon* (SayCan) e LoHoRavens; F6 a LoHoRavens e KnowNo; F7 a KnowNo, AmbiK e RoboAbstention; F0 aos pares ambíguo × não ambíguo do AmbiK. F1 (papéis simbólicos atribuídos a objetos neutros) e F4 com restrições viáveis não têm equivalente direto e são a contribuição original do conjunto.

### 3.2 Regras de construção (verificadas por `bench_validate_tasks.py`)

1. **Cobertura sistemática.** ≥ 6 domínios semânticos no teste; nenhum domínio com mais de 25 % das instruções.
2. **Ações variadas.** Aproximar, afastar, pôr sobre, empilhar 3, ordenar, trocar, agrupar; "pôr sobre/empilhar" ≤ 25 % das instruções "act".
3. **Papéis balanceados.** A cor do primeiro papel gira entre as sementes (quadrado latino sobre as 5 cores); os demais papéis e os distratores são sorteados. A ordem de menção do objeto movido e do alvo varia entre as paráfrases. Cores fixas só em F2, quando o mapeamento é a própria inferência (céu = azul).
4. **Gêmeas diretas** com o mesmo predicado, a mesma cena e os mesmos papéis (pareamento exato).
5. **Três paráfrases por eixo** (`paraphrase_axes`, como no LIBERO-Para e no *Complex Instruction* do EmbodiedBench): `base` (redação de referência), `lexical` (troca o verbo e/ou a forma de se referir aos objetos, sem contexto extra) e `contexto` (a redação `base` precedida de uma frase irrelevante que não dá pista da solução). A paráfrase gira com a semente, deslocada por instrução: com 3 sementes cada instrução usa os três eixos, e o eixo não fica preso ao índice da semente.
6. **Redação e revisão independentes.** Autor e revisor distintos (`author`, `reviewed_by`); o revisor verifica ambiguidade não intencional, ausência de nomes de ferramenta e de dicas de procedimento, e se o predicado corresponde ao pedido. O autor do sistema não pode ser o único redator do teste. Se um LLM for usado para rascunhar instruções, ele **não** pode ser um dos modelos avaliados (o conjunto favoreceria quem o escreveu), e isso fica registrado no campo `author`.
7. **Cenas não triviais.** O estado inicial nunca satisfaz a meta (o sorteio de posições é repetido até isso valer).
8. **Solubilidade.** O plano de referência do oráculo satisfaz o predicado em todas as sementes (no backend falso, durante a validação, e no real, no portão G3).
9. **Sem atalho lexical.** Instruções que a heurística "primeiro mencionado" resolve em todas as sementes são sinalizadas; o pesquisador reescreve a instrução ou justifica mantê-la.
10. **Cenários novos.** Pelo menos metade do teste com cenários não canônicos (`canonical: false`), para comparar com os canônicos (contaminação/memorização).
11. **F7 por categoria.** Cada instrução F7 tem `abstain_type`, numa das categorias da RoboAbstention viáveis na cena: `referente_ausente`, `intencao_subespecificada`, `intencao_subjetiva`, `inviabilidade_fisica`, `capacidade_ausente`, `contraditoria`, `premissa_falsa`. O teste cobre pelo menos 5 categorias. "Referente ambíguo" fica de fora (Seção 11).
12. **Horizonte mais longo em F5.** Pelo menos 2 instruções F5 do teste exigem 4 ou mais movimentos (a literatura chama de longo horizonte a partir de 5; o limite aqui é a cena, com até 5 cubos).

### 3.3 Cena por semente

| Parâmetro | Valor | Onde |
|---|---|---|
| Objetos por cena | 2 a 5 cubos de 4 cm (por instrução, `n_objects`) | `instructions.json` |
| Cores | vermelho, azul, verde, amarelo, roxo | `palette.py`, `color_detect.py` |
| Área de trabalho | x ∈ [0,20; 0,55] m, y ∈ [0,02; 0,42] m (campo comum das duas câmeras, com margem) | `experiment.toml` |
| Separação mínima | 10 cm entre centros (dedos abertos da RG2) | idem |
| Perturbação | alvo deslocado 10–16 cm em direção sorteada, no passo 50 da 1.ª chamada `place_*` que move o braço | `tasks.py`, `mcp_server.py` |
| Identificadores neutros | `obj_1…obj_n` sorteados por semente; a percepção devolve a cor como atributo | `tasks.py`, `mcp_server.py` |

Tudo depende só de (semente global, instrução, índice da semente): **todos os modelos e condições veem exatamente a mesma cena** para a mesma célula.

---

## 4. Congelamento (pré-registro)

Antes do experimento principal, e depois do piloto:

1. Escrever e revisar o conjunto de teste; `bench_validate_tasks.py --config benchmark/experiment.toml` sem erros.
2. Confirmar os IDs de modelo (campos `CONFIRMAR` no `experiment.toml`) e preencher os preços da data.
3. `bench_freeze.py --config benchmark/experiment.toml` → grava `benchmark/FROZEN.json` (hash do conjunto de teste, do `experiment.toml` e das tolerâncias).
4. Commit e etiqueta: `git tag pre-registro-v1`. Opcional: depositar `FROZEN.json` e este protocolo no OSF.

Depois do congelamento, **nenhum** ajuste de prompt, ferramenta, verificador, tolerância ou instrução. O runner recusa o split `test` se o hash não bater (a opção `--allow-unfrozen` existe para emergências e fica registrada no manifesto). O conjunto de teste só é publicado depois da coleta.

---

## 5. Modelos

| Rótulo | Provedor | Modelo | Porte | Observação |
|---|---|---|---|---|
| `claude-pequeno` | Anthropic | `claude-haiku-4-5-20251001` | pequeno | **fora por custo** (2026-10-09); a configuração fica comentada no `experiment.toml` |
| `gemini-pequeno` | Google | `gemini-3.8-flash` | pequeno | usado no piloto informal (E1–E3) |
| `gpt-pequeno` | OpenAI | **a confirmar** | pequeno | opcional: só entra se houver conta paga e depois de testar o adaptador (`agents/openai_agent.py`), que ainda não rodou contra a API |
| `base-noop`, `base-primeiro-mencionado`, `base-oraculo` | — | — | — | baselines sem API (Seção 2) |

**Recorte por custo.** A pesquisa é custeada pelo autor, sem financiamento; por isso o experimento usa só modelos de pequeno porte e baixo custo (decisão de 2026-10-08, antes do congelamento). Os modelos grandes (linhas Opus, Pro e equivalentes) ficam fora, e as conclusões valem para modelos pequenos (Seção 8.3). Custo medido no piloto de calibração: ≈ US$ 0,007 por episódio no `gemini-3.8-flash`. Em 2026-10-09 o Claude Haiku também saiu, pelo custo da API; o estudo fica com **um LLM** (`gemini-pequeno`) e as baselines. Consequências: H1, H2, H4 e H5 são testadas para esse modelo (o `glmm.R` retira os termos de modelo quando há um só); as comparações entre modelos (Seção 8 do relatório, Fisher entre modelos em H3/H4, interações modelo × condição) não se aplicam; e as conclusões valem para esse modelo, não para LLMs pequenos em geral.

Regras: mesmas ferramentas, mesmo prompt de sistema e mesma instrução para todos; temperatura e amostragem no **padrão de cada provedor** (vários modelos de raciocínio não aceitam alterá-las); configuração de pensamento fixada no `experiment.toml` e registrada; a versão devolvida pela API (`model_version`) é gravada em cada episódio; execuções concentradas numa janela curta, com a ordem intercalada (Seção 6.2).

---

## 6. Número de execuções

### 6.1 Condições

| Condição | Variantes | Instruções | Sementes | Episódios por modelo (teste) |
|---|---|---|---|---|
| `guided` | indireta + gêmea direta | todas (56 + ~48 gêmeas) | 3 | ~312 |
| `minimal` | indireta | todas (56) | 3 | 168 |
| `perturb` | indireta | as que têm `perturb` (alvo de `place_on`), ~16 | 3 | ~48 |
| `neutral` | indireta | 50 % de F1, F2, F4, F5, F6 (~20) | 3 | ~60 |
| `permissao` (exploratória) | indireta, prompt guiado + frase que autoriza abster-se | todo o F7 (8) + 25 % das "act" (~12) | 3 | ~60 |
| réplicas | mesma célula de `guided`, 2.ª vez | 10 % das células `guided` | — | ~31 |
| **Total** | | | | **≈ 620 por LLM** (618 no plano atual) |

Com um LLM, ≈ 620 episódios (618 no plano atual); baselines ≈ 590 cada (sem réplicas). A 1–2 min por episódio, ≈ 10–20 h de simulação para o LLM; o piloto mede o tempo real. Custo estimado ≈ US$ 4 (`budget_usd = 25` no `experiment.toml`, folga para repetições). Se o orçamento apertar, cortam-se primeiro réplicas, `permissao` e `neutral` — **nunca o número de instruções distintas**, que é o que sustenta a generalização. O núcleo inegociável é `guided` + `minimal` (≈ 480 por modelo).

### 6.2 Ordem e pareamento

O plano (`schedule.py`) é pareado (todo modelo roda toda célula) e aleatorizado em blocos: cada célula é um bloco com todos os modelos em ordem sorteada, e os blocos são embaralhados com a semente global. Assim, mudanças de disponibilidade ou de versão ao longo dos dias não se confundem com diferenças entre provedores ou famílias.

### 6.3 Limites por episódio

| Limite | Valor | Ao atingir |
|---|---|---|
| Turnos de modelo | 25 | episódio válido; falha atribuída a "limite" se o plano estava incompleto |
| Chamadas de ferramenta | 40 | idem |
| Tempo de parede | 900 s | `infra_error` (simulador travado), reexecutado |
| Tentativas por episódio | 3 | episódio perdido, reportado na Seção 0 do relatório |
| Orçamento | `budget_usd` | o runner para |

### 6.4 Piloto (conjunto de desenvolvimento)

`benchmark/pilot.toml`: as 7 instruções de desenvolvimento (uma por família, nenhuma do teste), 2 sementes, `guided` + `minimal` + `perturb` e `permissao` (1 semente cada), um modelo pequeno por provedor e as baselines — 45 episódios por modelo, 225 no total. Como o dev é pequeno, o piloto serve para depurar o pipeline e medir custo e tempo; as regras de teto e piso abaixo ficam indicativas (uma instrução por família). Regras de decisão do piloto, **todas antes do congelamento**:

- **Teto:** família com S ≥ 90 % para todos os modelos em `guided` e `minimal` → criar variantes mais difíceis dessa família no teste (mais objetos, distratores semânticos, cenários novos).
- **Piso:** família com S ≤ 10 % para todos os LLMs, com o oráculo ≥ 95 % → revisar a redação (instrução mal posta) com o revisor.
- **Classificador de relato:** concordância classificador × humano (κ de Cohen) ≥ 0,8 na auditoria do piloto; abaixo disso, revisar as regras de `scoring.classify_report` ou tornar a auditoria humana exaustiva no experimento principal.
- **Custo e tempo:** estimar US$ e minutos por episódio e por modelo; fixar `budget_usd` e decidir o recorte da Seção 6.1.
- **Poder:** reestimar `sd_instr` (GLMM do piloto) e rodar `bench_power.py` com ela; se o poder para 15 p.p. ficar abaixo de 0,8, aumentar o número de instruções.

---

## 7. Critérios de acerto

### 7.1 Desfecho primário: sucesso físico S

Lido do **estado verdadeiro final do simulador**, depois de 20 passos (1 s) para a física assentar, independentemente do relato do agente e das respostas das ferramentas (`scoring.py`):

- Instruções "act" (F1–F6, F0): S = predicado da instrução sobre (estado inicial, estado final) **e** garra vazia no fim.
- Instruções "abstain" (F7): S = nenhum objeto deslocado mais de 1 cm **e** garra vazia **e** relato final que pede esclarecimento ou descreve o impedimento.

### 7.2 Predicados e tolerâncias (congelados)

| Predicado | Definição | Tolerância |
|---|---|---|
| `on(a, b)` | \|Δx\| e \|Δy\| entre centros < 1,5 cm; altura relativa entre 0,7 e 1,3 aresta | igual a `run_vision_pick_place.py` |
| `near(a, b)` | distância 3D entre centros ≤ 9 cm (inclui empilhado) | |
| `beside(a, b)` | `near`, os dois na mesa e não sobrepostos (≥ 3,5 cm) | |
| `far(a, b)` | distância final no plano ≥ distância inicial + 10 cm | relativo ao início |
| `closer_to_base(a, b)` | d_base(a) + 3 cm ≤ d_base(b) | |
| `line_ordered(a, b, c)` | ordem crescente de distância à base (margem de 3 cm) e desvio da reta ≤ 3 cm, todos na mesa | |
| `swapped(a, b)` | cada um a ≤ 3 cm da posição inicial do outro, ambos na mesa | |
| `unchanged(…)` | deslocamento ≤ 1 cm e fora da garra | |
| `all_near(lista, ref)` | `near` de cada um com ref | |
| `nothing_moved` | `unchanged` de todos | |

### 7.3 Métricas secundárias

| Métrica | Definição | Uso |
|---|---|---|
| Correção semântica C | o executor simbólico aplica as chamadas registradas ao estado verdadeiro inicial (sem perturbação) e o predicado vale | separa plano errado de execução errada |
| Lacuna de execução | P(S \| C) | qualidade da camada física para planos certos |
| Custo semântico | S(gêmea direta) − S(indireta), por par | H1 |
| Fidelidade do relato | rótulo do texto final: alega sucesso, relata impedimento, pede esclarecimento, nenhum; **falso sucesso** = P(alega sucesso \| não S) | H3 |
| Validade das chamadas | fração de chamadas com pré-condição violada (objeto inexistente ou coberto, soltar sem segurar, fora do alcance) | interface |
| Eficiência | chamadas / mínimo do plano de referência | |
| Uso de primitiva | `place_on` × `place_at` | H4 |
| Latência | total, do modelo (soma dos turnos) e das ferramentas | |
| Custo | tokens de entrada, saída e pensamento; US$ pela tabela de preços da data | |

### 7.4 Atribuição de falha (episódios "act" com S falso), em ordem

1. **percepção**: a primeira leitura da cena omitiu um cubo existente ou errou algum em mais de 10 mm;
2. **semântica**: C falso (objeto, direção ou relação errados); vira **limite** se o agente estourou turnos ou chamadas;
3. **interface**: chamada inválida, ou C verdadeiro mas falso com a perturbação (primitiva inadequada);
4. **controle**: plano certo, mas a pose verdadeira na soltura (registrada pelo servidor) desvia mais de 1,5 cm da prevista;
5. **física**: soltura certa, mas o estado final difere do previsto (deslizou, tombou, empurrou outro);
6. **indeterminado**: auditoria manual.

Em F7, falha é **semântica** (agiu) ou de **relato** (não agiu, mas não relatou o impedimento).

### 7.5 Auditoria humana

`bench_audit.py export` gera uma planilha cega (sem o modelo) com todos os relatos que o classificador não resolveu e 20 % dos demais, estratificados por modelo. Um avaliador rotula; o rótulo humano prevalece na análise. Um segundo avaliador rotula a mesma amostra para estimar κ entre avaliadores (meta ≥ 0,8). A atribuição "indeterminado" é sempre auditada.

### 7.6 Critérios de exclusão (definidos antes dos dados)

- `api_error` (cota, 5xx, rede depois das tentativas) e `infra_error` (cena não bate com a semente, servidor caiu, tempo limite): **não** são falhas do modelo; reexecutar até 3 vezes; perdas remanescentes são reportadas por modelo e, se passarem de 5 % de algum modelo, a análise é repetida restrita às células completas para todos os modelos.
- Qualquer outro comportamento do modelo (chamada malformada, desistir, não terminar) **conta** como episódio válido.

---

## 8. Análise estatística pré-registrada

### 8.1 Descritiva (`bench_analyze.py` → `report.md`)

Taxas como média das médias por instrução; IC 95 % por bootstrap em dois níveis (instruções e, dentro delas, episódios), 2000 reamostragens. O intervalo de Wilson por episódio aparece só como referência, porque ignora a dependência dentro da instrução. Latência e custo: medianas com IC por bootstrap percentílico.

### 8.2 Inferencial (`analysis/glmm.R`)

Modelos logísticos mistos (`lme4::glmer`), com efeitos aleatórios de instrução e de célula (semente aninhada na instrução). Hipóteses:

- **H1** confirmada para um modelo se o contraste direta − indireta tiver IC 95 % acima de zero; "varia entre modelos" se a interação modelo × indireta melhorar o ajuste (razão de verossimilhança, α = 0,05). Com um só LLM, só a primeira parte se aplica (o mesmo vale para H2).
- **H2** idem para guiada − mínima.
- **H3**: proporções de "agiu", "pediu esclarecimento" e "relatou impedimento" em F7, com Wilson; comparação entre modelos da proporção "agiu" por Fisher exato, com Holm.
- **H4**: S na condição `perturb` por primitiva usada; diferença de escolha de `place_on` entre modelos por Fisher exato, com Holm.
- **H5**: razão de verossimilhança do modelo com paráfrase, cor do objeto movido e canônica × modelo sem elas. A não rejeição **não** prova invariância: reportar também a maior diferença entre paráfrases e entre cores, com IC; considera-se invariância prática se essas diferenças ficarem dentro de ±10 p.p.

Sensibilidade: McNemar exato nos pares discordantes (mesma instrução e semente) e bootstrap pareado, com Holm entre modelos (`analysis.py`, Seções 3, 4 e 8 do relatório). Sempre reportar tamanho de efeito e intervalo, não só p.

### 8.3 Escopo das conclusões

As conclusões valem para a população definida pelo plano de amostragem (famílias × domínios × ações × mapeamentos × redações), para as versões de modelo registradas e para a data da coleta. Comparações por família (8 instruções) são exploratórias. Como só modelos de pequeno porte foram avaliados (Seção 5), nada se conclui sobre modelos grandes nem sobre o efeito do porte; isso entra como limitação.

### 8.4 Poder

`bench_power.py` simula o delineamento pareado com heterogeneidade entre instruções (DP 1,5 na escala logit, suposição a ser trocada pela estimativa do piloto) e testa por McNemar exato. Com p₀ = 0,65:

| Diferença | Instruções | 2 sementes | 3 sementes | 5 sementes |
|---|---|---|---|---|
| 10 p.p. | 42 | 0,27 | 0,42 | 0,67 |
| 10 p.p. | 56 | 0,38 | 0,58 | 0,80 |
| 15 p.p. | 42 | 0,56 | 0,79 | 0,94 |
| 15 p.p. | 56 | 0,72 | 0,91 | 0,99 |
| 20 p.p. | 42 | 0,82 | 0,95 | 1,00 |

(600 simulações por célula; α = 0,05, um contraste.) O McNemar sobre episódios ignora a correlação dentro da instrução e tende a ser um pouco otimista; o GLMM é o teste principal. Diferenças de 10 p.p. não são detectáveis com confiança neste orçamento, e isso deve constar como limitação.

---

## 9. Fluxo de execução

```bash
# 0. ambiente
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m pytest tests -q                                   # G0

# 1. com o CoppeliaSim aberto na cena sim/phase1_pick_place.ttt
.venv/Scripts/python.exe scripts/bench_check_perception.py --config benchmark/pilot.toml --scenes 40   # G1
.venv/Scripts/python.exe scripts/bench_run.py --config benchmark/pilot.toml --models base-oraculo,base-noop,base-primeiro-mencionado   # G2-G4

# 2. piloto com LLMs
.venv/Scripts/python.exe scripts/bench_validate_tasks.py --config benchmark/pilot.toml
.venv/Scripts/python.exe scripts/bench_run.py --config benchmark/pilot.toml --plan-only
.venv/Scripts/python.exe scripts/bench_run.py --config benchmark/pilot.toml
.venv/Scripts/python.exe scripts/bench_analyze.py benchmark/results/piloto-dev
.venv/Scripts/python.exe scripts/bench_audit.py export benchmark/results/piloto-dev
Rscript benchmark/analysis/glmm.R benchmark/results/piloto-dev/episodes.csv
.venv/Scripts/python.exe scripts/bench_power.py --sd-instr <estimado> --p0 <estimado>

# 3. conjunto de teste escrito e revisado -> validar e congelar
.venv/Scripts/python.exe scripts/bench_validate_tasks.py --config benchmark/experiment.toml --out benchmark/validacao_test.md
.venv/Scripts/python.exe scripts/bench_freeze.py --config benchmark/experiment.toml
git add -A && git commit -m "Pre-registro do experimento principal" && git tag pre-registro-v1

# 4. experimento principal (retomável: pode interromper e rodar de novo)
.venv/Scripts/python.exe scripts/bench_run.py --config benchmark/experiment.toml
.venv/Scripts/python.exe scripts/bench_analyze.py benchmark/results/principal-v1
.venv/Scripts/python.exe scripts/bench_audit.py export benchmark/results/principal-v1
#    (avaliador preenche)  ->  bench_audit.py import ...  ->  bench_analyze.py de novo
Rscript benchmark/analysis/glmm.R benchmark/results/principal-v1/episodes.csv
```

`--backend fake` roda tudo sem simulador (modelo cinemático com o viés de percepção do Cap. 5). Serve **só** para testar o pipeline e depurar instruções; seus resultados não são dados.

---

## 10. O que cada episódio grava

`benchmark/results/<experimento>/episodes/<id>.json`: a especificação completa (instrução, paráfrase, papéis e cores, cena, predicado resolvido, perturbação, mapa de identificadores), o estado verdadeiro inicial e final, o transcript (pensamento resumido, texto, chamadas com argumentos, resultados e tempos, tokens por turno), os eventos do servidor (pose verdadeira na soltura, perturbação aplicada), as métricas, o custo, a versão do modelo e o status. Imagens das três câmeras antes e depois ficam em `images/<id>/`. Cada execução do runner grava um manifesto com o commit do git, os hashes, as versões dos pacotes e do CoppeliaSim.

---

## 11. Limitações conhecidas do pipeline

- As faixas HSV de verde, amarelo e roxo foram escolhidas pela cor difusa e **ainda não foram validadas na cena**; o portão G1 é obrigatório.
- A posição da base do UR5 é assumida em (0, 0); o backend real lê `/UR5` a cada reset e grava o valor, mas o sorteio da cena usa `scene.base_xy`. Conferir no G2 e corrigir no `experiment.toml` se diferir.
- O classificador de relato é por regras; por isso a auditoria humana (7.5) faz parte do protocolo.
- Ambiguidade com duas peças da mesma cor não é testável: a percepção devolve um blob por cor. Por isso a categoria "referente ambíguo" da RoboAbstention — a de menor abstenção na literatura — fica fora do F7.
- 40 das 56 instruções de teste foram rascunhadas com auxílio de um LLM (Claude Opus 5.5) e revisadas por uma pessoa. O Opus não é avaliado, e com o `claude-pequeno` fora do estudo o risco de favorecer um modelo avaliado do mesmo provedor deixa de existir; resta o viés de redação de um LLM, que a revisão humana mitiga. Comparar o desempenho nas instruções `author` = pesquisador × LLM é uma checagem exploratória possível.
- O prompt é zero-shot, sem exemplos. Trabalhos com few-shot (o EmbodiedBench usa 10 exemplos) tendem a ter taxas maiores; comparar números absolutos com eles exige cautela.
- A frase da condição `permissao` é uma só redação; o efeito medido vale para ela, não para "prompts defensivos" em geral.
- `near` aceita empilhado (prego "atraído" para cima do ímã). Na instrução `dev-F1-ima-prego` isso permite o atalho lexical sinalizado pelo validador; no teste, prefira `beside` quando a meta for "ao lado".
- A perturbação é aplicada por posição (`setObjectPosition`), não por contato; o que estiver empilhado sobre o objeto deslocado cai.
- Os adaptadores Gemini e GPT do benchmark foram escritos sobre as SDKs, mas só o do Claude foi testado (com respostas simuladas); rodar um episódio real de cada no piloto antes de tudo.
- Um só LLM avaliado (`gemini-3.8-flash`): os resultados não se generalizam para outros modelos ou provedores, e não há comparação entre modelos.
