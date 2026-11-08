# Instruções e prompts: comparação com a literatura

Levantamento feito em 2026-10-08 para orientar a redação do conjunto de teste (56 instruções, PROTOCOLO.md, Seção 3). Compara três coisas com trabalhos que avaliam LLMs/VLMs em manipulação guiada por linguagem: **(1)** as famílias de instrução, **(2)** o prompt de sistema e a interface com o modelo, **(3)** escala, geração e métricas. A Seção 4 traz as recomendações para o conjunto de teste.

Exemplos entre aspas são citações dos artigos; os nossos estão no formato de `tasks/instructions.json`.

---

## 1. Famílias de instrução

### 1.1 Mapeamento F1–F7 × literatura

| Nossa família | O que exige | Equivalente mais próximo | Exemplo da literatura | Situação |
|---|---|---|---|---|
| **F1** metáfora / papéis atribuídos ("O cubo {A} representa lava e o {B} água. Apague o fogo.") | mapear um papel fictício para uma relação física | RT-2 "raciocínio semântico" (objeto como "martelo improvisado"); VLABench, intenção implícita | VLABench: "I want a toy in the Disney series" (alvo implícito) | **Sem equivalente direto.** Nenhum benchmark encontrado atribui papéis simbólicos a objetos neutros. É a contribuição original, mas precisa de justificativa própria. |
| **F2** conhecimento de mundo → atributo ("o cubo que lembra o céu") | ligar conceito a cor | RT-2 *symbol understanding*; EmbodiedBench *Common Sense* e *Visual Appearance* | EB-ALFRED: "Place washed leafy green vegetable in a receptacle that can keep it fresh for several days" | Bem coberto pela literatura. |
| **F3** relação espacial relativa à base | raciocínio métrico sobre a cena | EmbodiedBench *Spatial Awareness*; LoHoRavens *space*; KnowNo, ambiguidade espacial | RT-2: pegar "o menor/maior objeto" | Bem coberto. |
| **F4** restrição ("sem mexer nos outros", "{C} é de vidro, não pode ser tocado") | cumprir a meta respeitando uma proibição | RoboFollow, *trajectory constraints*; SafeAgentBench (riscos) | RoboAbstention usa "Give me the red block without touching it" como **contraditória** | Pouco explorado como tarefa *viável*. Bom diferencial. |
| **F5** esquema com várias etapas ("Construa a casa", "troca da guarda") | decompor em subtarefas | SayCan *Long-Horizon*; LoHoRavens (≥ 5 pick-and-place); VLABench, tarefas compostas | SayCan: "I spilled my coke on the table, throw it away…" | **Horizonte curto.** As nossas têm 2 ou 3 movimentos; a literatura chama de longo horizonte a partir de 5. |
| **F6** seleção e quantificação ("todos exceto {A}", "o mais perto dele") | referência por condição | LoHoRavens *arithmetics/reference*; KnowNo, ambiguidade numérica | KnowNo: "put some blocks in the green bowl" | Coberto; as nossas são determinadas (sem ambiguidade). |
| **F7** inviável ou ambígua | abster-se ou pedir esclarecimento | KnowNo; AmbiK; RoboAbstention; RAMA; SafeAgentBench | ver 1.2 | Coberto, mas com **menos subtipos** que a literatura. |
| **F0** gêmea direta | mesma meta, linguagem explícita | AmbiK (1000 pares ambíguo × não ambíguo); SayCan *Structured Language*; EmbodiedBench *Base* | SayCan: "Move the rice chips to the far counter" | O pareamento exato (mesma cena e mesmo predicado) é **mais rigoroso** que na literatura. |

### 1.2 F7 em detalhe: taxonomia de abstenção

A RoboAbstention (2026) organiza a abstenção em 8 categorias. Comparação com o nosso F7:

| Categoria (RoboAbstention) | Exemplo deles | Nosso F7 hoje | Viável na nossa cena? |
|---|---|---|---|
| Referente ausente | "Give me the rubber duck" | `dev-F7-cor-ausente` | sim |
| Referente ambíguo | dois blocos indistinguíveis | — | **não**: a percepção devolve um blob por cor (limitação já registrada) |
| Intenção subespecificada | "Move that over there" | — | sim |
| Intenção subjetiva | "Hand me my favorite drink" | — | sim ("o cubo da minha cor favorita") |
| Inviabilidade física | "Put the keyboard inside the cup" | — | sim ("coloque o {A} dentro do {B}") |
| Capacidade ausente | "Does the orange smell bad?" | `dev-F7-mais-pesado` (peso não é observável) | sim (peso, temperatura, material) |
| Contraditória | "Give me the red block without touching it" | `dev-F7-contraditoria` | sim |
| Premissa falsa | "Turn off the monitor" (já desligado) | — | sim ("tire o {A} de cima do {B}" quando não está em cima) |

Resultado da RoboAbstention que importa para nós: com o prompt padrão, sem orientação sobre abstenção, os modelos se abstêm pouco (Gemini 2.5 Flash 39 %, Gemini Robotics-ER 1.6 16,5 %). A categoria "referente ambíguo" foi a pior, com 2,6 %. Com prompt defensivo e exemplos, a abstenção sobe para cerca de 90 %. **O resultado de H3 depende fortemente do prompt** (ver 2.2).

---

## 2. Prompt de sistema e interface

### 2.1 Como a literatura formula o prompt

| Trabalho | Papel atribuído | Interface de ação | Exemplos no prompt | Saída |
|---|---|---|---|---|
| SayCan (2022) | diálogo "Robot: Hi there, I'm a robot operating in an office kitchen…" | o LLM pontua habilidades pré-definidas; a afordância filtra | few-shot de diálogos | a próxima habilidade, escolhida por *scoring* |
| Code as Policies (2022) | nenhum (completar código) | API Python de percepção e controle | comandos como comentários + código | programa |
| KnowNo (2023) | planejador | múltipla escolha (A–D + "E: opção não listada") | few-shot | letra da opção; pede ajuda se o conjunto conforme tiver mais de 1 opção |
| EmbodiedBench, EB-Manipulation (2025) | "You are a Franka Panda robot with a parallel gripper…" | ação discreta 7D [X, Y, Z, Roll, Pitch, Yaw, Gripper] | **10 exemplos** por padrão | JSON: descrição visual → reflexão → raciocínio → plano → ações |
| VLABench (2025) | agente com biblioteca de habilidades | DSL: `Pick("Apple", {...})`, `Place("Basket", {...})` | few-shot | sequência em DSL |
| RoboAbstention (2026) | robô com visão e manipulação | resposta livre: "respond with what you would do to complete the task" | nenhum (padrão) | texto, classificado por LLM juiz |
| **Nosso** | "Você controla um braço robótico (UR5 com garra) numa simulação…" | **chamada de função nativa** via MCP (4 ferramentas) | **nenhum** (zero-shot) | chamadas de ferramenta + frase final de relato |

### 2.2 Onde o nosso difere, e se isso é problema

1. **Zero-shot e sem exemplos.** A maioria usa few-shot (o EmbodiedBench usa 10). Em protocolo comparativo isso é defensável, porque exemplos vazam procedimento e enviesam a escolha de primitivas (H4). Convém declarar no Cap. 6 que isso tende a **subestimar** o desempenho em relação a trabalhos com few-shot.
2. **Ferramentas nativas em vez de DSL, código ou múltipla escolha.** É a interface que os modelos atuais usam em agentes, e por isso tem mais validade ecológica que os formatos de 2022–2023. Os números absolutos não são comparáveis aos de SayCan ou Code as Policies.
3. **Sem orientação sobre abstenção.** O `guided` pede que o modelo "pense em voz alta" e confira resultados, mas não diz que ele pode recusar ou pedir esclarecimento. Isso coincide com o prompt padrão da RoboAbstention, que é a condição mais conservadora. Tem três consequências:
   - H3 mede a abstenção **espontânea**, e isso precisa estar escrito assim;
   - a literatura mostra efeitos de até cerca de 75 p.p. com prompt defensivo, então uma condição exploratória com uma frase de permissão (por exemplo, "se a tarefa for impossível ou ambígua, explique e não aja") isolaria esse efeito. Ela só pode entrar **antes** do congelamento;
   - o `minimal` também não menciona abstenção, então H2 × F7 não confunde as duas coisas.
4. **Dois níveis de scaffolding (H2).** Não encontrei benchmark que manipule o prompt de forma controlada como variável experimental. A literatura fixa um prompt. Isso é diferencial.
5. **Percepção por ferramenta, não por imagem.** EmbodiedBench e VLABench passam a imagem ao modelo; nós passamos coordenadas detectadas. Isso isola o raciocínio da visão, que é o objetivo, mas o estudo deixa de responder sobre percepção visual do LLM. Vale citar como recorte.

---

## 3. Escala, geração e métricas

### 3.1 Quantas instruções a literatura usa

| Trabalho | Unidade | Tamanho |
|---|---|---|
| SayCan | instruções | 101 (≈ 15 por família, 7 famílias) |
| EmbodiedBench, EB-Manipulation | tarefas | 228 (36–48 por subconjunto, 5 subconjuntos) |
| KnowNo | tentativas em simulação | 200 por método e cenário |
| RT-2 | tentativas de avaliação | ≈ 6.000 (no total, todas as categorias) |
| VLABench | categorias de tarefa | 100 (60 primitivas + 40 compostas), 10 redações por instância |
| AmbiK | pares de tarefas | 1.000 (só texto, sem execução) |
| RoboAbstention | instruções | 6.069 (só resposta, sem execução) |
| **Nosso (teste)** | instruções | **56 indiretas + ~48 gêmeas ≈ 104 prompts**, 8 por família, × 3 sementes |

Em número de instruções **executadas fisicamente**, ficamos na faixa do SayCan e abaixo do EB-Manipulation. Por família (8), estamos abaixo da literatura (15–48). Isso confirma a decisão do protocolo de tratar comparações por família como exploratórias. Os benchmarks com milhares de itens só avaliam texto, sem executar.

### 3.2 Como as instruções são geradas

| Prática | Quem usa | Nós |
|---|---|---|
| Gerar com LLM a partir de templates + validação humana | VLABench (GPT-4, 10 redações por instância), EmbodiedBench (GPT-4o), AmbiK (LLM + validação humana) | redação humana com revisão independente |
| Paráfrases controladas por eixo (verbo × referência ao objeto) | LIBERO-Para (43 tipos; quedas de 22–52 p.p., sobretudo por variação na referência ao objeto) | 3 paráfrases em "registros diferentes", sem eixo rotulado |
| Variação de novidade em níveis | VIMA-Bench (L1 posição → L4 tarefa nova) | `canonical` × não canônica |
| Contexto irrelevante que esconde a instrução | EmbodiedBench *Complex Instruction* | — |
| Cena que obriga a usar a linguagem | RoboFollow (alta "entropia de cena") | regra 9 (sem atalho lexical) e baseline "primeiro mencionado" |

**Cuidado se usarmos LLM para rascunhar instruções:** se o rascunho vier do Claude ou do Gemini, o conjunto pode favorecer o modelo que escreveu. Se for usado, o rascunho deve vir de um modelo que **não** é avaliado, e isso deve ser registrado no campo `author`.

### 3.3 Métricas

| Nossa métrica | Equivalente na literatura |
|---|---|
| S (sucesso físico) × C (correção semântica) | SayCan: *planning rate* 70 % × *execution rate* 61 % |
| Atribuição de falha (percepção, semântica, interface, controle, física) | Embodied Agent Interface: alucinação, afordância, erros de planejamento; LIBERO-Para: 80–96 % das falhas são de planejamento |
| Classificador de relato por regras + auditoria humana (κ) | RoboAbstention: LLM juiz validado com humanos (97,5 %, κ de Fleiss 0,871) |
| GLMM com efeitos de instrução, bootstrap e poder | quase ausente na literatura, que reporta taxa média sem intervalo |

O rigor estatístico (pareamento, efeitos aleatórios de instrução, análise de poder, pré-registro) é o ponto em que o protocolo **supera** os benchmarks levantados. Vale destacar no Cap. 6.

---

## 4. Recomendações para o conjunto de teste

Em ordem de impacto. Tudo antes do congelamento.

**Situação (2026-10-08):** as recomendações 1 a 7 entraram no protocolo (Seções 1, 3.1, 3.2 regras 5, 6, 11 e 12, 6.1 e 11), no validador e no conjunto de desenvolvimento (versão dev-0.2: eixos de paráfrase em todas as instruções, 4 novas F7, uma F5 com 4 movimentos e a condição exploratória `permissao`). Falta escrever o conjunto de teste seguindo essas regras.

1. **Ancorar cada família na literatura** (tabela 1.1) no texto do Cap. 6. F1 e F4 viáveis ficam como contribuição original; F2, F3, F5, F6 e F7 citam os equivalentes.
2. **Diversificar o F7.** Distribuir as 8 instruções por ao menos 5 categorias da RoboAbstention viáveis na cena: referente ausente, capacidade ausente, contraditória, premissa falsa, inviabilidade física, subespecificada e subjetiva. Registrar a categoria num campo novo (por exemplo, `abstain_type`) para analisar H3 por categoria.
3. **Rotular as paráfrases por eixo**: verbo, referência ao objeto e registro/contexto, como no LIBERO-Para. Isso torna H5 interpretável ("cai quando muda a referência ao objeto?") sem aumentar o número de episódios, porque a paráfrase gira com a semente.
4. **Uma paráfrase com contexto irrelevante** por instrução, no estilo *Complex Instruction* do EmbodiedBench. Ela entra como um dos três registros, e não como condição nova.
5. **Esticar parte do F5.** Com até 5 cubos, incluir 2 ou 3 instruções com 4 movimentos (por exemplo, empilhar 4 ou ordenar 4), para chegar perto do que a literatura chama de longo horizonte.
6. **Decidir sobre a frase de permissão para abster-se.** Ou manter só a abstenção espontânea, declarada como tal, ou acrescentar uma condição exploratória pequena só com F7 (8 instruções × 3 sementes = 24 episódios por modelo, menos de US$ 0,50 no total).
7. **Se usar LLM para rascunhar**, usar um modelo fora do estudo e manter a revisão humana independente (regra 6 do protocolo).

---

## Fontes

- Ahn et al., *Do As I Can, Not As I Say* (SayCan), 2022. https://arxiv.org/abs/2204.01691
- Liang et al., *Code as Policies*, 2022. https://arxiv.org/abs/2209.07753
- Brohan et al., *RT-2: Vision-Language-Action Models*, 2023. https://arxiv.org/abs/2307.15818
- Ren et al., *Robots That Ask For Help* (KnowNo), CoRL 2023. https://arxiv.org/abs/2307.01928
- Jiang et al., *VIMA: General Robot Manipulation with Multimodal Prompts*, ICML 2023. https://arxiv.org/abs/2210.03094
- Zhang et al., *LoHoRavens*, 2023. https://arxiv.org/abs/2310.12020
- Li et al., *Embodied Agent Interface*, NeurIPS 2024. https://arxiv.org/abs/2410.07166
- Yin et al., *SafeAgentBench*, 2024. https://arxiv.org/abs/2412.13178
- Zhang et al., *VLABench*, ICCV 2025. https://arxiv.org/abs/2412.18194
- Yang et al., *EmbodiedBench*, ICML 2025. https://arxiv.org/abs/2502.09560
- Ivanova et al., *AmbiK: Dataset of Ambiguous Tasks in Kitchen Environment*, ACL 2025. https://arxiv.org/abs/2506.04089
- Song et al., *RationalVLA* / benchmark RAMA, 2025. https://arxiv.org/abs/2506.10826
- Fei et al., *LIBERO-Plus*, CVPR 2026. https://openaccess.thecvf.com/content/CVPR2026/html/Fei_LIBERO-Plus_A_Progressive_Robustness_Benchmark_for_Visual-Language-Action_Models_CVPR_2026_paper.html
- *LIBERO-Para: A Diagnostic Benchmark and Metrics for Paraphrase Robustness in VLA Models*, 2026. https://arxiv.org/abs/2603.28301
- *The Yes-Man Syndrome: Benchmarking Abstention in Embodied Robotic Agents* (RoboAbstention), 2026. https://arxiv.org/abs/2605.20544
- *RoboFollow: Unveiling the Instruction Following Mirage in Embodied Agents*, 2026. https://arxiv.org/abs/2609.25636

Limitação deste levantamento: algumas citações (RoboAbstention, SafeAgentBench, RAMA, LIBERO-Para) vêm de resumos e páginas de apresentação, não da leitura integral dos artigos; conferir os números no PDF antes de citar na tese. Os primeiros autores de LIBERO-Para, RoboAbstention e RoboFollow não foram confirmados.
