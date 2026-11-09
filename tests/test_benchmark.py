"""Testes do pipeline do benchmark que NAO precisam de simulador nem de API.

  .venv/Scripts/python.exe -m pytest tests -q
"""
import asyncio
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from benchmark import analysis, predicates, symbolic  # noqa: E402
from benchmark.agents.scripted import FirstMentionedAgent, NoopAgent, OracleAgent  # noqa: E402
from benchmark.backends.fake import FakeBackend  # noqa: E402
from benchmark.config import load_config  # noqa: E402
from benchmark.schedule import build_plan  # noqa: E402
from benchmark.scoring import classify_report, score_episode  # noqa: E402
from benchmark.tasks import instantiate, load_tasks  # noqa: E402
from benchmark.tool_specs import SYSTEM_PROMPTS, tool_description  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TASKS = load_tasks(ROOT / "benchmark" / "tasks" / "instructions.json")
BY_ID = {t["id"]: t for t in TASKS}


def st(objects, held=None):
    return {"objects": objects, "held": held, "base_xy": [0.0, 0.0], "cube": 0.04}


# ---- predicados ----------------------------------------------------------------
def test_on_and_tolerances():
    init = st({"red": [0.4, 0.1, 0.02], "blue": [0.3, 0.3, 0.02]})
    ok = st({"red": [0.4, 0.1, 0.02], "blue": [0.405, 0.11, 0.06]})
    off = st({"red": [0.4, 0.1, 0.02], "blue": [0.42, 0.1, 0.06]})
    assert predicates.evaluate({"on": ["blue", "red"]}, init, ok)
    assert not predicates.evaluate({"on": ["blue", "red"]}, init, off)  # 2 cm > 1,5 cm
    assert not predicates.evaluate({"on": ["red", "blue"]}, init, ok)  # direcao importa


def test_far_is_relative_to_initial_and_unchanged():
    init = st({"a": [0.3, 0.2, 0.02], "b": [0.4, 0.2, 0.02]})
    fin = st({"a": [0.3, 0.2, 0.02], "b": [0.48, 0.2, 0.02]})
    assert not predicates.evaluate({"far": ["b", "a"]}, init, fin)  # ganhou 8 cm < 10 cm
    fin2 = st({"a": [0.3, 0.2, 0.02], "b": [0.55, 0.3, 0.02]})
    assert predicates.evaluate({"far": ["b", "a"]}, init, fin2)
    assert predicates.evaluate({"unchanged": ["a"]}, init, fin2)
    assert not predicates.evaluate({"unchanged": ["b"]}, init, fin2)


def test_line_ordered_and_swapped():
    init = st({"a": [0.3, 0.3, 0.02], "b": [0.4, 0.1, 0.02], "c": [0.2, 0.4, 0.02]})
    line = st({"a": [0.25, 0.0, 0.02], "b": [0.35, 0.0, 0.02], "c": [0.45, 0.01, 0.02]})
    assert predicates.evaluate({"line_ordered": ["a", "b", "c"]}, init, line)
    assert not predicates.evaluate({"line_ordered": ["c", "b", "a"]}, init, line)
    sw = st({"a": [0.4, 0.1, 0.02], "b": [0.3, 0.3, 0.02], "c": [0.2, 0.4, 0.02]})
    assert predicates.evaluate({"swapped": ["a", "b"]}, init, sw)


def test_resolve_roles_and_others():
    spec = {"and": [{"on": ["A", "B"]}, {"unchanged": ["@others"]}]}
    out = predicates.resolve(spec, {"A": "red", "B": "blue"}, ["green", "yellow"])
    assert out == {"and": [{"on": ["red", "blue"]}, {"unchanged": ["green", "yellow"]}]}


# ---- instanciacao -------------------------------------------------------------------
def test_instantiation_is_deterministic_and_paired():
    t = BY_ID["test-F1-lava-agua"]
    a = instantiate(t, "indirect", 1)
    b = instantiate(t, "indirect", 1)
    d = instantiate(t, "direct", 1)
    assert a == b
    assert a["scene"] == d["scene"] and a["roles"] == d["roles"]  # gemeas: mesma cena e papeis
    p = instantiate(t, "indirect", 1, perturb=True)
    assert p["scene"] == a["scene"] and p["perturbation"] is not None


def test_initial_scene_never_satisfies_goal():
    for t in TASKS:
        if t["expected"] != "act":
            continue
        for s in range(3):
            sp = instantiate(t, "indirect", s)
            state = st(sp["scene"])
            assert not predicates.evaluate(sp["success"], state, state), (t["id"], s)


def test_first_role_color_rotates_across_seeds():
    t = BY_ID["test-F1-vulcao-aldeia"]
    first = [instantiate(t, "indirect", s)["roles"]["A"] for s in range(5)]
    assert len(set(first)) == 5  # quadrado latino sobre as 5 cores


def test_geometric_roles():
    t = BY_ID["test-F3-longe-perto"]
    for s in range(3):
        sp = instantiate(t, "indirect", s)
        dist = {c: (p[0] ** 2 + p[1] ** 2) ** 0.5 for c, p in sp["scene"].items()}
        assert sp["roles"]["A"] == max(dist, key=dist.get)
        assert sp["roles"]["B"] == min(dist, key=dist.get)


def test_no_tool_names_in_prompts():
    for t in TASKS:
        for v in ("indirect", "direct") if t.get("direct") else ("indirect",):
            prompt = instantiate(t, v, 0)["prompt"]
            assert "place_" not in prompt and "pick_" not in prompt and "{" not in prompt


# ---- executor simbolico e metricas --------------------------------------------------
def test_symbolic_stack_and_perturbation():
    init = st({"red": [0.4, 0.1, 0.02], "blue": [0.3, 0.3, 0.02]})
    calls_on = [{"name": "pick_object", "args": {"object_name": "blue"}},
                {"name": "place_on", "args": {"object_name": "red"}}]
    calls_at = [{"name": "pick_object", "args": {"object_name": "blue"}},
                {"name": "place_at", "args": {"x": 0.4, "y": 0.1, "z": 0.06}}]
    pert = {"object": "red", "delta": [-0.12, 0.1], "call_index": 0}
    goal = {"on": ["blue", "red"]}
    for calls in (calls_on, calls_at):
        fin, _ = symbolic.execute(init, calls)
        assert predicates.evaluate(goal, init, fin)
    fin_on, _ = symbolic.execute(init, calls_on, pert)
    fin_at, _ = symbolic.execute(init, calls_at, pert)
    assert predicates.evaluate(goal, init, fin_on)  # place_on rele o alvo
    assert not predicates.evaluate(goal, init, fin_at)  # place_at fica congelado


def test_symbolic_flags_invalid_calls():
    init = st({"red": [0.4, 0.1, 0.02]})
    _, info = symbolic.execute(init, [{"name": "place_on", "args": {"object_name": "red"}},
                                      {"name": "pick_object", "args": {"object_name": "purple"}}])
    reasons = {i["reason"] for i in info["invalid_calls"]}
    assert reasons == {"place_on_not_holding", "pick_nonexistent_or_covered"}


@pytest.mark.parametrize("text,label", [
    ("Coloquei o cubo azul sobre o vermelho para apagar o fogo.", "claims_success"),
    ("Não há cubo roxo na mesa, então não consegui realizar a tarefa.", "reports_failure"),
    ("Qual cubo você considera o mais pesado? Todos parecem idênticos.", "asks_clarification"),
    ("Os cubos são idênticos; não dá para saber qual é o mais pesado.", "reports_failure"),
    (None, "none"),
])
def test_report_classifier(text, label):
    assert classify_report(text) == label


def _episode(agent, t, variant="indirect", seed=0, cond=None, perturb=False):
    cond = cond or {"scaffolding": "guided"}
    sp = instantiate(t, variant, seed, perturb=perturb)
    be = FakeBackend()
    init = be.reset(sp)

    async def go():
        async with be.tools(sp, cond) as tc:
            return await agent.run(sp["prompt"], tc, SYSTEM_PROMPTS["guided"], sp)

    tr = asyncio.run(go())
    return score_episode(sp, init, be.finalize(sp), tr, be.events(), cond), tr


def test_oracle_solves_every_dev_instruction():
    for t in TASKS:
        for v in ("indirect", "direct") if t.get("direct") else ("indirect",):
            for s in range(2):
                m, _ = _episode(OracleAgent(), t, v, s)
                assert m["S"], (t["id"], v, s, m["attribution"])


def test_noop_fails_act_and_passes_abstain():
    for t in TASKS:
        m, _ = _episode(NoopAgent(), t)
        assert m["S"] == (t["expected"] == "abstain"), t["id"]


def test_attribution_semantic_for_wrong_direction():
    m, _ = _episode(FirstMentionedAgent(), BY_ID["test-F1-lava-agua"])  # empilha lava sobre agua
    assert not m["S"] and not m["C"] and m["attribution"] == "semantica"


def test_neutral_ids_round_trip():
    t = BY_ID["test-F1-lava-agua"]
    cond = {"scaffolding": "guided", "neutral_ids": True}
    m, tr = _episode(OracleAgent(), t, cond=cond)
    assert m["S"]
    first = json.loads(tr["turns"][0]["tool_results"][0]["result"])
    assert all(k.startswith("obj_") and "color" in v for k, v in first["objects"].items())
    assert "obj_1" in tool_description("pick_object", "guided", neutral_ids=True)


# ---- plano e analise -------------------------------------------------------------------
def test_plan_is_paired_across_models():
    cfg = load_config(ROOT / "benchmark" / "pilot.toml")
    plan = build_plan(cfg, TASKS)
    cells = {}
    for ep in plan:
        key = (ep["task_id"], ep["variant"], ep["seed_idx"], ep["condition"]["name"], ep["rep"])
        cells.setdefault(key, set()).add(ep["model"]["label"])
    labels = {m["label"] for m in cfg["models"]} | {b["label"] for b in cfg["baselines"]}
    assert all(v == labels for k, v in cells.items() if k[4] == 0)
    assert len({ep["episode_id"] for ep in plan}) == len(plan)


def test_stats_helpers():
    lo, hi = analysis.wilson(3, 4)
    assert abs(lo - 0.3006) < 1e-3 and abs(hi - 0.9544) < 1e-3  # valores do Cap. 5
    pairs = [({"S": 1}, {"S": 0})] * 8 + [({"S": 0}, {"S": 1})] * 1
    assert abs(analysis.mcnemar_exact(pairs)["p"] - 0.0390625) < 1e-9
    adj = analysis.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj == {"a": 0.03, "c": 0.06, "b": 0.06}


# ---- adaptador LLM com respostas simuladas (sem API) --------------------------------
class _B:
    def __init__(self, type, **kw):
        self.type = type
        self.__dict__.update(kw)


class _R:
    def __init__(self, content, stop):
        self.content, self.stop_reason, self.model = content, stop, "fake-claude-0"
        self.usage = _B("usage", input_tokens=100, output_tokens=20, cache_read_input_tokens=0)


def test_claude_adapter_loop_against_fake_backend(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-teste")
    from benchmark.agents.claude import ClaudeAgent

    t = BY_ID["test-F1-lava-agua"]
    sp = instantiate(t, "indirect", 0)
    water, lava = sp["roles"]["B"], sp["roles"]["A"]
    script = [
        _R([_B("thinking", thinking="ver a cena"), _B("tool_use", id="1", name="get_scene_objects", input={})],
           "tool_use"),
        _R([_B("tool_use", id="2", name="pick_object", input={"object_name": water})], "tool_use"),
        _R([_B("tool_use", id="3", name="place_on", input={"object_name": lava})], "tool_use"),
        _R([_B("text", text="Coloquei a agua sobre a lava e apaguei o fogo.")], "end_turn"),
    ]
    seen = []

    async def create(**kw):
        seen.append(kw)
        return script[len(seen) - 1]

    agent = ClaudeAgent("fake-claude")
    agent.client = SimpleNamespace(messages=SimpleNamespace(create=create))
    be = FakeBackend()
    init = be.reset(sp)

    async def go():
        async with be.tools(sp, {"scaffolding": "minimal"}) as tc:
            return await agent.run(sp["prompt"], tc, SYSTEM_PROMPTS["minimal"], sp)

    tr = asyncio.run(go())
    m = score_episode(sp, init, be.finalize(sp), tr, be.events(), {"scaffolding": "minimal"})
    assert m["S"] and m["C"] and m["report"] == "claims_success"
    assert tr["usage_total"]["input_tokens"] == 400 and tr["model_version"] == "fake-claude-0"
    assert seen[0]["system"] == SYSTEM_PROMPTS["minimal"]
    assert "re-detectada" not in seen[0]["tools"][1]["description"]  # descricao minima, sem conselhos


# ---- eixos de parafrase, categorias de abstencao e condicao "permissao" -------------
def test_paraphrase_axis_and_abstain_type_in_spec():
    sp = [instantiate(BY_ID["test-F1-lava-agua"], "indirect", s) for s in range(3)]
    assert sorted(s["paraphrase_axis"] for s in sp) == ["base", "contexto", "lexical"]
    assert instantiate(BY_ID["test-F1-lava-agua"], "direct", 0)["paraphrase_axis"] is None
    assert instantiate(BY_ID["test-F7-premissa-falsa"], "indirect", 0)["abstain_type"] == "premissa_falsa"


def test_schema_rejects_bad_abstain_type_and_axes():
    from benchmark.tasks import check_schema
    t = dict(BY_ID["test-F7-dentro"], abstain_type="inventada")
    with pytest.raises(ValueError):
        check_schema(t)
    t = dict(BY_ID["test-F7-dentro"], paraphrase_axes=["base", "lexical"])
    with pytest.raises(ValueError):
        check_schema(t)


def test_system_prompt_abstain_hint():
    from benchmark.tool_specs import ABSTAIN_HINT, system_prompt
    assert system_prompt({"scaffolding": "minimal"}) == SYSTEM_PROMPTS["minimal"]
    assert system_prompt({"scaffolding": "guided", "abstain_hint": True}).endswith(ABSTAIN_HINT)


def test_permission_condition_includes_all_f7():
    cfg = load_config(ROOT / "benchmark" / "pilot.toml")
    plan = [ep for ep in build_plan(cfg, TASKS) if ep["condition"]["name"] == "permissao"]
    f7 = {t["id"] for t in TASKS if t["family"] == "F7" and t["split"] == "dev"}
    assert f7 <= {ep["task_id"] for ep in plan}
    assert any(ep["family"] != "F7" for ep in plan)  # tambem mede abstencao indevida
