"""Execucao do plano: um episodio por vez, retomavel.

Cada episodio grava benchmark/results/<experimento>/episodes/<id>.json com a
especificacao, a verdade-terreno inicial e final, o transcript, os eventos
do servidor, as metricas e o status. Ao reiniciar, episodios com status
"done" sao pulados e os demais (api_error, infra_error) sao tentados de novo,
ate `max_attempts`. Criterios de exclusao (definidos antes dos dados):
  - api_error: a API falhou depois das tentativas (cota, 5xx, rede) -> nao e
    falha do modelo; reexecutar; se persistir, reportar como perda.
  - infra_error: o harness ou o simulador falhou (cena nao bateu com a
    semente, servidor caiu, tempo limite) -> idem.
  - qualquer outra coisa que o modelo faca (chamada malformada, nao terminar
    em max_turns, desistir) CONTA como episodio valido.
"""
import asyncio
import json
import platform
import traceback
from datetime import datetime
from pathlib import Path

from . import scoring
from .agents import make_agent
from .agents.base import ProviderError
from .config import freeze_record, git_commit, workspace
from .geometry import dist3
from .tasks import instantiate
from .tool_specs import system_prompt as make_system_prompt


def episode_cost(tokens: dict, price: dict | None) -> float | None:
    """Precos em USD por milhao de tokens: {input, output, cache_read?,
    thinking_separate?}. thinking_separate=true quando o provedor conta os
    tokens de pensamento fora de output_tokens (Gemini)."""
    if not price or not tokens:
        return None
    cost = tokens["input_tokens"] * price["input"] + tokens["output_tokens"] * price["output"]
    cost += tokens.get("cache_read_tokens", 0) * price.get("cache_read", 0)
    if price.get("thinking_separate"):
        cost += tokens.get("thinking_tokens", 0) * price["output"]
    return round(cost / 1e6, 6)


class Runner:
    def __init__(self, cfg, tasks, backend, out_dir, save_images=True, log=print):
        self.cfg = cfg
        self.exp = cfg["experiment"]
        self.tasks = {t["id"]: t for t in tasks}
        self.task_list = tasks
        self.backend = backend
        self.out_dir = Path(out_dir)
        self.ep_dir = self.out_dir / "episodes"
        self.ep_dir.mkdir(parents=True, exist_ok=True)
        self.save_images = save_images
        self.log = log
        self._agents = {}

    def write_manifest(self, plan, unfrozen: bool, extra=None):
        import importlib.metadata as md
        versions = {}
        for pkg in ("anthropic", "google-genai", "openai", "mcp", "numpy", "opencv-python"):
            try:
                versions[pkg] = md.version(pkg)
            except md.PackageNotFoundError:
                versions[pkg] = None
        manifest = {
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "config_path": self.cfg["_path"], "config_sha256": self.cfg["_sha256"],
            "freeze": freeze_record(self.cfg, self.task_list), "unfrozen": unfrozen,
            "git_commit": git_commit(), "python": platform.python_version(), "platform": platform.platform(),
            "packages": versions, "backend": self.backend.describe(), "n_planned": len(plan), **(extra or {}),
        }
        path = self.out_dir / f"manifest_{datetime.now():%Y%m%d_%H%M%S}.json"
        path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def _agent(self, m):
        key = m["label"]
        if key not in self._agents:
            opts = {k: v for k, v in m.items() if k not in ("provider", "model", "label", "tier", "conditions",
                                                             "notes")}
            self._agents[key] = make_agent(m["provider"], m["model"], max_turns=self.exp["max_turns"],
                                           max_tool_calls=self.exp["max_tool_calls"], **opts)
        return self._agents[key]

    def existing(self, ep_id):
        p = self.ep_dir / f"{ep_id}.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    def spent_usd(self) -> float:
        total = 0.0
        for p in self.ep_dir.glob("*.json"):
            rec = json.loads(p.read_text(encoding="utf-8"))
            total += rec.get("cost_usd") or 0.0
        return total

    def spec_for(self, ep):
        t = self.tasks[ep["task_id"]]
        sc = self.cfg["scene"]
        return instantiate(t, ep["variant"], ep["seed_idx"], lang=self.exp["lang"],
                           global_seed=self.exp["global_seed"], perturb=ep["condition"].get("perturb", False),
                           ws=workspace(self.cfg), min_sep=sc.get("min_sep", 0.10), base_xy=sc.get("base_xy"))

    async def run_episode(self, ep) -> dict:
        cond = ep["condition"]
        spec = self.spec_for(ep)
        rec = {"episode_id": ep["episode_id"], "task_id": ep["task_id"], "family": spec["family"],
               "source_family": spec["source_family"], "variant": ep["variant"], "seed_idx": ep["seed_idx"],
               "rep": ep["rep"], "condition": cond, "model_label": ep["model"]["label"],
               "provider": ep["model"]["provider"], "model": ep["model"]["model"],
               "tier": ep["model"].get("tier"), "spec": spec, "started_at": datetime.now().isoformat(timespec="seconds"),
               "status": None}
        try:
            agent = self._agent(ep["model"])
            if hasattr(self.backend, "image_dir"):
                self.backend.image_dir = (self.out_dir / "images" / ep["episode_id"]) if self.save_images else None
            init = self.backend.reset(spec)
            bad = [c for c, p in spec["scene"].items()
                   if c not in init["objects"] or dist3(p, init["objects"][c]) > 0.005]
            if bad:
                raise RuntimeError(f"cena inicial nao bate com a semente: {bad}")
            system_prompt = make_system_prompt(cond)
            async with self.backend.tools(spec, cond) as tc:
                transcript = await asyncio.wait_for(agent.run(spec["prompt"], tc, system_prompt, spec),
                                                    timeout=self.exp["episode_timeout_s"])
            final = self.backend.finalize(spec)
            events = self.backend.events()
            metrics = scoring.score_episode(spec, init, final, transcript, events, cond,
                                            self.cfg["tolerances"], self.cfg["attribution"])
            price = self.cfg["prices"].get(ep["model"]["label"]) or self.cfg["prices"].get(ep["model"]["model"])
            rec.update(status="done", init_gt=init, final_gt=final, transcript=transcript, events=events,
                       metrics=metrics, model_version=transcript.get("model_version"),
                       cost_usd=episode_cost(transcript.get("usage_total"), price))
        except ProviderError as e:
            rec.update(status="api_error", error=str(e))
        except Exception as e:  # noqa: BLE001 -- qualquer falha do harness vira infra_error, com traceback
            rec.update(status="infra_error", error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc())
        rec["finished_at"] = datetime.now().isoformat(timespec="seconds")
        return rec

    async def run(self, plan, limit=None, budget_usd=None):
        done = skipped = 0
        spent = self.spent_usd()
        if budget_usd is not None:
            prices = self.cfg["prices"]
            unpriced = sorted({ep["model"]["label"] for ep in plan if ep["model"]["provider"] != "baseline"
                               and ep["model"]["label"] not in prices and ep["model"]["model"] not in prices})
            if unpriced:
                self.log(f"AVISO: sem preco em [prices], fora do orcamento de US$ {budget_usd}: {unpriced}")
        for i, ep in enumerate(plan):
            prev = self.existing(ep["episode_id"])
            attempts = (prev or {}).get("attempts", 0)
            if prev and (prev["status"] == "done" or attempts >= self.exp["max_attempts"]):
                skipped += 1
                continue
            if limit is not None and done >= limit:
                break
            stop_file = self.out_dir / "STOP"
            if stop_file.exists():
                # parada limpa entre episodios (matar o processo no meio de um
                # episodio deixa um cliente em modo passo a passo e trava o
                # CoppeliaSim inteiro). Para retomar: apagar STOP e rodar de novo.
                self.log(f"arquivo {stop_file} encontrado; parando entre episodios")
                break
            if budget_usd is not None and spent >= budget_usd:
                self.log(f"orcamento atingido (US$ {spent:.2f} >= {budget_usd}); parando")
                break
            rec = await self.run_episode(ep)
            rec["attempts"] = attempts + 1
            (self.ep_dir / f"{ep['episode_id']}.json").write_text(
                json.dumps(rec, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
            spent += rec.get("cost_usd") or 0.0
            done += 1
            m = rec.get("metrics") or {}
            self.log(f"[{i + 1}/{len(plan)}] {ep['episode_id']}: {rec['status']}"
                     + (f" S={int(m['S'])} C={int(m['C'])} relato={m['report']}" if m else f" ({rec.get('error')})"))
        return {"executed": done, "skipped": skipped, "spent_usd": round(spent, 4)}
