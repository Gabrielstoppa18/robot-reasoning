"""Carga do experiment.toml e congelamento (pre-registro) do experimento."""
import hashlib
import json
import subprocess
import tomllib
from datetime import datetime
from pathlib import Path

from . import predicates

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_config(path) -> dict:
    path = Path(path)
    cfg = tomllib.loads(path.read_text(encoding="utf-8"))
    cfg["_path"] = str(path)
    cfg["_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    exp = cfg["experiment"]
    exp.setdefault("lang", "pt")
    exp.setdefault("seeds", 3)
    exp.setdefault("max_turns", 25)
    exp.setdefault("max_tool_calls", 40)
    exp.setdefault("episode_timeout_s", 900)
    exp.setdefault("max_attempts", 3)
    cfg.setdefault("scene", {})
    cfg.setdefault("tolerances", {})
    cfg.setdefault("attribution", {})
    cfg.setdefault("prices", {})
    cfg.setdefault("baselines", [])
    return cfg


def resolve_path(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


def workspace(cfg) -> dict:
    sc = cfg["scene"]
    return {"x": sc.get("workspace_x", [0.20, 0.55]), "y": sc.get("workspace_y", [0.02, 0.42]),
            "r_max": sc.get("max_reach_xy", 0.62)}


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def tasks_hash(tasks: list[dict], split: str) -> str:
    sel = sorted((t for t in tasks if t["split"] == split), key=lambda t: t["id"])
    return hashlib.sha256(canonical_json(sel).encode("utf-8")).hexdigest()


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True,
                               timeout=10).stdout.strip()
        return out.stdout.strip() + ("+dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        return None


def freeze_record(cfg, tasks) -> dict:
    split = cfg["experiment"]["split"]
    return {
        "split": split,
        "tasks_sha256": tasks_hash(tasks, split),
        "n_instructions": sum(1 for t in tasks if t["split"] == split),
        "config_sha256": cfg["_sha256"],
        "tolerances": {**predicates.TOL, **cfg["tolerances"]},
        "global_seed": cfg["experiment"]["global_seed"],
        "git_commit": git_commit(),
        "frozen_at": datetime.now().isoformat(timespec="seconds"),
    }


def check_frozen(cfg, tasks, lock_path) -> list[str]:
    """Diferencas entre o experimento atual e o congelado (lista vazia = ok)."""
    lock_path = Path(lock_path)
    if not lock_path.exists():
        return [f"{lock_path} nao existe: rode scripts/bench_freeze.py antes do experimento principal"]
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    cur = freeze_record(cfg, tasks)
    return [f"{k}: congelado={lock.get(k)} atual={cur[k]}"
            for k in ("split", "tasks_sha256", "config_sha256", "tolerances", "global_seed") if lock.get(k) != cur[k]]
