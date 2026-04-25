"""Wrappers around the repo's `appworld run` / `appworld evaluate` flow.

The pipeline doesn't reimplement the agent loop. It generates a stage-local
jsonnet config, writes/symlinks it under `experiments/configs/` so
`appworld run <name>` can find it, then shells out. Per-task results are
read back using the repo's own `evaluate_task()` + `lm_calls.jsonl`.

Subprocess is invoked via the ace conda env's python (`APPWORLD_PY` env
var, falling back to the current interpreter). The caller is responsible
for having `APPWORLD_ROOT` set in the environment.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from typing import Any

from . import paths
from .io import read_jsonl, read_text


def _python_bin() -> str:
    return os.environ.get("APPWORLD_PY") or sys.executable


def write_stage_config(jsonnet_text: str, experiment_name: str, mirror_dir: str) -> tuple[str, str]:
    """Write a stage-local jsonnet config and mirror it into experiments/configs/.

    Returns (canonical_path, mirrored_path). If the mirrored path already
    exists with identical contents, skip overwriting it.
    """
    canonical = os.path.join(mirror_dir, f"{experiment_name}.jsonnet")
    os.makedirs(os.path.dirname(canonical), exist_ok=True)
    with open(canonical, "w", encoding="utf-8") as f:
        f.write(jsonnet_text)
    mirror = os.path.join(paths.EXPERIMENT_CONFIGS_DIR, f"{experiment_name}.jsonnet")
    if os.path.lexists(mirror):
        try:
            existing = read_text(mirror)
            if existing == jsonnet_text:
                return canonical, mirror
        except OSError:
            pass
        try:
            os.unlink(mirror)
        except OSError:
            pass
    shutil.copyfile(canonical, mirror)
    return canonical, mirror


def run_appworld(
    experiment_name: str,
    override: dict[str, Any] | None = None,
    task_id: str | None = None,
    num_processes: int = 1,
    stream_output: bool = True,
) -> int:
    """Invoke `appworld run <experiment_name>` as a subprocess.

    Returns the child's exit code. stdout/stderr are inherited so logs flow
    through the current stage_context's Tee and land in the stage log.
    """
    cmd: list[str] = [
        _python_bin(), "-m", "appworld.cli", "run", experiment_name,
        "--root", paths.REPO_ROOT,
        "--num-processes", str(num_processes),
    ]
    if task_id:
        cmd.extend(["--task-id", task_id])
    if override:
        cmd.extend(["--override", json.dumps(override)])
    print(f"[run_appworld] $ {' '.join(cmd)}", flush=True)
    env = os.environ.copy()
    env["APPWORLD_ROOT"] = paths.REPO_ROOT
    env.setdefault("APPWORLD_PROJECT_PATH", paths.REPO_ROOT)
    if stream_output:
        proc = subprocess.run(cmd, env=env)
        return proc.returncode
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    print(proc.stdout)
    print(proc.stderr, file=sys.stderr)
    return proc.returncode


def evaluate_one_task(task_id: str, experiment_name: str) -> dict[str, Any]:
    """Import-level use of `evaluate_task` for a single task.

    Must be called inside the ace env (i.e. from stage code that runs under
    APPWORLD_PY). Returns {success: bool, num_failed: int, num_passed: int}.
    """
    from appworld.evaluator import evaluate_task
    result = evaluate_task(task_id=task_id, experiment_name=experiment_name)
    tracker = result[0] if isinstance(result, tuple) else result
    num_failed = int(getattr(tracker, "fail_count", None) or len(getattr(tracker, "failures", []) or []))
    num_passed = int(getattr(tracker, "pass_count", None) or len(getattr(tracker, "passes", []) or []))
    success_attr = getattr(tracker, "success", None)
    if success_attr is None:
        success = (num_failed == 0 and num_passed > 0)
    else:
        success = bool(success_attr)
    return {
        "success": success,
        "num_failed": num_failed,
        "num_passed": num_passed,
    }


def read_trajectory(experiment_name: str, task_id: str) -> list[dict[str, Any]]:
    """Read per-step LLM call records from lm_calls.jsonl.

    Returns a list of dicts (one per step) with at least:
      - `messages`: the full input messages to the model
      - `content`: the assistant's output text
      - `step_idx`: running index starting at 0
    Empty list if the file doesn't exist.
    """
    lm_path = os.path.join(
        paths.task_output_dir(experiment_name, task_id),
        "logs", "lm_calls.jsonl",
    )
    if not os.path.exists(lm_path):
        return []
    out: list[dict[str, Any]] = []
    for i, entry in enumerate(read_jsonl(lm_path)):
        inp = entry.get("input") or {}
        output = entry.get("output") or {}
        content = ""
        if isinstance(output, dict):
            choices = output.get("choices") or []
            if choices:
                msg = (choices[0] or {}).get("message") or {}
                content = msg.get("content", "") or ""
            if not content:
                content = output.get("content", "") or ""
        out.append({
            "step_idx": i,
            "messages": inp.get("messages", []),
            "content": content,
            "raw": entry,
        })
    return out


def read_task_cost(experiment_name: str, task_id: str) -> float:
    cost_path = os.path.join(
        paths.task_output_dir(experiment_name, task_id),
        "misc", "cost.txt",
    )
    if not os.path.exists(cost_path):
        return 0.0
    try:
        return float(read_text(cost_path).strip())
    except (ValueError, OSError):
        return 0.0


def task_has_output(experiment_name: str, task_id: str) -> bool:
    d = paths.task_output_dir(experiment_name, task_id)
    return os.path.isdir(os.path.join(d, "logs"))
