"""Re-run only the failed tasks of a previously evaluated checkpoint.

This reads ``failed_tasks_ckpt_<N>.json`` produced by CheckpointWatcher and
invokes ``appworld run <eval_config> --override '{"config": {"task_ids": [...]}}'``
followed by ``appworld evaluate`` on the existing experiment to refresh the
score in-place.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime

from appworld.common.path_store import path_store
from appworld.common.utils import maybe_create_parent_directory


def _materialize_resume_config(template: str, exp_name: str, snapshot_path: str | None) -> str:
    """Create a per-checkpoint wrapper jsonnet so we hit the same outputs dir."""
    configs_dir = path_store.experiment_configs
    target = os.path.join(configs_dir, f"{exp_name}.jsonnet")
    if os.path.exists(target):
        return target
    if snapshot_path is None:
        raise FileNotFoundError(
            f"Wrapper config {target} does not exist and no snapshot path was provided "
            "to recreate it. Either keep the original wrapper or pass --snapshot."
        )
    wrapper = (
        f"local base = import './{template}.jsonnet';\n"
        f"base + {{\n"
        f"  config+: {{\n"
        f"    agent+: {{\n"
        f"      trained_playbook_file_path: {json.dumps(snapshot_path)},\n"
        f"    }},\n"
        f"  }},\n"
        f"}}\n"
    )
    with open(target, "w", encoding="utf-8") as f:
        f.write(wrapper)
    return target


def _parse_score(stdout: str) -> float | None:
    if not stdout:
        return None
    matches = re.findall(r"task_goal_completion[^\d\-]*([0-9]+(?:\.[0-9]+)?)", stdout)
    if matches:
        try:
            return float(matches[-1])
        except ValueError:
            return None
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Re-run failed tasks for a checkpoint and refresh its eval score."
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Checkpoint experiment name, e.g. eval_ckpt_30.",
    )
    parser.add_argument(
        "--failed", required=True, help="Path to failed_tasks_ckpt_<N>.json."
    )
    parser.add_argument(
        "--eval-config", required=True,
        help="Eval config TEMPLATE name (the one your watcher used).",
    )
    parser.add_argument(
        "--snapshot", default=None,
        help="Path to playbook snapshot (only needed if the per-checkpoint wrapper "
             "config no longer exists).",
    )
    parser.add_argument(
        "--num-processes", type=int, default=1,
        help="Parallelism for re-running failed tasks.",
    )
    parser.add_argument(
        "--dataset", default="test_normal",
        help="Dataset name used by appworld evaluate.",
    )
    parser.add_argument(
        "--results", default=None,
        help="Optional results JSONL to append the refreshed score to.",
    )
    parser.add_argument(
        "--root", default=None, help="AppWorld root directory."
    )
    args = parser.parse_args()

    if not os.path.exists(args.failed):
        print(f"failed-tasks file not found: {args.failed}", file=sys.stderr)
        return 2
    with open(args.failed, "r", encoding="utf-8") as f:
        failed_ids = json.load(f)
    if not isinstance(failed_ids, list) or not failed_ids:
        print(f"No failed tasks to re-run in {args.failed}.")
        return 0

    config_path = _materialize_resume_config(
        template=args.eval_config,
        exp_name=args.checkpoint,
        snapshot_path=args.snapshot,
    )
    print(f"Re-running {len(failed_ids)} failed tasks for '{args.checkpoint}' "
          f"using {config_path}")

    env = os.environ.copy()
    if args.root:
        env["APPWORLD_ROOT"] = args.root

    override = {"config": {"task_ids": failed_ids}}
    run_cmd = [
        "appworld", "run", args.checkpoint,
        "--override", json.dumps(override),
        "--num-processes", str(max(1, args.num_processes)),
    ]
    if args.root:
        run_cmd += ["--root", args.root]
    rc = subprocess.run(run_cmd, env=env).returncode
    if rc != 0:
        print(f"appworld run failed (exit={rc})", file=sys.stderr)
        return rc

    eval_cmd = ["appworld", "evaluate", args.checkpoint, args.dataset]
    if args.root:
        eval_cmd += ["--root", args.root]
    eval_proc = subprocess.run(eval_cmd, env=env, capture_output=True, text=True)
    print(eval_proc.stdout)
    if eval_proc.returncode != 0:
        print(eval_proc.stderr, file=sys.stderr)
        return eval_proc.returncode
    score = _parse_score(eval_proc.stdout)
    print(f"Refreshed score for {args.checkpoint}: {score}")

    if args.results and score is not None:
        maybe_create_parent_directory(args.results)
        m = re.search(r"_(\d+)$", args.checkpoint) or re.search(r"(\d+)$", args.checkpoint)
        task_idx = int(m.group(1)) if m else None
        entry = {
            "task_index": task_idx,
            "exp_name": args.checkpoint,
            "score": score,
            "rerun": True,
            "num_failed_input": len(failed_ids),
            "timestamp": datetime.utcnow().isoformat(),
        }
        with open(args.results, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"Appended refreshed entry to {args.results}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
