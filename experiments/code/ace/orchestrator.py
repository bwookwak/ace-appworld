"""Top-level orchestrator: run adaptation + concurrent checkpoint evaluation.

Workflow:
  1. Compute a single ``--run-dir`` and route ALL output paths through it
     (playbooks, snapshots, checkpoint state, reflections, llm failures,
     learning curve, plot, failed-task lists, generated wrapper configs,
     and symlinks to AppWorld's per-experiment output trees).
  2. Start a CheckpointWatcher background thread that polls the playbook
     directory for new ``*_snapshot_<N>.txt`` snapshots.
  3. Spawn the adaptation run as a child process (``appworld run ...``)
     with overrides that point all custom paths inside ``--run-dir``.
  4. On Ctrl+C: SIGTERM both the adaptation child process group and the
     watcher's in-flight eval child process group.
  5. When the adaptation finishes, wait for the watcher to drain its eval
     queue and render the final learning curve.

Layout under ``--run-dir``:

    <run_dir>/
    +-- playbooks/
    |   +-- trained_playbook.txt
    |   +-- trained_playbook_snapshot_<N>.txt
    |   +-- trained_playbook_checkpoint_state.json
    |   +-- reflections.jsonl
    |   +-- llm_failures.jsonl                  (adaptation)
    |   +-- llm_failures_eval.jsonl             (eval checkpoints)
    +-- results/
    |   +-- learning_curve.jsonl
    |   +-- learning_curve.png
    |   +-- failed_tasks_ckpt_<N>.json
    +-- configs/
    |   +-- eval_ckpt_<N>.jsonnet               (canonical wrappers; mirrored
    |                                            into experiments/configs/)
    +-- appworld_outputs/
        +-- adaptation -> experiments/outputs/<adaptation_config>/
        +-- eval_ckpt_<N> -> experiments/outputs/eval_ckpt_<N>/
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from typing import Any

from appworld.common.path_store import path_store

from appworld_experiments.code.ace.checkpoint_watcher import CheckpointWatcher
from appworld_experiments.code.ace.plot_learning_curve import plot_learning_curve


def _kill_process_group(proc: subprocess.Popen, label: str) -> None:
    if proc.poll() is not None:
        return
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    print(f"[orchestrator] Sending SIGTERM to {label} pgid={pgid}", flush=True)
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=20)
        return
    except subprocess.TimeoutExpired:
        print(
            f"[orchestrator] {label} pgid={pgid} did not exit within 20s; "
            "sending SIGKILL",
            flush=True,
        )
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        print(
            f"[orchestrator] [WARN] {label} pgid={pgid} still alive after SIGKILL",
            flush=True,
        )


def _ensure_symlink(src: str, link_path: str) -> None:
    """Create or refresh ``link_path`` pointing at ``src``."""
    src_abs = os.path.abspath(src)
    os.makedirs(os.path.dirname(link_path), exist_ok=True)
    try:
        if os.path.lexists(link_path):
            if os.path.islink(link_path) and os.readlink(link_path) == src_abs:
                return
            os.remove(link_path)
        os.symlink(src_abs, link_path)
    except OSError as exc:
        print(f"[orchestrator] [WARN] failed to symlink {src_abs} -> {link_path}: {exc}",
              flush=True)


def _deep_merge(dst: dict, src: dict) -> dict:
    """Recursively merge ``src`` into ``dst`` (mutating dst). dict-leaves only."""
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge(dst[k], v)
        else:
            dst[k] = v
    return dst


def _run_adaptation(
    adaptation_config: str,
    adaptation_override: dict[str, Any] | None,
    appworld_root: str | None,
    num_processes: int,
    interrupted_event: threading.Event,
) -> int:
    cmd = ["appworld", "run", adaptation_config, "--num-processes", str(num_processes)]
    if adaptation_override:
        cmd += ["--override", json.dumps(adaptation_override)]
    if appworld_root:
        cmd += ["--root", appworld_root]
    print(f"[orchestrator] launching adaptation: {' '.join(cmd)}", flush=True)
    env = os.environ.copy()
    if appworld_root:
        env["APPWORLD_ROOT"] = appworld_root
    # Use process_group=0 (Python 3.11+) instead of start_new_session=True so
    # that `appworld run --num-processes N` (which calls os.setpgrp() inside
    # cli.py) does not crash with EPERM. See checkpoint_watcher._spawn_and_wait
    # for the full rationale.
    proc = subprocess.Popen(cmd, env=env, process_group=0)
    try:
        while True:
            try:
                rc = proc.wait(timeout=1.0)
                return rc
            except subprocess.TimeoutExpired:
                if interrupted_event.is_set():
                    _kill_process_group(proc, "adaptation")
                    return proc.returncode if proc.returncode is not None else 130
    except KeyboardInterrupt:
        interrupted_event.set()
        _kill_process_group(proc, "adaptation")
        return 130


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run ACE adaptation with concurrent checkpoint evaluation."
    )
    parser.add_argument(
        "--adaptation-config", required=True,
        help="Adaptation experiment config name (without .jsonnet).",
    )
    parser.add_argument(
        "--eval-config", required=True,
        help="Eval config template name (without .jsonnet).",
    )
    parser.add_argument(
        "--run-dir", default=None,
        help=(
            "Single output root for this run. Defaults to "
            "experiments/runs/<adaptation_config>. All custom artifacts "
            "(playbooks, results, plots, failed_tasks, generated wrapper "
            "configs, and symlinks to AppWorld outputs) live here."
        ),
    )
    parser.add_argument(
        "--playbook-base", default=None,
        help=(
            "Optional override for the trained playbook .txt path. "
            "Defaults to <run-dir>/playbooks/trained_playbook.txt."
        ),
    )
    parser.add_argument(
        "--results", default=None,
        help="Optional override for results.jsonl path. Defaults to <run-dir>/results/learning_curve.jsonl.",
    )
    parser.add_argument(
        "--plot", default=None,
        help="Optional override for plot PNG path. Defaults to <run-dir>/results/learning_curve.png.",
    )
    parser.add_argument(
        "--failed-dir", default=None,
        help="Optional override for failed_tasks dir. Defaults to <run-dir>/results.",
    )
    parser.add_argument(
        "--eval-dataset", default="test_normal",
        help="Dataset to evaluate on. Default: test_normal.",
    )
    parser.add_argument(
        "--eval-num-processes", type=int, default=1,
        help="Parallel processes inside each checkpoint eval.",
    )
    parser.add_argument(
        "--adaptation-num-processes", type=int, default=1,
        help="Parallel processes for the adaptation run.",
    )
    parser.add_argument(
        "--poll-interval", type=float, default=30.0,
        help="Seconds between checkpoint polls.",
    )
    parser.add_argument(
        "--adaptation-override", default=None,
        help=(
            "Optional JSON string of additional deep-overrides for the "
            "adaptation config (merged with orchestrator's path overrides; "
            "user values win on conflict)."
        ),
    )
    parser.add_argument(
        "--root", default=None,
        help="AppWorld root directory (sets APPWORLD_ROOT for child processes).",
    )
    parser.add_argument(
        "--no-plot", action="store_true", help="Skip final plot generation."
    )
    parser.add_argument(
        "--watcher-shutdown-timeout", type=float, default=60.0,
        help="Max seconds to wait for the watcher to stop after a Ctrl+C.",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Resolve the central run directory and derive every output path.
    # ------------------------------------------------------------------
    appworld_root = os.path.abspath(args.root) if args.root else os.getcwd()
    default_run_dir = os.path.join(
        appworld_root, "experiments", "runs", args.adaptation_config
    )
    run_dir = os.path.abspath(args.run_dir) if args.run_dir else default_run_dir
    playbooks_dir = os.path.join(run_dir, "playbooks")
    results_dir = os.path.join(run_dir, "results")
    configs_dir = os.path.join(run_dir, "configs")
    appworld_outputs_link_dir = os.path.join(run_dir, "appworld_outputs")
    for d in (playbooks_dir, results_dir, configs_dir, appworld_outputs_link_dir):
        os.makedirs(d, exist_ok=True)

    playbook_base = (
        os.path.abspath(args.playbook_base)
        if args.playbook_base
        else os.path.join(playbooks_dir, "trained_playbook.txt")
    )
    results_path = (
        os.path.abspath(args.results)
        if args.results
        else os.path.join(results_dir, "learning_curve.jsonl")
    )
    plot_path = (
        os.path.abspath(args.plot)
        if args.plot
        else os.path.join(results_dir, "learning_curve.png")
    )
    failed_dir = (
        os.path.abspath(args.failed_dir) if args.failed_dir else results_dir
    )

    adaptation_failure_log = os.path.join(playbooks_dir, "llm_failures.jsonl")
    eval_failure_log = os.path.join(playbooks_dir, "llm_failures_eval.jsonl")

    print(f"[orchestrator] run_dir = {run_dir}", flush=True)
    print(f"[orchestrator]   playbooks/    -> {playbooks_dir}", flush=True)
    print(f"[orchestrator]   results/      -> {results_dir}", flush=True)
    print(f"[orchestrator]   configs/      -> {configs_dir}", flush=True)
    print(f"[orchestrator]   appworld_outputs/ -> {appworld_outputs_link_dir}", flush=True)

    # ------------------------------------------------------------------
    # Build the adaptation override (merged with user-supplied overrides).
    # ------------------------------------------------------------------
    failure_override = {"failure_log_path": adaptation_failure_log}
    auto_override: dict[str, Any] = {
        "config": {
            "agent": {
                "trained_playbook_file_path": playbook_base,
                "generator_model_config": failure_override,
                "reflector_model_config": failure_override,
                "curator_model_config": failure_override,
            }
        }
    }
    user_override = (
        json.loads(args.adaptation_override) if args.adaptation_override else {}
    )
    adaptation_override = _deep_merge(auto_override, user_override) if user_override else auto_override

    # ------------------------------------------------------------------
    # Signal-driven interruption.
    # ------------------------------------------------------------------
    interrupted_event = threading.Event()
    interrupt_count = {"n": 0}

    def _on_sigint(signum, frame):
        interrupt_count["n"] += 1
        interrupted_event.set()
        if interrupt_count["n"] == 1:
            print(
                "\n[orchestrator] Ctrl+C received. Initiating graceful shutdown. "
                "Press Ctrl+C again to force exit.",
                flush=True,
            )
        else:
            print("[orchestrator] Second Ctrl+C; forcing immediate exit.", flush=True)
            os._exit(130)

    signal.signal(signal.SIGINT, _on_sigint)
    signal.signal(signal.SIGTERM, _on_sigint)

    # ------------------------------------------------------------------
    # Watcher knows where to put its wrapper configs, eval failure log,
    # and the appworld_outputs symlinks.
    # ------------------------------------------------------------------
    watcher = CheckpointWatcher(
        playbook_base=playbook_base,
        eval_config_template=args.eval_config,
        results_path=results_path,
        failed_dir=failed_dir,
        eval_dataset=args.eval_dataset,
        eval_num_processes=args.eval_num_processes,
        poll_interval=args.poll_interval,
        appworld_root=args.root,
        wrapper_configs_dir=configs_dir,
        eval_failure_log_path=eval_failure_log,
        appworld_outputs_link_dir=appworld_outputs_link_dir,
    )
    watcher.start()
    rc = 0
    try:
        rc = _run_adaptation(
            args.adaptation_config,
            adaptation_override,
            args.root,
            args.adaptation_num_processes,
            interrupted_event,
        )
        # Symlink AppWorld's adaptation output dir into the run-dir hub.
        adaptation_appworld_dir = os.path.join(
            path_store.experiment_outputs, args.adaptation_config
        )
        if os.path.exists(adaptation_appworld_dir):
            _ensure_symlink(
                adaptation_appworld_dir,
                os.path.join(appworld_outputs_link_dir, "adaptation"),
            )
        if rc != 0:
            print(f"[orchestrator] adaptation exited with code {rc}", file=sys.stderr)
    finally:
        if interrupted_event.is_set():
            print(
                "[orchestrator] Interruption flagged; tearing down watcher and "
                "in-flight eval subprocess (if any).",
                flush=True,
            )
        else:
            print("[orchestrator] adaptation finished; waiting for watcher to drain...")
        t0 = time.time()
        watcher.stop_and_join(timeout=args.watcher_shutdown_timeout)
        if watcher.is_alive():
            print(
                f"[orchestrator] [WARN] watcher did not stop within "
                f"{args.watcher_shutdown_timeout}s; exiting anyway.",
                file=sys.stderr,
            )
        else:
            print(
                f"[orchestrator] watcher stopped after {time.time() - t0:.1f}s.",
                flush=True,
            )

    if (
        not interrupted_event.is_set()
        and not args.no_plot
        and os.path.exists(results_path)
    ):
        try:
            plot_learning_curve(results_path, plot_path)
        except Exception as exc:
            print(f"[orchestrator] failed to render plot: {exc}", file=sys.stderr)

    if interrupted_event.is_set():
        return 130
    return rc


if __name__ == "__main__":
    sys.exit(main())
