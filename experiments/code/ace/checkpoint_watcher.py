"""Background watcher that triggers test_normal evaluation for each new playbook checkpoint.

The watcher polls the playbook directory for new ``*_snapshot_<N>.txt`` files
emitted by the adaptation agent and, for each new file, runs:

  1. ``appworld run <eval_config_template> --override ...``
     with ``--num-processes`` for intra-checkpoint task parallelism.
  2. ``appworld evaluate <experiment_name> <dataset>`` to score the run.

Checkpoints are processed serially (one at a time) to avoid concurrent writes to
``evaluations/<dataset>.json``. Each checkpoint uses a unique experiment name
(``eval_ckpt_<N>``) so output trees never collide.
"""
from __future__ import annotations

import glob
import json
import os
import queue
import re
import signal
import subprocess
import threading
import time
from datetime import datetime
from typing import Any

from appworld.common.path_store import path_store
from appworld.common.utils import maybe_create_parent_directory


SNAPSHOT_RE = re.compile(r"_snapshot_(\d+)\.txt$")


def _append_jsonl(path: str, entry: dict[str, Any]) -> None:
    maybe_create_parent_directory(path)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _parse_score(stdout: str) -> float | None:
    """Extract task_goal_completion (%) from `appworld evaluate` stdout."""
    if not stdout:
        return None
    matches = re.findall(r"task_goal_completion[^\d\-]*([0-9]+(?:\.[0-9]+)?)", stdout)
    if matches:
        try:
            return float(matches[-1])
        except ValueError:
            return None
    return None


def _read_eval_json(experiment_name: str, dataset: str) -> dict | None:
    eval_path = os.path.join(
        path_store.experiment_outputs, experiment_name, "evaluations", f"{dataset}.json"
    )
    if not os.path.exists(eval_path):
        return None
    with open(eval_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_failed_task_ids(eval_json: dict) -> list[str]:
    individual = eval_json.get("individual", {}) or {}
    failed: list[str] = []
    for tid, tt in individual.items():
        if isinstance(tt, dict) and not tt.get("success", tt.get("passes_fully", False)):
            failed.append(tid)
    return sorted(failed)


class CheckpointWatcher(threading.Thread):
    """Polls for new playbook snapshots and serially evaluates each one."""

    def __init__(
        self,
        playbook_base: str,
        eval_config_template: str,
        results_path: str,
        failed_dir: str | None = None,
        eval_dataset: str = "test_normal",
        eval_num_processes: int = 1,
        poll_interval: float = 30.0,
        appworld_root: str | None = None,
        verbose: bool = True,
        wrapper_configs_dir: str | None = None,
        eval_failure_log_path: str | None = None,
        appworld_outputs_link_dir: str | None = None,
    ) -> None:
        super().__init__(daemon=True)
        if not playbook_base.endswith(".txt"):
            raise ValueError(
                f"playbook_base must end with .txt; got: {playbook_base}"
            )
        self.playbook_base = os.path.abspath(playbook_base)
        self.snapshot_glob = self.playbook_base[:-4] + "_snapshot_*.txt"
        self.eval_config_template = eval_config_template
        self.results_path = os.path.abspath(results_path)
        self.failed_dir = (
            os.path.abspath(failed_dir)
            if failed_dir
            else os.path.dirname(self.results_path) or "."
        )
        self.eval_dataset = eval_dataset
        self.eval_num_processes = max(1, int(eval_num_processes))
        self.poll_interval = max(1.0, float(poll_interval))
        self.appworld_root = appworld_root
        self.verbose = verbose
        # If set, wrapper jsonnet files are written here (their canonical home,
        # under the run-dir) and ALSO mirrored into path_store.experiment_configs
        # so `appworld run <name>` can find them.
        self.wrapper_configs_dir = (
            os.path.abspath(wrapper_configs_dir) if wrapper_configs_dir else None
        )
        # Optional shared LLM failure log path for eval runs (override into the
        # generated wrapper jsonnet for each model_config).
        self.eval_failure_log_path = (
            os.path.abspath(eval_failure_log_path) if eval_failure_log_path else None
        )
        # If set, after each successful eval we symlink
        # experiments/outputs/<exp_name> -> <appworld_outputs_link_dir>/<exp_name>
        self.appworld_outputs_link_dir = (
            os.path.abspath(appworld_outputs_link_dir)
            if appworld_outputs_link_dir
            else None
        )

        # Per-checkpoint eval logs (stdout+stderr of `appworld run` and
        # `appworld evaluate`) so failures can be diagnosed without re-running.
        self.eval_logs_dir = os.path.join(self.failed_dir, "eval_logs")

        self._stop_event = threading.Event()
        self._eval_queue: "queue.Queue[str | None]" = queue.Queue()
        self._seen: set[str] = set()
        self._worker = threading.Thread(target=self._eval_worker, daemon=True)
        # Track currently-running eval subprocess so stop_and_join() can kill
        # the entire process group (each child is launched in its own session
        # so terminal SIGINT does NOT propagate to it automatically).
        self._active_proc: subprocess.Popen | None = None
        self._active_proc_lock = threading.Lock()
        # Path of the log file the currently-running checkpoint is writing to;
        # used by _eval_worker to surface it in the error JSONL entry.
        self._current_eval_log_path: str | None = None

    # ------------------------------------------------------------------
    # Public lifecycle
    # ------------------------------------------------------------------
    def run(self) -> None:
        self._log("Watcher started.")
        self._worker.start()
        try:
            while not self._stop_event.is_set():
                self._scan_once()
                if self._stop_event.wait(self.poll_interval):
                    break
            # Final scan to pick up any snapshots written between last poll and stop.
            self._scan_once()
        finally:
            # If a hard stop was requested, drain remaining queue items without
            # running them so the worker can exit immediately.
            if self._stop_event.is_set():
                self._drain_queue_without_running()
            else:
                self._eval_queue.join()
            self._eval_queue.put(None)
            self._worker.join()
            self._log("Watcher stopped.")

    def stop_and_join(self, timeout: float | None = None) -> None:
        """Signal the watcher to stop and kill any in-flight eval subprocess.

        Terminal Ctrl+C does NOT reach our eval children because they are
        spawned in their own process group (see ``_spawn_and_wait``), so they
        are not part of the foreground process group and never see SIGINT
        from the controlling terminal. We must kill the process group
        explicitly here.
        """
        self._stop_event.set()
        self._kill_active_proc()
        self.join(timeout=timeout)

    def _drain_queue_without_running(self) -> None:
        try:
            while True:
                self._eval_queue.get_nowait()
                self._eval_queue.task_done()
        except queue.Empty:
            return

    def _kill_active_proc(self) -> None:
        with self._active_proc_lock:
            proc = self._active_proc
        if proc is None or proc.poll() is not None:
            return
        try:
            pgid = os.getpgid(proc.pid)
        except ProcessLookupError:
            return
        self._log(f"Sending SIGTERM to eval process group pgid={pgid}")
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=15)
            return
        except subprocess.TimeoutExpired:
            self._log(f"Eval pgid={pgid} did not exit within 15s; sending SIGKILL")
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._log(f"[WARN] Eval pgid={pgid} still alive after SIGKILL")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _scan_once(self) -> None:
        try:
            paths = sorted(glob.glob(self.snapshot_glob), key=self._snapshot_index)
        except Exception as exc:
            self._log(f"[WARN] glob failed: {exc}")
            return
        for path in paths:
            if path in self._seen:
                continue
            idx = self._snapshot_index(path)
            if idx < 0:
                continue
            self._seen.add(path)
            self._log(f"Detected new checkpoint: {os.path.basename(path)} (idx={idx})")
            self._eval_queue.put(path)

    @staticmethod
    def _snapshot_index(path: str) -> int:
        m = SNAPSHOT_RE.search(path)
        return int(m.group(1)) if m else -1

    def _eval_worker(self) -> None:
        while True:
            item = self._eval_queue.get()
            try:
                if item is None:
                    return
                try:
                    self._evaluate_checkpoint(item)
                except Exception as exc:
                    log_path = self._current_eval_log_path
                    log_hint = f" | log={log_path}" if log_path else ""
                    self._log(f"[ERROR] eval failed for {item}: {exc}{log_hint}")
                    _append_jsonl(
                        self.results_path,
                        {
                            "snapshot_path": item,
                            "task_index": self._snapshot_index(item),
                            "error": str(exc)[:500],
                            "eval_log_path": log_path,
                            "timestamp": datetime.utcnow().isoformat(),
                        },
                    )
            finally:
                self._current_eval_log_path = None
                self._eval_queue.task_done()

    def _evaluate_checkpoint(self, snapshot_path: str) -> None:
        task_idx = self._snapshot_index(snapshot_path)
        exp_name = f"eval_ckpt_{task_idx}"
        os.makedirs(self.eval_logs_dir, exist_ok=True)
        log_path = os.path.join(self.eval_logs_dir, f"{exp_name}.log")
        self._current_eval_log_path = log_path
        # Truncate at start so each (re)attempt starts with a clean log.
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(
                f"# eval log for {exp_name}\n"
                f"# snapshot: {snapshot_path}\n"
                f"# started:  {datetime.utcnow().isoformat()}Z\n"
            )
        self._log(
            f"Evaluating checkpoint #{task_idx} as experiment '{exp_name}' "
            f"(log={log_path}) ..."
        )

        env = os.environ.copy()
        if self.appworld_root:
            env["APPWORLD_ROOT"] = self.appworld_root

        # In `appworld run <experiment_name>`, the positional argument doubles as
        # the config file name (configs/<experiment_name>.jsonnet) AND the output
        # directory name. To get a unique output directory per checkpoint while
        # reusing the same template, we materialize a per-checkpoint jsonnet that
        # imports the template.
        ckpt_config_name = exp_name  # configs/eval_ckpt_<N>.jsonnet
        ckpt_config_path = self._materialize_checkpoint_config(
            ckpt_config_name, snapshot_path
        )

        if self._stop_event.is_set():
            self._log(f"Stop requested before evaluating {snapshot_path}; skipping.")
            return

        override = {
            "config": {
                "agent": {
                    "trained_playbook_file_path": snapshot_path,
                }
            }
        }
        run_cmd = [
            "appworld",
            "run",
            ckpt_config_name,
            "--override",
            json.dumps(override),
            "--num-processes",
            str(self.eval_num_processes),
        ]
        if self.appworld_root:
            run_cmd += ["--root", self.appworld_root]
        t0 = time.time()
        run_rc = self._spawn_and_wait(run_cmd, env=env, log_path=log_path)
        run_duration = time.time() - t0
        if self._stop_event.is_set():
            self._log(f"Stop requested while running {snapshot_path}; aborting.")
            return
        if run_rc != 0:
            raise RuntimeError(
                f"appworld run failed for checkpoint #{task_idx} "
                f"(exit={run_rc}), config={ckpt_config_path}, log={log_path}"
            )

        eval_cmd = ["appworld", "evaluate", exp_name, self.eval_dataset]
        if self.appworld_root:
            eval_cmd += ["--root", self.appworld_root]
        eval_rc, eval_stdout, eval_stderr = self._spawn_and_capture(
            eval_cmd, env=env, log_path=log_path
        )
        if self._stop_event.is_set():
            self._log(f"Stop requested while scoring {snapshot_path}; aborting.")
            return
        if eval_rc != 0:
            raise RuntimeError(
                f"appworld evaluate failed for {exp_name} "
                f"(exit={eval_rc}, log={log_path}): {eval_stderr[-500:]}"
            )

        # Surface the appworld output dir under the run-dir for easy browsing.
        self._link_appworld_output(exp_name)

        score = _parse_score(eval_stdout)
        eval_json = _read_eval_json(exp_name, self.eval_dataset) or {}
        if score is None:
            try:
                score = float(eval_json["aggregate"]["task_goal_completion"])
            except (KeyError, TypeError, ValueError):
                score = None

        failed_ids = _extract_failed_task_ids(eval_json) if eval_json else []
        failed_path = os.path.join(
            self.failed_dir, f"failed_tasks_ckpt_{task_idx}.json"
        )
        if eval_json:
            maybe_create_parent_directory(failed_path)
            with open(failed_path, "w", encoding="utf-8") as f:
                json.dump(failed_ids, f, indent=2)

        entry = {
            "task_index": task_idx,
            "exp_name": exp_name,
            "snapshot_path": snapshot_path,
            "score": score,
            "num_failed": len(failed_ids),
            "failed_tasks_path": failed_path if eval_json else None,
            "eval_log_path": log_path,
            "run_duration_s": round(run_duration, 1),
            "timestamp": datetime.utcnow().isoformat(),
        }
        _append_jsonl(self.results_path, entry)
        self._log(
            f"Checkpoint #{task_idx} done | score={score} | failed={len(failed_ids)} | "
            f"run={run_duration:.1f}s | exp={exp_name}"
        )

    def _spawn_and_wait(
        self, cmd: list[str], env: dict, log_path: str | None = None
    ) -> int:
        """Popen-based subprocess that we can kill via process group on stop.

        We use ``process_group=0`` (Python 3.11+) instead of
        ``start_new_session=True`` because ``appworld run --num-processes N``
        internally calls ``os.setpgrp()`` to manage its own worker pool, which
        fails with EPERM if the child is already a session leader (which
        ``setsid()`` would make it). ``process_group=0`` only puts the child
        in its own process group without becoming a session leader, so:
          - cli.py's ``os.setpgrp()`` becomes a harmless no-op (already pgrp leader),
          - we can still kill the whole tree via ``os.killpg(pgid, SIGTERM)``,
          - the controlling terminal's SIGINT does not propagate (child is not
            in the foreground process group).

        If ``log_path`` is given, stdout+stderr are appended to it (so the
        original failure trace can be diagnosed without re-running). Otherwise
        the child inherits the parent's stdout/stderr.
        """
        if log_path:
            with open(log_path, "ab") as logf:
                logf.write(
                    f"\n# $ {' '.join(cmd)}\n".encode("utf-8", errors="replace")
                )
                proc = subprocess.Popen(
                    cmd, env=env, process_group=0,
                    stdout=logf, stderr=subprocess.STDOUT,
                )
                with self._active_proc_lock:
                    self._active_proc = proc
                try:
                    return proc.wait()
                finally:
                    with self._active_proc_lock:
                        self._active_proc = None
        proc = subprocess.Popen(cmd, env=env, process_group=0)
        with self._active_proc_lock:
            self._active_proc = proc
        try:
            return proc.wait()
        finally:
            with self._active_proc_lock:
                self._active_proc = None

    def _spawn_and_capture(
        self, cmd: list[str], env: dict, log_path: str | None = None
    ) -> tuple[int, str, str]:
        """Like _spawn_and_wait but capture stdout/stderr.

        Captured output is also appended to ``log_path`` (if given) so the full
        eval trace lives in one place alongside the run output.

        Uses ``process_group=0`` for the same reason as ``_spawn_and_wait``.
        """
        proc = subprocess.Popen(
            cmd, env=env, process_group=0,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        with self._active_proc_lock:
            self._active_proc = proc
        try:
            out, err = proc.communicate()
        finally:
            with self._active_proc_lock:
                self._active_proc = None
        if log_path:
            try:
                with open(log_path, "a", encoding="utf-8") as logf:
                    logf.write(f"\n# $ {' '.join(cmd)}\n")
                    if out:
                        logf.write(out)
                    if err:
                        logf.write("\n# --- stderr ---\n")
                        logf.write(err)
            except OSError as exc:
                self._log(f"[WARN] failed to append to {log_path}: {exc}")
        return proc.returncode, out or "", err or ""

    def _materialize_checkpoint_config(
        self, ckpt_config_name: str, snapshot_path: str
    ) -> str:
        """Create a wrapper jsonnet that imports the eval template.

        The wrapper's canonical home is ``self.wrapper_configs_dir`` (under the
        run-dir, when set). We then mirror it into ``path_store.experiment_configs``
        because ``appworld run <name>`` only looks there. This way users see the
        generated configs alongside everything else under their run-dir.
        """
        appworld_configs_dir = path_store.experiment_configs
        template_path = os.path.join(
            appworld_configs_dir, f"{self.eval_config_template}.jsonnet"
        )
        if not os.path.exists(template_path):
            raise FileNotFoundError(
                f"Eval config template not found: {template_path}"
            )
        agent_overrides = [
            f"      trained_playbook_file_path: {json.dumps(snapshot_path)},\n"
        ]
        # Inject failure_log_path only into model_configs that the eval template
        # actually defines. The ACE evaluation agent uses generator only; using
        # `+:` on a key that does not exist in the base template would create a
        # bogus model_config containing only `failure_log_path`, which breaks
        # appworld's config validation at load time (immediate exit=1).
        if self.eval_failure_log_path:
            agent_overrides.append(
                f"      generator_model_config+: {{ failure_log_path: "
                f"{json.dumps(self.eval_failure_log_path)} }},\n"
            )

        # Import the template by absolute path so the wrapper works regardless
        # of which directory contains it.
        abs_template_path = os.path.abspath(template_path)
        wrapper = (
            f"local base = import {json.dumps(abs_template_path)};\n"
            f"base + {{\n"
            f"  config+: {{\n"
            f"    agent+: {{\n"
            f"{''.join(agent_overrides)}"
            f"    }},\n"
            f"  }},\n"
            f"}}\n"
        )

        canonical_dir = self.wrapper_configs_dir or appworld_configs_dir
        canonical_path = os.path.join(canonical_dir, f"{ckpt_config_name}.jsonnet")
        os.makedirs(canonical_dir, exist_ok=True)
        with open(canonical_path, "w", encoding="utf-8") as f:
            f.write(wrapper)

        # Mirror into appworld's expected configs dir if different.
        appworld_path = os.path.join(
            appworld_configs_dir, f"{ckpt_config_name}.jsonnet"
        )
        if os.path.abspath(canonical_path) != os.path.abspath(appworld_path):
            try:
                if os.path.lexists(appworld_path):
                    os.remove(appworld_path)
                os.symlink(os.path.abspath(canonical_path), appworld_path)
            except OSError:
                # Symlink may fail (e.g. cross-device or permission); fall back
                # to writing the same file at both locations.
                with open(appworld_path, "w", encoding="utf-8") as f:
                    f.write(wrapper)
        return canonical_path

    def _link_appworld_output(self, experiment_name: str) -> None:
        """Symlink experiments/outputs/<exp_name> into the run-dir for one-stop browsing."""
        if not self.appworld_outputs_link_dir:
            return
        src = os.path.join(path_store.experiment_outputs, experiment_name)
        if not os.path.exists(src):
            return
        os.makedirs(self.appworld_outputs_link_dir, exist_ok=True)
        link_path = os.path.join(self.appworld_outputs_link_dir, experiment_name)
        try:
            if os.path.lexists(link_path):
                # Refresh existing link only if it points elsewhere.
                if os.path.islink(link_path) and os.readlink(link_path) == os.path.abspath(src):
                    return
                os.remove(link_path)
            os.symlink(os.path.abspath(src), link_path)
        except OSError as exc:
            self._log(f"[WARN] failed to symlink {src} -> {link_path}: {exc}")

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[CheckpointWatcher] {message}", flush=True)
