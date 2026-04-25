"""Stage 2 (wrapper) — runs Stage 2a (agent) then Stage 2b (analyze).

Kept for backward compatibility with the original single-stage CLI. New
code should prefer the split form:

  python -m memory_utility.stage2a_run    --memory ... --out-dir <D>
  python -m memory_utility.stage2b_analyze --memory ... --stage2a-dir <D> --out-dir <D>

This wrapper calls them in sequence. If Stage 2a's appworld agent crashes
mid-run, Stage 2b still analyzes whatever output landed on disk.
"""
from __future__ import annotations

import argparse
import sys

from . import stage2a_run, stage2b_analyze


def run(
    memory_path: str,
    out_dir: str,
    tasks: str = "dev",
    sample_size: int | None = None,
    task_ids_csv: str | None = None,
    reference_detection: str = "both",
    judge_model: str = "gpt-4o-mini",
    judge_provider: str = "openai",
    judge_sample_rate: float = 1.0,
    from_file: str | None = None,
    force: bool = False,
    num_processes: int = 1,
    seed: int = 100,
) -> str:
    if from_file:
        import os, shutil
        os.makedirs(out_dir, exist_ok=True)
        for name in ("instrumented_summary.json", "instrumented_logs.jsonl", "judge_calibration.json", "stage2a_manifest.json"):
            src = os.path.join(from_file, name)
            if os.path.exists(src):
                shutil.copyfile(src, os.path.join(out_dir, name))
        return out_dir
    stage2a_run.run(
        memory_path=memory_path,
        out_dir=out_dir,
        tasks=tasks,
        sample_size=sample_size,
        task_ids_csv=task_ids_csv,
        force=force,
        num_processes=num_processes,
    )
    stage2b_analyze.run(
        memory_path=memory_path,
        stage2a_dir=out_dir,
        out_dir=out_dir,
        reference_detection=reference_detection,
        judge_model=judge_model,
        judge_provider=judge_provider,
        judge_sample_rate=judge_sample_rate,
        seed=seed,
        force=force,
    )
    return out_dir


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Stage 2 wrapper: run 2a (agent) then 2b (analyze)")
    p.add_argument("--memory", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--tasks", default="dev")
    p.add_argument("--sample-size", type=int, default=None)
    p.add_argument("--task-ids", default=None)
    p.add_argument("--reference-detection", choices=["citation", "judge", "both"], default="both")
    p.add_argument("--judge-model", default="gpt-4o-mini")
    p.add_argument("--judge-provider", default="openai")
    p.add_argument("--judge-sample-rate", type=float, default=1.0)
    p.add_argument("--from-file", default=None)
    p.add_argument("--num-processes", type=int, default=1)
    p.add_argument("--seed", type=int, default=100)
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)
    run(
        memory_path=args.memory,
        out_dir=args.out_dir,
        tasks=args.tasks,
        sample_size=args.sample_size,
        task_ids_csv=args.task_ids,
        reference_detection=args.reference_detection,
        judge_model=args.judge_model,
        judge_provider=args.judge_provider,
        judge_sample_rate=args.judge_sample_rate,
        from_file=args.from_file,
        force=args.force,
        num_processes=args.num_processes,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
