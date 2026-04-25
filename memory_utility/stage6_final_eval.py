"""Stage 6 — Final GT Memory Build + Evaluation (SKELETON).

Build a curated "ground-truth" memory from Stage 5 labels, render it as a
playbook, and run the agent once more under that curated memory. Optionally
re-run under full memory (the Stage 2 playbook) head-to-head.

CUSTOMIZE:
  - which labels to include as GT — default: GT_tier1, GT_tier2, rare_critical
  - whether to re-run full-memory for head-to-head comparison (recommended)
  - whether to also re-run baseline (typically no, as Stage 1 is canonical)
  - task subset: typically the same subset as Stage 1/2, but could be a
    disjoint "held-out" set for generalization check.
"""
from __future__ import annotations

import argparse
import os
import sys

from .common.config_hash import hash_config, stage_cache_ok
from .common.io import read_json, write_json
from .common.progress import stage_context

STAGE_NAME = "stage6_final_eval"


def run(
    memory_path: str,
    verified_path: str,
    out_dir: str,
    include_labels: tuple[str, ...] = ("GT_tier1", "GT_tier2", "rare_critical"),
    also_run_full_memory: bool = True,
    force: bool = False,
) -> str:
    run_dir = os.path.dirname(os.path.abspath(out_dir).rstrip("/"))
    cfg = {
        "memory_path": os.path.abspath(memory_path),
        "verified_path": os.path.abspath(verified_path),
        "include_labels": list(include_labels),
        "also_run_full_memory": also_run_full_memory,
    }
    cfg_hash = hash_config(cfg, input_file_paths=[memory_path, verified_path])
    summary_path = os.path.join(out_dir, "final_eval_summary.json")
    if not force and stage_cache_ok(summary_path, cfg_hash):
        print(f"[{STAGE_NAME}] cache hit: {summary_path}")
        return out_dir

    os.makedirs(out_dir, exist_ok=True)
    with stage_context(STAGE_NAME, run_dir):
        memory = read_json(memory_path)
        verified = read_json(verified_path)
        print(f"[{STAGE_NAME}] memory: {len(memory.get('insights', []))} insights; "
              f"verified labels: {len(verified.get('per_insight_label', {}))}")

        # TODO(stage-6): filter memory by Stage 5 labels → render as
        # playbook → generate jsonnet → `appworld run` → evaluate. If
        # `also_run_full_memory`, run again under full memory for head-to-
        # head comparison and record both per-task results.

        out = {
            "gt_memory_insight_ids": [],
            "per_task": {},
            "aggregate": {},
            "also_run_full_memory": also_run_full_memory,
            "config": cfg,
            "config_hash": cfg_hash,
        }
        write_json(summary_path, out)
        print(f"[{STAGE_NAME}] skeleton output: {summary_path}")
        print(f"[{STAGE_NAME}] Stage 6 skeleton — logic not yet implemented. See CUSTOMIZE comment for pending decisions.")
    return out_dir


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Stage 6: Final GT memory build + evaluation (skeleton)")
    p.add_argument("--memory", required=True)
    p.add_argument("--verified", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--include-labels", default="GT_tier1,GT_tier2,rare_critical")
    p.add_argument("--no-fullmem-head-to-head", action="store_true")
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)
    labels = tuple(l.strip() for l in args.include_labels.split(",") if l.strip())
    run(
        memory_path=args.memory,
        verified_path=args.verified,
        out_dir=args.out_dir,
        include_labels=labels,
        also_run_full_memory=not args.no_fullmem_head_to_head,
        force=args.force,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
