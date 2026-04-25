"""Stage 7 — Report Generation (SKELETON).

Aggregate all stage outputs under a run directory into a human-readable
report (markdown by default).

CUSTOMIZE:
  - report format: markdown / HTML / Jupyter notebook
  - which plots to include (lift histogram, per-bucket counts, per-task
    success heatmap, citation vs judge agreement scatter)
  - per-task table columns: baseline_success, fullmem_success, lift,
    cited_insights, judged_insights, step_count, cost
  - whether to embed dummy_generation_log snippets, judge_raw_response
    snippets, or just link to them
  - page grouping: per-stage sections vs per-insight sections
"""
from __future__ import annotations

import argparse
import os
import sys

from .common.io import read_json, write_text
from .common.progress import stage_context

STAGE_NAME = "stage7_report"


def run(run_dir: str, out_path: str, force: bool = False) -> str:
    with stage_context(STAGE_NAME, run_dir):
        # TODO(stage-7): walk the run_dir for each stage's output and
        # assemble a markdown report. Keep raw data links for the user to
        # dig into per-task trajectories.

        memory_path = os.path.join(run_dir, "memory.json")
        baseline_path = os.path.join(run_dir, "stage1", "baseline_results.json")
        stage2_summary = os.path.join(run_dir, "stage2", "instrumented_summary.json")
        stats_path = os.path.join(run_dir, "stage3", "insight_stats.json")

        found_stages = []
        for p, name in [
            (memory_path, "Stage 0 memory"),
            (baseline_path, "Stage 1 baseline"),
            (stage2_summary, "Stage 2 instrumented"),
            (stats_path, "Stage 3 stats"),
        ]:
            if os.path.exists(p):
                found_stages.append((name, p))

        lines = []
        lines.append(f"# memory_utility report — run `{os.path.basename(run_dir.rstrip('/'))}`")
        lines.append("")
        lines.append("## Stages found")
        for name, p in found_stages:
            lines.append(f"- **{name}** → `{p}`")
        lines.append("")
        lines.append(f"*Stage 7 skeleton — full report logic not yet implemented. See CUSTOMIZE comment for pending decisions.*")
        write_text(out_path, "\n".join(lines) + "\n")
        print(f"[{STAGE_NAME}] skeleton output: {out_path}")
        print(f"[{STAGE_NAME}] Stage 7 skeleton — logic not yet implemented. See CUSTOMIZE comment for pending decisions.")
    return out_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Stage 7: Report generation (skeleton)")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)
    run(run_dir=args.run_dir, out_path=args.out, force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
