"""Stage 3 — Insight Statistics.

Pure computation (no LLM or agent calls). Takes Stage 1 baseline + Stage 2
instrumented outputs and computes per-insight:
  - reference count (total & per-task)
  - reference_tasks (list of task IDs where the insight was referenced)
  - lift_per_task (in {-1, 0, +1}) against baseline
  - lift_mean, lift_variance
  - has_positive_spike (any task where lift_per_task == +1)

Reference source is one of:
  - citation       : agent's own [cited_insights: ...] annotations
  - judge          : post-hoc judge calls
  - union          : either
  - intersection   : both (conservative)
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
from collections import defaultdict
from datetime import datetime
from typing import Any

from .common.config_hash import hash_config, stage_cache_ok
from .common.io import read_json, read_jsonl, write_json
from .common.progress import stage_context


STAGE_NAME = "stage3_stats"


def _collect_references(
    logs_path: str,
    reference_source: str,
    min_references_per_task: int,
) -> dict[str, dict[str, int]]:
    """Build {insight_id: {task_id: step_count_where_referenced}}.

    A task counts toward `reference_tasks[j]` only if
    `step_count_where_referenced[task_id] >= min_references_per_task`.
    """
    # insight_id -> task_id -> count of steps where it was referenced
    per_insight: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for record in read_jsonl(logs_path):
        task_id = record["task_id"]
        citation_ids = set(record.get("cited_insights") or [])
        judge_ids = set(record.get("judge_referenced_insights") or [])
        if reference_source == "citation":
            ids = citation_ids
        elif reference_source == "judge":
            ids = judge_ids
        elif reference_source == "union":
            ids = citation_ids | judge_ids
        elif reference_source == "intersection":
            ids = citation_ids & judge_ids
        else:
            raise ValueError(f"Unknown reference_source: {reference_source}")
        for ins_id in ids:
            per_insight[ins_id][task_id] += 1
    # Filter per min_references_per_task
    filtered: dict[str, dict[str, int]] = {}
    for ins_id, per_task in per_insight.items():
        kept = {tid: n for tid, n in per_task.items() if n >= min_references_per_task}
        if kept:
            filtered[ins_id] = kept
    return filtered


def run(
    baseline_path: str,
    stage2_dir: str,
    memory_path: str,
    out_path: str,
    reference_source: str = "union",
    min_references_per_task: int = 1,
    force: bool = False,
) -> str:
    run_dir = os.path.dirname(os.path.dirname(os.path.abspath(out_path)))
    cfg = {
        "baseline_path": os.path.abspath(baseline_path),
        "stage2_dir": os.path.abspath(stage2_dir),
        "memory_path": os.path.abspath(memory_path),
        "reference_source": reference_source,
        "min_references_per_task": min_references_per_task,
    }
    cfg_hash = hash_config(
        cfg,
        input_file_paths=[
            baseline_path,
            os.path.join(stage2_dir, "instrumented_summary.json"),
            os.path.join(stage2_dir, "instrumented_logs.jsonl"),
            memory_path,
        ],
    )
    if not force and stage_cache_ok(out_path, cfg_hash):
        print(f"[{STAGE_NAME}] cache hit: {out_path}")
        return out_path

    with stage_context(STAGE_NAME, run_dir) as ctx:
        baseline = read_json(baseline_path)
        instr_summary = read_json(os.path.join(stage2_dir, "instrumented_summary.json"))
        logs_path = os.path.join(stage2_dir, "instrumented_logs.jsonl")
        memory = read_json(memory_path)

        base_per_task = baseline["per_task"]
        instr_per_task = instr_summary["per_task"]

        # Common tasks = intersection of tasks with output in both stages.
        common_task_ids = sorted(set(base_per_task.keys()) & set(instr_per_task.keys()))
        print(f"[{STAGE_NAME}] common tasks: {len(common_task_ids)}")

        # Per-task binary lift (original)
        lift_per_task: dict[str, int] = {}
        # Per-task fractional lift = score_full - score_baseline, score = passed / total
        lift_frac_per_task: dict[str, float] = {}
        def _score(d: dict[str, Any]) -> float | None:
            p = int(d.get("num_passed", 0) or 0)
            f = int(d.get("num_failed", 0) or 0)
            tot = p + f
            return (p / tot) if tot > 0 else None
        for tid in common_task_ids:
            b = int(base_per_task[tid].get("success", 0))
            f = int(instr_per_task[tid].get("success", 0))
            lift_per_task[tid] = f - b  # in {-1, 0, +1}
            sb = _score(base_per_task[tid])
            sf = _score(instr_per_task[tid])
            if sb is not None and sf is not None:
                lift_frac_per_task[tid] = sf - sb

        references = _collect_references(logs_path, reference_source, min_references_per_task)
        print(f"[{STAGE_NAME}] insights with ≥1 reference: {len(references)}")

        per_insight: dict[str, Any] = {}
        for ins in memory["insights"]:
            ins_id = ins["id"]
            per_task_refs = references.get(ins_id, {})
            ref_task_ids = [tid for tid in per_task_refs.keys() if tid in lift_per_task]
            ref_task_ids.sort()
            lifts = [lift_per_task[tid] for tid in ref_task_ids]
            frac_lifts = [lift_frac_per_task[tid] for tid in ref_task_ids if tid in lift_frac_per_task]
            if lifts:
                lift_mean = sum(lifts) / len(lifts)
                lift_var = statistics.pvariance(lifts) if len(lifts) >= 1 else 0.0
                has_positive_spike = any(x == 1 for x in lifts)
            else:
                lift_mean = 0.0
                lift_var = 0.0
                has_positive_spike = False
            if frac_lifts:
                lift_frac_mean = sum(frac_lifts) / len(frac_lifts)
                lift_frac_var = statistics.pvariance(frac_lifts) if len(frac_lifts) >= 1 else 0.0
                has_fractional_positive_spike = any(x > 0 for x in frac_lifts)
            else:
                lift_frac_mean = 0.0
                lift_frac_var = 0.0
                has_fractional_positive_spike = False
            per_insight[ins_id] = {
                "source": ins.get("source", "unknown"),
                "domain": ins.get("domain", "unknown"),
                "section": ins.get("section", "OTHERS"),
                "token_length": ins.get("token_length", 0),
                "reference_count_total": sum(per_task_refs.values()),
                "reference_count_by_task": dict(per_task_refs),
                "reference_tasks": ref_task_ids,
                "lift_per_task": {tid: lift_per_task[tid] for tid in ref_task_ids},
                "lift_mean": lift_mean,
                "lift_variance": lift_var,
                "has_positive_spike": has_positive_spike,
                "lift_fractional_per_task": {tid: lift_frac_per_task[tid] for tid in ref_task_ids if tid in lift_frac_per_task},
                "lift_fractional_mean": lift_frac_mean,
                "lift_fractional_variance": lift_frac_var,
                "has_fractional_positive_spike": has_fractional_positive_spike,
            }

        # Global stats — binary
        n_dummies = sum(1 for ins in memory["insights"] if ins.get("source") == "dummy")
        all_lifts = list(lift_per_task.values())
        if all_lifts:
            lift_mean_global = sum(all_lifts) / len(all_lifts)
            lift_std_global = statistics.pstdev(all_lifts)
            sorted_l = sorted(all_lifts)
            p25 = sorted_l[max(0, int(0.25 * (len(sorted_l) - 1)))]
            p75 = sorted_l[max(0, int(0.75 * (len(sorted_l) - 1)))]
        else:
            lift_mean_global = lift_std_global = p25 = p75 = 0.0
        # Global stats — fractional
        all_frac = list(lift_frac_per_task.values())
        if all_frac:
            frac_mean_global = sum(all_frac) / len(all_frac)
            frac_std_global = statistics.pstdev(all_frac)
            sorted_f = sorted(all_frac)
            fp25 = sorted_f[max(0, int(0.25 * (len(sorted_f) - 1)))]
            fp75 = sorted_f[max(0, int(0.75 * (len(sorted_f) - 1)))]
        else:
            frac_mean_global = frac_std_global = fp25 = fp75 = 0.0
        # Requirement-level rollup (fine-grained progress)
        base_req_passed = sum(int(base_per_task[tid].get("num_passed", 0) or 0) for tid in common_task_ids)
        base_req_total = sum(int((base_per_task[tid].get("num_passed", 0) or 0) + (base_per_task[tid].get("num_failed", 0) or 0)) for tid in common_task_ids)
        full_req_passed = sum(int(instr_per_task[tid].get("num_passed", 0) or 0) for tid in common_task_ids)
        full_req_total = sum(int((instr_per_task[tid].get("num_passed", 0) or 0) + (instr_per_task[tid].get("num_failed", 0) or 0)) for tid in common_task_ids)

        out = {
            "per_insight": per_insight,
            "global": {
                "reference_source_used": reference_source,
                "min_references_per_task": min_references_per_task,
                "n_tasks_common": len(common_task_ids),
                "n_insights": len(memory["insights"]),
                "n_dummies": n_dummies,
                "baseline_success": sum(int(base_per_task[tid].get("success", 0)) for tid in common_task_ids),
                "fullmem_success": sum(int(instr_per_task[tid].get("success", 0)) for tid in common_task_ids),
                "baseline_req_passed_total": base_req_passed,
                "baseline_req_total": base_req_total,
                "fullmem_req_passed_total": full_req_passed,
                "fullmem_req_total": full_req_total,
                "lift_distribution_summary": {
                    "mean": lift_mean_global,
                    "std": lift_std_global,
                    "p25": p25,
                    "p75": p75,
                },
                "lift_fractional_distribution_summary": {
                    "mean": frac_mean_global,
                    "std": frac_std_global,
                    "p25": fp25,
                    "p75": fp75,
                    "n_tasks_with_score": len(all_frac),
                },
            },
            "config": cfg,
            "config_hash": cfg_hash,
            "created_at": datetime.utcnow().isoformat() + "Z",
        }
        write_json(out_path, out)
        print(f"[{STAGE_NAME}] wrote {out_path}")
    return out_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Stage 3: Insight statistics (pure computation)")
    p.add_argument("--baseline", required=True)
    p.add_argument("--instrumented", required=True, help="Stage 2 directory")
    p.add_argument("--memory", required=True, help="memory.json from Stage 0")
    p.add_argument("--out", required=True)
    p.add_argument("--reference-source", choices=["citation", "judge", "union", "intersection"], default="union")
    p.add_argument("--min-references-per-task", type=int, default=1)
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)
    run(
        baseline_path=args.baseline,
        stage2_dir=args.instrumented,
        memory_path=args.memory,
        out_path=args.out,
        reference_source=args.reference_source,
        min_references_per_task=args.min_references_per_task,
        force=args.force,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
