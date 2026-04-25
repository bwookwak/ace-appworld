"""Stage 2b — Trajectory Analysis (JUDGE + CITATION EXTRACTION).

Reads the per-task trajectories produced by Stage 2a, extracts citation
signals, optionally calls the judge, and writes instrumented_logs.jsonl,
instrumented_summary.json, and judge_calibration.json.

This stage does NOT run the agent. If a task is missing agent output,
it's skipped and marked "no_output" in the summary. Rerunning Stage 2a
to fill in the gaps is cheap if appworld output caching works; otherwise
you can just accept partial results and let Stage 3 work on the
intersection.

Outputs under `<out_dir>/`:
  - instrumented_logs.jsonl      (one record per step, per task)
  - instrumented_summary.json    (per-task rollup + config_hash)
  - judge_calibration.json       (citation vs judge agreement)
  - trajectories/<tid>.jsonl     (copy of each task's lm_calls.jsonl)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import sys
from datetime import datetime
from typing import Any

from .common import paths
from .common.appworld_runner import (
    evaluate_one_task,
    read_task_cost,
    read_trajectory,
    task_has_output,
)
from .common.config_hash import hash_config, stage_cache_ok
from .common.io import append_jsonl, read_json, write_json
from .common.litellm_client import JudgeClient, ensure_openai_env
from .common.playbook import parse_citations
from .common.progress import stage_context


STAGE_NAME = "stage2b_analyze"


JUDGE_SYSTEM = (
    "You are evaluating whether an AI assistant actually used specific "
    "playbook bullets when producing its latest action. Count an insight "
    "as USED only if the action's logic was shaped by the insight's "
    "specific advice, not just incidental topical overlap."
)


JUDGE_EXAMPLES = """\
Examples of meaningful use:
 - The insight says "always look up API docs before calling", and the action
   starts with apis.api_docs.show_api_doc(...). → USED.
 - The insight says "for pagination use while True over page_index", and the
   action implements such a loop. → USED.
 - The insight says "prefer library APIs over per-item calls for duration",
   and the action calls show_song_library() instead of per-song show_song().
   → USED.

Examples of NOT meaningful use:
 - The insight mentions Spotify but the action queries Phone. The topic
   overlap is coincidental. → NOT used.
 - The insight is a general hint to "complete the task". Used in ALL
   tasks trivially → NOT meaningful, do NOT cite.
 - The action re-implements basic Python that the insight happens to
   describe. The agent did not consult the insight; the code would be the
   same regardless. → NOT used.
"""


JUDGE_USER_TEMPLATE = """\
Below is a list of playbook insights (each with an ID in brackets), and the
agent's latest action (one step of a task). Identify which insights were
MEANINGFULLY used when producing the action, using the examples above as
your rubric.

Insights currently in memory:
{insight_block}

Agent's latest step:
BEGIN_STEP
{action_text}
END_STEP

Respond with a single JSON object on one line:
{{"used": ["id1", "id2"], "reasoning": "one short sentence"}}
If none of the insights were meaningfully used, return {{"used": [], "reasoning": "..."}}.
"""


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_judge_response(text: str) -> dict[str, Any]:
    if not text:
        return {"used": [], "reasoning": "", "parse_error": "empty"}
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    m = _JSON_RE.search(text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return {"used": [], "reasoning": text[:200], "parse_error": "no_json"}


def _insight_block(memory_insights: list[dict[str, Any]]) -> str:
    return "\n".join(f"[{ins['id']}] {ins['text']}" for ins in memory_insights)


def run(
    memory_path: str,
    stage2a_dir: str,
    out_dir: str,
    reference_detection: str = "both",
    judge_model: str = "gpt-4o-mini",
    judge_provider: str = "openai",
    judge_sample_rate: float = 1.0,
    force: bool = False,
    seed: int = 100,
) -> str:
    run_dir = os.path.dirname(os.path.abspath(out_dir).rstrip("/"))
    summary_path = os.path.join(out_dir, "instrumented_summary.json")

    memory_blob = read_json(memory_path)
    memory_insights = list(memory_blob["insights"])

    # Stage 2a manifest tells us which tasks have agent output.
    manifest_path = os.path.join(stage2a_dir, "stage2a_manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"stage2a_manifest.json not found at {manifest_path}. Run Stage 2a first.")
    manifest = read_json(manifest_path)
    experiment_name = manifest["experiment_name"]
    task_ids = list(manifest["task_ids_requested"])
    known_ids = {ins["id"] for ins in memory_insights}

    cfg = {
        "memory_path": os.path.abspath(memory_path),
        "stage2a_dir": os.path.abspath(stage2a_dir),
        "reference_detection": reference_detection,
        "judge_model": judge_model,
        "judge_provider": judge_provider,
        "judge_sample_rate": judge_sample_rate,
        "experiment_name": experiment_name,
        "seed": seed,
        "task_ids": task_ids,
    }
    cfg_hash = hash_config(cfg, input_file_paths=[memory_path, manifest_path])
    if not force and stage_cache_ok(summary_path, cfg_hash):
        print(f"[{STAGE_NAME}] cache hit: {summary_path}")
        return out_dir

    os.makedirs(out_dir, exist_ok=True)
    traj_dir = os.path.join(out_dir, "trajectories")
    os.makedirs(traj_dir, exist_ok=True)
    logs_path = os.path.join(out_dir, "instrumented_logs.jsonl")
    if os.path.exists(logs_path):
        os.remove(logs_path)

    with stage_context(STAGE_NAME, run_dir) as ctx:
        use_citation = reference_detection in ("citation", "both")
        use_judge = reference_detection in ("judge", "both")
        judge: JudgeClient | None = None
        if use_judge:
            ensure_openai_env(provider=judge_provider)
            judge = JudgeClient(
                model=judge_model, provider=judge_provider, use_cache=True,
                cost_accumulator=ctx.cost,
            )
            insight_block = _insight_block(memory_insights)
            judge_system = JUDGE_SYSTEM + "\n\n" + JUDGE_EXAMPLES
            print(f"[{STAGE_NAME}] judge: {judge_model} sample_rate={judge_sample_rate}")

        rng = random.Random(seed)
        per_task_summary: dict[str, Any] = {}
        calib_rows: list[dict[str, Any]] = []

        for tid in ctx.iter(task_ids, total=len(task_ids), desc="analyze"):
            if not task_has_output(experiment_name, tid):
                per_task_summary[tid] = {
                    "success": 0, "step_count": 0, "cost_usd": 0.0,
                    "cited_count": 0, "judged_count": 0,
                    "cited_insights": [], "judged_insights": [],
                    "num_passed": -1, "num_failed": -1,
                    "note": "no_output",
                }
                continue

            traj = read_trajectory(experiment_name, tid)
            step_count = len(traj)
            src_traj = os.path.join(paths.task_output_dir(experiment_name, tid), "logs", "lm_calls.jsonl")
            if os.path.exists(src_traj):
                try:
                    shutil.copyfile(src_traj, os.path.join(traj_dir, f"{tid}.jsonl"))
                except OSError:
                    pass

            try:
                ev = evaluate_one_task(tid, experiment_name)
                success = 1 if ev["success"] else 0
                num_passed = ev["num_passed"]
                num_failed = ev["num_failed"]
            except Exception as exc:
                print(f"[{STAGE_NAME}] evaluate_task({tid}) failed: {exc}")
                success = 0
                num_passed = -1
                num_failed = -1

            task_cost = read_task_cost(experiment_name, tid)
            task_cited_ids: set[str] = set()
            task_judged_ids: set[str] = set()

            for step in traj:
                step_idx = step["step_idx"]
                content = step["content"] or ""

                cite_parse = parse_citations(content)
                cited = [c for c in (cite_parse["structured"] + cite_parse["inline"]) if c in known_ids]
                cited = list(dict.fromkeys(cited))
                task_cited_ids.update(cited)

                judge_used: list[str] = []
                judge_raw: str | None = None
                judge_parsed: dict[str, Any] | None = None
                if use_judge and judge is not None:
                    if judge_sample_rate >= 1.0 or rng.random() < judge_sample_rate:
                        prompt = JUDGE_USER_TEMPLATE.format(
                            insight_block=insight_block,
                            action_text=content[:8000],
                        )
                        try:
                            judge_raw, _ = judge.generate(prompt, system=judge_system)
                            judge_parsed = _parse_judge_response(judge_raw)
                            raw_used = judge_parsed.get("used", [])
                            judge_used = [u for u in raw_used if u in known_ids]
                            task_judged_ids.update(judge_used)
                        except Exception as exc:
                            print(f"[{STAGE_NAME}] judge call failed ({tid} step {step_idx}): {exc}")

                record = {
                    "task_id": tid,
                    "step_idx": step_idx,
                    "insights_in_context": [ins["id"] for ins in memory_insights],
                    "cited_insights": cited if use_citation else [],
                    "citation_parse_raw": cite_parse,
                    "judge_referenced_insights": judge_used,
                    "judge_raw_response": judge_raw,
                    "judge_parsed": judge_parsed,
                    "action_preview": content[:400],
                }
                append_jsonl(logs_path, record)

                if use_citation and use_judge:
                    cset = set(cited)
                    jset = set(judge_used)
                    union = cset | jset
                    inter = cset & jset
                    calib_rows.append({
                        "task_id": tid,
                        "step_idx": step_idx,
                        "citation_ids": sorted(cset),
                        "judge_ids": sorted(jset),
                        "agreement_jaccard": (len(inter) / len(union)) if union else 1.0,
                    })

            per_task_summary[tid] = {
                "success": success,
                "step_count": step_count,
                "cost_usd": task_cost,
                "cited_count": len(task_cited_ids),
                "judged_count": len(task_judged_ids),
                "cited_insights": sorted(task_cited_ids),
                "judged_insights": sorted(task_judged_ids),
                "num_passed": num_passed,
                "num_failed": num_failed,
            }

        n_tasks = len(task_ids)
        n_ran = sum(1 for v in per_task_summary.values() if v.get("note") != "no_output")
        n_success = sum(v["success"] for v in per_task_summary.values())
        total_steps = sum(v["step_count"] for v in per_task_summary.values())
        total_cost = sum(v["cost_usd"] for v in per_task_summary.values())

        summary = {
            "per_task": per_task_summary,
            "aggregate": {
                "n_tasks": n_tasks,
                "n_ran": n_ran,
                "n_success": n_success,
                "success_rate": (n_success / n_ran) if n_ran else 0.0,
                "total_steps": total_steps,
                "mean_steps": (total_steps / n_ran) if n_ran else 0.0,
                "total_cost_usd": round(total_cost, 4),
            },
            "config": cfg,
            "config_hash": cfg_hash,
            "experiment_name": experiment_name,
            "created_at": datetime.utcnow().isoformat() + "Z",
        }
        write_json(summary_path, summary)
        print(f"[{STAGE_NAME}] wrote {summary_path}  (success={n_success}/{n_ran})")

        if use_citation and use_judge and calib_rows:
            total_union = sum(len(set(r["citation_ids"]) | set(r["judge_ids"])) for r in calib_rows)
            total_inter = sum(len(set(r["citation_ids"]) & set(r["judge_ids"])) for r in calib_rows)
            calib = {
                "per_step": calib_rows,
                "global": {
                    "steps_evaluated": len(calib_rows),
                    "jaccard_mean": sum(r["agreement_jaccard"] for r in calib_rows) / len(calib_rows),
                    "global_jaccard": (total_inter / total_union) if total_union else 1.0,
                },
            }
        else:
            calib = {"per_step": [], "global": {"note": "calibration requires reference_detection=both"}}
        write_json(os.path.join(out_dir, "judge_calibration.json"), calib)

    return out_dir


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Stage 2b: Trajectory analysis (no agent calls)")
    p.add_argument("--memory", required=True)
    p.add_argument("--stage2a-dir", required=True, help="Stage 2a output directory (same as --out-dir for stage2b)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--reference-detection", choices=["citation", "judge", "both"], default="both")
    p.add_argument("--judge-model", default="gpt-4o-mini")
    p.add_argument("--judge-provider", default="openai")
    p.add_argument("--judge-sample-rate", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=100)
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)
    run(
        memory_path=args.memory,
        stage2a_dir=args.stage2a_dir,
        out_dir=args.out_dir,
        reference_detection=args.reference_detection,
        judge_model=args.judge_model,
        judge_provider=args.judge_provider,
        judge_sample_rate=args.judge_sample_rate,
        seed=args.seed,
        force=args.force,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
