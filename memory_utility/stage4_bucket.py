"""Stage 4 — Bucketing (SKELETON).

Partition insights into buckets based on (reference count) × (lift sign /
magnitude) × (lift variance) and the dummy-derived noise threshold. Stage 5
then runs targeted verification only on ambiguous buckets.

CUSTOMIZE:
  - dummy-based θ_lift calibration: should θ_lift be the Nth percentile of
    dummy lift_means, or dummy_mean + k*dummy_std? Current design says
    percentile but can flip; see `--theta-method`.
  - ref-count split percentile (Bucket A vs others): default p50 across
    non-dummy insights; could be by absolute count (e.g. ≥5).
  - variance-split percentile for spiky detection: default p80.
  - handling of zero-reference insights: currently dumped into their own
    bucket; could be merged into Bucket A ("noise/unused") depending on
    interpretation.
  - whether to treat `has_positive_spike` as a promotion signal (move into
    "rare_critical") regardless of mean.

Expected outputs (runs/<RUN>/stage4/buckets.json):
  {
    "per_insight": { "id": {"bucket": "A"|"B"|"C"|"D"|"E", "score": ...} },
    "buckets": {
      "A_noise":       [...],  # dummy-like / below noise floor
      "B_useful":      [...],  # high refs + lift above noise
      "C_ambiguous":   [...],  # mid refs or mid lift
      "D_harmful":     [...],  # lift significantly negative
      "E_rare_critical": [...],# low refs but positive spike
      "zero_ref":      [...]   # never referenced
    },
    "thresholds": {
      "theta_lift": ...,
      "ref_count_split": ...,
      "variance_split": ...,
      "method": "..."
    },
    "config_hash": "..."
  }
"""
from __future__ import annotations

import argparse
import os
import sys

from .common.config_hash import hash_config, stage_cache_ok
from .common.io import read_json, write_json
from .common.progress import stage_context

STAGE_NAME = "stage4_bucket"


def run(
    stats_path: str,
    out_path: str,
    theta_method: str = "dummy_p95",
    ref_count_split: str = "p50",
    variance_split: str = "p80",
    force: bool = False,
) -> str:
    run_dir = os.path.dirname(os.path.dirname(os.path.abspath(out_path)))
    cfg = {
        "stats_path": os.path.abspath(stats_path),
        "theta_method": theta_method,
        "ref_count_split": ref_count_split,
        "variance_split": variance_split,
    }
    cfg_hash = hash_config(cfg, input_file_paths=[stats_path])
    if not force and stage_cache_ok(out_path, cfg_hash):
        print(f"[{STAGE_NAME}] cache hit: {out_path}")
        return out_path

    with stage_context(STAGE_NAME, run_dir):
        stats = read_json(stats_path)
        print(f"[{STAGE_NAME}] loaded stats with {len(stats.get('per_insight', {}))} insights")

        # TODO(stage-4): compute thresholds from dummy distribution, assign
        # each insight to a bucket, and write out the bucket file with the
        # schema in the module docstring.

        out = {
            "per_insight": {},
            "buckets": {
                "A_noise": [],
                "B_useful": [],
                "C_ambiguous": [],
                "D_harmful": [],
                "E_rare_critical": [],
                "zero_ref": [],
            },
            "thresholds": {
                "theta_lift": None,
                "ref_count_split": None,
                "variance_split": None,
                "method": theta_method,
            },
            "config": cfg,
            "config_hash": cfg_hash,
        }
        write_json(out_path, out)
        print(f"[{STAGE_NAME}] skeleton output: {out_path}")
        print(f"[{STAGE_NAME}] Stage 4 skeleton — logic not yet implemented. See CUSTOMIZE comment for pending decisions.")
    return out_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Stage 4: Bucketing (skeleton)")
    p.add_argument("--stats", required=True, help="insight_stats.json from Stage 3")
    p.add_argument("--out", required=True)
    p.add_argument("--theta-method", default="dummy_p95")
    p.add_argument("--ref-count-split", default="p50")
    p.add_argument("--variance-split", default="p80")
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)
    run(
        stats_path=args.stats,
        out_path=args.out,
        theta_method=args.theta_method,
        ref_count_split=args.ref_count_split,
        variance_split=args.variance_split,
        force=args.force,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
