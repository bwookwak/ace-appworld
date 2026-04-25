"""Stage 5 — Bucket Verification (SKELETON).

Run targeted ablation only on buckets where Stage 3's proxy signal was
ambiguous (typically Bucket C, optionally Bucket D for LOO/group
ablation). Produces per-insight labels used by Stage 6 to assemble the
final GT memory.

CUSTOMIZE:
  - per-bucket seed count (default 1 everywhere, matching project-wide
    single-seed choice; raise to 3 or 5 for C and D once v1 is validated).
  - Bucket A sample-size formula: `min(20, ceil(0.2 * |A|))`.
  - whether to run Bucket D group ablation AND individual LOO, or only
    one of those.
  - whether to add filler-matched (context-length-controlled) ablation
    alongside simple deletion — the spec's v1 caveat is that deletion
    changes prompt length.
  - which buckets to skip entirely (e.g. skip A if A is verified by the
    dummies' being all-A; skip B if threshold was conservative).
"""
from __future__ import annotations

import argparse
import os
import sys

from .common.config_hash import hash_config, stage_cache_ok
from .common.io import read_json, write_json
from .common.progress import stage_context

STAGE_NAME = "stage5_verify"


def run(
    memory_path: str,
    buckets_path: str,
    out_path: str,
    per_bucket_seeds: dict[str, int] | None = None,
    force: bool = False,
) -> str:
    run_dir = os.path.dirname(os.path.dirname(os.path.abspath(out_path)))
    seed_cfg = per_bucket_seeds or {"A": 1, "B": 1, "C": 1, "D": 1, "E": 1}
    cfg = {
        "memory_path": os.path.abspath(memory_path),
        "buckets_path": os.path.abspath(buckets_path),
        "per_bucket_seeds": seed_cfg,
    }
    cfg_hash = hash_config(cfg, input_file_paths=[memory_path, buckets_path])
    if not force and stage_cache_ok(out_path, cfg_hash):
        print(f"[{STAGE_NAME}] cache hit: {out_path}")
        return out_path

    with stage_context(STAGE_NAME, run_dir):
        memory = read_json(memory_path)
        buckets = read_json(buckets_path)
        print(f"[{STAGE_NAME}] loaded {len(memory.get('insights', []))} insights, "
              f"buckets: {list(buckets.get('buckets', {}).keys())}")

        # TODO(stage-5): for each targeted insight, generate an ablated
        # playbook, run appworld on the ablation subset (using the same
        # conda env + jsonnet-generation pattern as Stage 1/2), compare
        # lift vs full memory, and write per-insight GT_tier1/GT_tier2/
        # rare_critical/noise labels.

        out = {
            "per_insight_label": {},
            "bucket_verifications": {},
            "config": cfg,
            "config_hash": cfg_hash,
        }
        write_json(out_path, out)
        print(f"[{STAGE_NAME}] skeleton output: {out_path}")
        print(f"[{STAGE_NAME}] Stage 5 skeleton — logic not yet implemented. See CUSTOMIZE comment for pending decisions.")
    return out_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Stage 5: Bucket verification (skeleton)")
    p.add_argument("--memory", required=True)
    p.add_argument("--buckets", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)
    run(memory_path=args.memory, buckets_path=args.buckets, out_path=args.out, force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
