"""Stable hashing of stage inputs + config for the caching layer.

Each stage computes its own hash from the resolved inputs (paths + hashes of
their contents) plus the stage's CLI config. If the hash matches an existing
output's recorded hash, the stage is skipped unless `--force` is passed.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any


def hash_file(path: str) -> str:
    """SHA-256 of a file's bytes, hex-encoded."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_value(value: Any) -> str:
    """SHA-256 of a JSON-canonical representation of `value`."""
    payload = json.dumps(value, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def hash_config(
    config: dict[str, Any],
    input_file_paths: list[str] | None = None,
) -> str:
    """Hash a stage's config + the contents of each input file path."""
    materialized = {
        "config": config,
        "inputs": {
            p: (hash_file(p) if os.path.exists(p) else "MISSING")
            for p in sorted(input_file_paths or [])
        },
    }
    return hash_value(materialized)


def stage_cache_ok(out_path: str, expected_hash: str) -> bool:
    """True if `out_path` exists, is JSON, and has matching `config_hash`."""
    if not os.path.exists(out_path):
        return False
    try:
        with open(out_path, "r", encoding="utf-8") as f:
            blob = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    return blob.get("config_hash") == expected_hash
