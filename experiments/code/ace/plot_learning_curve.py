"""Render the learning curve from a results.jsonl produced by CheckpointWatcher."""
from __future__ import annotations

import argparse
import json
import os
from typing import Any

from appworld.common.utils import maybe_create_parent_directory


def _load_records(results_jsonl: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(results_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def plot_learning_curve(
    results_jsonl: str,
    output_path: str,
    title: str = "ACE Learning Curve",
    xlabel: str = "Number of Adaptation Episodes",
    ylabel: str = "Task Goal Completion (%)",
) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records = _load_records(results_jsonl)
    valid = [
        r for r in records
        if isinstance(r.get("task_index"), (int, float)) and isinstance(r.get("score"), (int, float))
    ]
    if not valid:
        raise ValueError(
            f"No valid (task_index, score) records in {results_jsonl}. "
            f"Found {len(records)} entries (some may be errors)."
        )
    valid.sort(key=lambda r: r["task_index"])
    xs = [r["task_index"] for r in valid]
    ys = [r["score"] for r in valid]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, ys, marker="o", linewidth=1.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)

    maybe_create_parent_directory(output_path)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved learning curve to {output_path} ({len(valid)} points)")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot ACE learning curve.")
    parser.add_argument("--results", required=True, help="Path to results.jsonl.")
    parser.add_argument(
        "--output", default=None,
        help="Output PNG path (default: <results>.png).",
    )
    parser.add_argument("--title", default="ACE Learning Curve")
    parser.add_argument("--xlabel", default="Number of Adaptation Episodes")
    parser.add_argument("--ylabel", default="Task Goal Completion (%)")
    args = parser.parse_args()
    output = args.output or (os.path.splitext(args.results)[0] + ".png")
    plot_learning_curve(
        results_jsonl=args.results,
        output_path=output,
        title=args.title,
        xlabel=args.xlabel,
        ylabel=args.ylabel,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
