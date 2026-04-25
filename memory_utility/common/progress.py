"""Progress + timing + cost accounting for pipeline stages.

Everything a stage logs goes through this module so the output format stays
consistent. The heartbeat file is a tiny JSON blob rewritten periodically so
external tooling (or a watching user) can see progress without tailing logs.
"""
from __future__ import annotations

import contextlib
import json
import os
import sys
import threading
import time
from datetime import datetime
from typing import Any, Iterable, Iterator


# Rough per-1M-token prices in USD. Update as needed; this is a sanity
# estimator, not a billing ledger. Costs outside this table fall back to
# whatever the call itself reports, if anything.
PRICE_TABLE = {
    "gpt-4o-mini":       {"input_per_1m": 0.15,  "output_per_1m": 0.60},
    "gpt-4o":            {"input_per_1m": 2.50,  "output_per_1m": 10.00},
    "gpt-4.1-mini":      {"input_per_1m": 0.40,  "output_per_1m": 1.60},
    "deepseek/deepseek-v3.2": {"input_per_1m": 0.27, "output_per_1m": 1.10},
}


class CostAccumulator:
    """Tallies API calls for a stage. Registers via `.record(...)`."""

    def __init__(self) -> None:
        self.calls: int = 0
        self.tokens_in: int = 0
        self.tokens_out: int = 0
        self.reported_cost: float = 0.0  # sum of per-call costs the provider returned
        self._per_model: dict[str, dict[str, float]] = {}

    def record(
        self,
        model: str,
        tokens_in: int = 0,
        tokens_out: int = 0,
        reported_cost: float = 0.0,
    ) -> None:
        self.calls += 1
        self.tokens_in += int(tokens_in)
        self.tokens_out += int(tokens_out)
        self.reported_cost += float(reported_cost)
        pm = self._per_model.setdefault(model, {"calls": 0, "tokens_in": 0, "tokens_out": 0, "cost": 0.0})
        pm["calls"] += 1
        pm["tokens_in"] += int(tokens_in)
        pm["tokens_out"] += int(tokens_out)
        pm["cost"] += float(reported_cost)

    def estimate_cost(self) -> float:
        """Rough $ estimate using PRICE_TABLE; falls back to reported_cost when unknown."""
        total = 0.0
        for model, pm in self._per_model.items():
            prices = PRICE_TABLE.get(model)
            if prices is None:
                total += pm["cost"]
                continue
            total += pm["tokens_in"] * prices["input_per_1m"] / 1_000_000
            total += pm["tokens_out"] * prices["output_per_1m"] / 1_000_000
        return total

    def summary(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "reported_cost_usd": round(self.reported_cost, 4),
            "estimated_cost_usd": round(self.estimate_cost(), 4),
            "per_model": {k: {**v, "cost": round(v["cost"], 4)} for k, v in self._per_model.items()},
        }


class _Tee:
    """Duplicate writes to the original stream plus a log file."""
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def write(self, s):
        try:
            self.a.write(s)
        except Exception:
            pass
        try:
            self.b.write(s)
        except Exception:
            pass

    def flush(self):
        for s in (self.a, self.b):
            try:
                s.flush()
            except Exception:
                pass


class _Heartbeat:
    def __init__(self, path: str, stage: str, interval_s: float = 30.0):
        self.path = path
        self.stage = stage
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._t: threading.Thread | None = None
        self._progress: str = "starting"
        self._done: int = 0
        self._total: int | None = None
        self._lock = threading.Lock()

    def update(self, done: int | None = None, total: int | None = None, progress: str | None = None):
        with self._lock:
            if done is not None:
                self._done = done
            if total is not None:
                self._total = total
            if progress is not None:
                self._progress = progress

    def _tick(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                blob = {
                    "stage": self.stage,
                    "progress": self._progress,
                    "done": self._done,
                    "total": self._total,
                    "last_update": datetime.utcnow().isoformat() + "Z",
                }
            try:
                os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
                with open(self.path, "w", encoding="utf-8") as f:
                    json.dump(blob, f)
            except OSError:
                pass
            if self._stop.wait(self.interval_s):
                break

    def start(self) -> None:
        self._t = threading.Thread(target=self._tick, daemon=True)
        self._t.start()

    def stop(self) -> None:
        self._stop.set()
        if self._t is not None:
            self._t.join(timeout=2.0)


class StageContext:
    """Handle returned by stage_context()."""

    def __init__(self, name: str, run_dir: str):
        self.name = name
        self.run_dir = run_dir
        self.log_path = os.path.join(run_dir, "logs", f"{name}.log")
        self.heartbeat_path = os.path.join(run_dir, "logs", f"{name}.heartbeat.json")
        self.cost = CostAccumulator()
        self._start_time: float = 0.0
        self._orig_stdout = None
        self._orig_stderr = None
        self._log_fh = None
        self._hb = _Heartbeat(self.heartbeat_path, name)

    def progress(self, done: int | None = None, total: int | None = None, msg: str | None = None) -> None:
        self._hb.update(done=done, total=total, progress=msg)

    def iter(self, iterable: Iterable, total: int | None = None, desc: str | None = None) -> Iterator:
        """Simple progress iterator with heartbeat updates; avoids tqdm dep."""
        if total is None:
            try:
                total = len(iterable)  # type: ignore[arg-type]
            except TypeError:
                total = None
        self._hb.update(total=total, progress=desc or self.name)
        start = time.time()
        for i, item in enumerate(iterable):
            yield item
            done = i + 1
            self._hb.update(done=done, total=total, progress=desc or self.name)
            if done == 1 or done % max(1, (total or 20) // 20) == 0 or (total is not None and done == total):
                elapsed = time.time() - start
                if total:
                    eta = (elapsed / done) * (total - done)
                    print(f"  [{self.name}] {done}/{total} ({done*100//total}%) elapsed={elapsed:.1f}s eta={eta:.1f}s", flush=True)
                else:
                    print(f"  [{self.name}] {done} items elapsed={elapsed:.1f}s", flush=True)


@contextlib.contextmanager
def stage_context(name: str, run_dir: str) -> Iterator[StageContext]:
    """Context manager: mirrors stdout/stderr to a log file, prints banners, times the stage."""
    ctx = StageContext(name, run_dir)
    os.makedirs(os.path.dirname(ctx.log_path), exist_ok=True)
    ctx._log_fh = open(ctx.log_path, "a", encoding="utf-8", buffering=1)
    ctx._orig_stdout = sys.stdout
    ctx._orig_stderr = sys.stderr
    sys.stdout = _Tee(ctx._orig_stdout, ctx._log_fh)
    sys.stderr = _Tee(ctx._orig_stderr, ctx._log_fh)
    ctx._start_time = time.time()
    print(f"\n====== [{name}] start @ {datetime.utcnow().isoformat()}Z ======", flush=True)
    ctx._hb.start()
    try:
        yield ctx
    finally:
        ctx._hb.stop()
        wall = time.time() - ctx._start_time
        summary = ctx.cost.summary()
        print(f"====== [{name}] done in {wall:.1f}s ======", flush=True)
        print(f"  api calls: {summary['calls']}  tokens_in: {summary['tokens_in']}  tokens_out: {summary['tokens_out']}", flush=True)
        print(f"  reported cost: ${summary['reported_cost_usd']}  est. cost: ${summary['estimated_cost_usd']}", flush=True)
        if summary["per_model"]:
            for m, pm in summary["per_model"].items():
                print(f"    {m}: calls={pm['calls']} in={pm['tokens_in']} out={pm['tokens_out']} cost=${pm['cost']}", flush=True)
        sys.stdout = ctx._orig_stdout  # type: ignore[assignment]
        sys.stderr = ctx._orig_stderr  # type: ignore[assignment]
        try:
            ctx._log_fh.close()
        except Exception:
            pass
