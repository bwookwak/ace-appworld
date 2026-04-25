"""Judge / dummy-generation LLM client.

Per the design decision, we reuse the repo's existing `LiteLLMGenerator`
rather than introducing a new `litellm` direct dependency. Despite the
name, `LiteLLMGenerator` is an OpenAI-compatible wrapper and works with
both OpenAI and OpenRouter without modification. This module is a thin
adapter that: (a) sets a sensible default config for judge-style calls,
(b) extracts cost + token info into the pipeline's CostAccumulator format,
(c) surfaces a smaller, judge-shaped interface.
"""
from __future__ import annotations

import os
from typing import Any

from .progress import CostAccumulator


class JudgeClient:
    """Wraps LiteLLMGenerator for pipeline judge + dummy-generation calls.

    Instantiate once per stage. Uses the ace env's `LiteLLMGenerator` under
    the hood. `.generate(prompt, system=...)` returns (content, cost_usd).
    All call telemetry is forwarded to the passed CostAccumulator so the
    stage's end-of-run summary reflects judge cost.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        provider: str = "openai",
        completion_method: str = "openai",
        temperature: float = 0.0,
        max_retries: int = 3,
        use_cache: bool = True,
        cost_accumulator: CostAccumulator | None = None,
        **extra: Any,
    ) -> None:
        from appworld_experiments.code.ace.lite_llm_generator import LiteLLMGenerator
        self.model = model
        self.provider = provider
        self.cost_accumulator = cost_accumulator
        kwargs = {
            "name": model,
            "provider": provider,
            "completion_method": completion_method,
            "temperature": temperature,
            "seed": 100,
            "max_retries": max_retries,
            "retry_after_n_seconds": 3,
            "retry_backoff_factor": 2.0,
            "max_retry_wait_seconds": 30.0,
            "retry_on_empty_response": True,
            "on_failure": "raise",
            "use_cache": use_cache,
        }
        # When routing through OpenRouter, auto-fill api_key + base_url so the
        # caller can just pass provider="openrouter" without extra plumbing.
        if provider == "openrouter":
            if "api_key" not in extra:
                or_key = os.environ.get("OPENROUTER_API_KEY")
                if or_key:
                    extra["api_key"] = or_key
            extra.setdefault("base_url", "https://openrouter.ai/api/v1")
        kwargs.update(extra)
        self._gen = LiteLLMGenerator(**kwargs)

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        **kwargs: Any,
    ) -> tuple[str, float]:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        output: dict[str, Any] = self._gen.generate(messages=messages, **kwargs)
        content: str = output.get("content", "") or ""
        cost: float = float(output.get("cost", 0.0) or 0.0)
        if self.cost_accumulator is not None:
            self.cost_accumulator.record(
                model=self.model,
                tokens_in=int(output.get("input_tokens", 0) or 0),
                tokens_out=int(output.get("output_tokens", 0) or 0),
                reported_cost=cost,
            )
        return content, cost

    def chat(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> tuple[str, float]:
        output: dict[str, Any] = self._gen.generate(messages=messages, **kwargs)
        content: str = output.get("content", "") or ""
        cost: float = float(output.get("cost", 0.0) or 0.0)
        if self.cost_accumulator is not None:
            self.cost_accumulator.record(
                model=self.model,
                tokens_in=int(output.get("input_tokens", 0) or 0),
                tokens_out=int(output.get("output_tokens", 0) or 0),
                reported_cost=cost,
            )
        return content, cost


def ensure_openai_env(provider: str = "openai") -> None:
    """Sanity check that the right env var exists for the chosen provider."""
    if provider == "openrouter":
        if not os.environ.get("OPENROUTER_API_KEY"):
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set. Required when judge_provider=openrouter."
            )
    else:
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Required when judge_provider=openai. "
                "If your OPENAI_API_KEY is actually an OpenRouter key (sk-or-...), pass "
                "--judge-provider openrouter and a model like 'openai/gpt-4o-mini' instead."
            )
