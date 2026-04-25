import inspect
import json
import os
import sys
import threading
import time
import traceback
import uuid
from datetime import datetime
from typing import Any, Literal

import httpx
import litellm
from joblib import Memory
from litellm import completion_cost, token_counter
from openai import (
    APIConnectionError,
    APIError,
    APIResponseValidationError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    ConflictError,
    InternalServerError,
    NotFoundError,
    OpenAI,
    OpenAIError,
    PermissionDeniedError,
    RateLimitError,
    UnprocessableEntityError,
)
from rich.panel import Panel

from appworld import AppWorld
from appworld.common.path_store import path_store
from appworld.common.utils import maybe_create_parent_directory, rprint, write_jsonl

litellm.drop_params = True
cache = Memory(os.path.join(path_store.cache, "llm_calls"), verbose=0)


class LLMEmptyResponseError(Exception):
    """Raised when LLM returns an empty response and we want to retry."""


class LLMMaxRetriesError(Exception):
    """Raised when LLM call fails after exhausting all retry attempts."""


# Errors that are typically transient and worth retrying with backoff.
RETRY_ERROR = (
    APIConnectionError,
    APITimeoutError,
    APIResponseValidationError,
    InternalServerError,
    RateLimitError,
    APIStatusError,
    APIError,
    OpenAIError,
    LLMEmptyResponseError,
)
# Errors that are permanent: API key wrong, model name wrong, request malformed.
# Retrying these only wastes time and can be confusing to debug. Fail fast.
FATAL_ERROR = (
    AuthenticationError,
    PermissionDeniedError,
    NotFoundError,
    BadRequestError,
    UnprocessableEntityError,
    ConflictError,
)
CHAT_COMPLETION = {  # These are lambda so set environment variables take effect at runtime
    "openai": lambda: OpenAI(api_key="dummy", base_url="https://api.openai.com/v1").chat.completions.create,
    "litellm": lambda: litellm.completion,
}


# OpenAI Python SDK's default request timeout is 600s (10 min). When OpenRouter
# routes to a slow/dead provider, the call hangs for the full 10 min before our
# retry loop ever sees an APITimeoutError. Override with a tight budget so a
# stuck connection becomes a normal retriable error after at most ~3 min, and
# disable the SDK's internal retries (we have our own retry loop with backoff).
_HTTP_TIMEOUT = httpx.Timeout(connect=10.0, read=180.0, write=30.0, pool=10.0)
_SDK_MAX_RETRIES = 0


def _build_client(provider: str):
    p = provider.strip().lower()
    if p == "sambanova":
        from sambanova import SambaNova
        return SambaNova()
    if p == "together":
        from together import Together
        return Together()
    if p == "openai":
        return OpenAI(timeout=_HTTP_TIMEOUT, max_retries=_SDK_MAX_RETRIES)
    if p == "openrouter":
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY environment variable is required for provider=openrouter."
            )
        base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        return OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=_HTTP_TIMEOUT,
            max_retries=_SDK_MAX_RETRIES,
        )
    if p == "gemini":
        # Google AI Studio exposes an OpenAI-compatible endpoint for Gemini /
        # Gemma models. Point an OpenAI client at it and authenticate with
        # GEMINI_API_KEY (common Google AI Studio key).
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY (or GOOGLE_API_KEY) environment variable is required for provider=gemini."
            )
        base_url = os.environ.get(
            "GEMINI_BASE_URL",
            "https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        return OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=_HTTP_TIMEOUT,
            max_retries=_SDK_MAX_RETRIES,
        )
    raise ValueError(f"Invalid provider: {provider}.")


def non_cached_chat_completion(
    completion_method: str,
    provider: str,
    model: str,
    messages: list[dict[str, str]],
    frequency_penalty: float | None = None,
    logprobs: bool | None = None,
    top_logprobs: int | None = None,
    max_completion_tokens: int | None = None,
    max_tokens: int | None = None,
    n: int | None = None,
    parallel_tool_calls: bool | None = None,
    presence_penalty: float | None = None,
    reasoning_effort: Literal["low", "medium", "high"] | None = None,
    response_format: dict | None = None,
    seed: int | None = None,
    stop: str | list[str] | None = None,
    temperature: float | None = None,
    tool_choice: str | dict | None = None,
    tools: list | None = None,
    top_p: float | None = None,
    # above params are shared by litellm and openai
    # below params are only for litellm
    logit_bias: dict | None = None,
    thinking: dict | None = None,
    base_url: str | None = None,
    api_version: str | None = None,
    api_key: str | None = None,
    model_list: list | None = None,
    custom_llm_provider: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    kwargs["model"] = model
    kwargs["messages"] = messages
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if stop is not None:
        kwargs["stop"] = stop
    if temperature is not None:
        kwargs["temperature"] = temperature
    if top_p is not None:
        kwargs["top_p"] = top_p
    if completion_method not in ["openai", "litellm"]:
        raise ValueError(
            f"Invalid completion_method: {completion_method}. "
            "Valid values are: 'openai' or 'litellm'."
        )

    client = _build_client(provider)
    response = client.chat.completions.create(**kwargs)
    response = to_dict(response)
    return response


@cache.cache
def cached_chat_completion(
    completion_method: str,
    provider: str,
    model: str,
    messages: list[dict[str, str]],
    frequency_penalty: float | None = None,
    logprobs: bool | None = None,
    top_logprobs: int | None = None,
    max_completion_tokens: int | None = None,
    max_tokens: int | None = None,
    n: int | None = None,
    parallel_tool_calls: bool | None = None,
    presence_penalty: float | None = None,
    reasoning_effort: Literal["low", "medium", "high"] | None = None,
    response_format: dict | None = None,
    seed: int | None = None,
    stop: str | list[str] | None = None,
    temperature: float | None = None,
    tool_choice: str | dict | None = None,
    tools: list | None = None,
    top_p: float | None = None,
    # above params are shared by litellm and openai
    # below params are only for litellm
    logit_bias: dict | None = None,
    thinking: dict | None = None,
    base_url: str | None = None,
    api_version: str | None = None,
    api_key: str | None = None,
    model_list: list | None = None,
    custom_llm_provider: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:

    return non_cached_chat_completion(
        completion_method=completion_method,
        provider=provider,
        model=model,
        messages=messages,
        frequency_penalty=frequency_penalty,
        logprobs=logprobs,
        top_logprobs=top_logprobs,
        max_completion_tokens=max_completion_tokens,
        max_tokens=max_tokens,
        n=n,
        parallel_tool_calls=parallel_tool_calls,
        presence_penalty=presence_penalty,
        reasoning_effort=reasoning_effort,
        response_format=response_format,
        seed=seed,
        stop=stop,
        temperature=temperature,
        tool_choice=tool_choice,
        tools=tools,
        top_p=top_p,
        logit_bias=logit_bias,
        thinking=thinking,
        base_url=base_url,
        api_version=api_version,
        api_key=api_key,
        model_list=model_list,
        custom_llm_provider=custom_llm_provider,
        **kwargs,
    )


def _build_fatal_hint(provider: str, model: str, error_type: str) -> str:
    """Generate a human-readable hint for permanent (non-retriable) errors."""
    p = provider.strip().lower()
    if error_type in ("AuthenticationError", "PermissionDeniedError"):
        if p == "openrouter":
            has_key = bool(os.environ.get("OPENROUTER_API_KEY"))
            return (
                "Hint: provider=openrouter requires OPENROUTER_API_KEY env var. "
                f"Currently set: {has_key}. "
                "Verify the key at https://openrouter.ai/keys ."
            )
        if p == "openai":
            has_key = bool(os.environ.get("OPENAI_API_KEY"))
            return (
                "Hint: provider=openai requires OPENAI_API_KEY env var. "
                f"Currently set: {has_key}. "
                "Also confirm OPENAI_BASE_URL points at the right endpoint."
            )
        if p == "sambanova":
            return "Hint: provider=sambanova requires SAMBANOVA_API_KEY (see SambaNova SDK docs)."
        if p == "together":
            return "Hint: provider=together requires TOGETHER_API_KEY env var."
        return f"Hint: check authentication for provider={provider}."
    if error_type in ("NotFoundError",):
        if p == "openrouter":
            return (
                f"Hint: model '{model}' may not exist on OpenRouter. "
                "OpenRouter expects fully-qualified ids like 'deepseek/deepseek-chat' "
                "or 'openai/gpt-4o'. See https://openrouter.ai/models ."
            )
        return (
            f"Hint: model '{model}' was not found on provider '{provider}'. "
            "Check the model id matches the provider's catalog."
        )
    if error_type in ("BadRequestError", "UnprocessableEntityError"):
        return (
            "Hint: the request payload was rejected by the server. "
            "Common causes: unsupported parameter for this model "
            "(e.g. response_format, stop, seed) or invalid messages format."
        )
    if error_type == "ConflictError":
        return "Hint: server reported a conflict; the request will not succeed as-is."
    return f"Hint: provider={provider} returned a permanent error; will not retry."


def _write_failure_log(
    log_path: str | None,
    model: str,
    provider: str,
    error_type: str,
    error_msg: str,
    attempts: int,
) -> None:
    if not log_path:
        return
    try:
        maybe_create_parent_directory(log_path)
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "model": model,
            "provider": provider,
            "error_type": error_type,
            "error_msg": error_msg[:1000],
            "attempts": attempts,
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        traceback.print_exc()


class _CallWatchdog:
    """Logs 'still waiting' periodically while an LLM call is in-flight.

    Without this, a slow/hung HTTP request looks like the program is frozen for
    minutes. The watchdog spawns a daemon thread that prints elapsed time at
    each threshold (in seconds) and stops as soon as the wrapped block exits.
    """

    def __init__(
        self,
        model: str,
        attempt: int,
        max_attempts: int,
        thresholds: tuple[int, ...] = (30, 60, 120, 240),
    ) -> None:
        self._stop = threading.Event()
        self._t0 = time.time()
        self._thresholds = thresholds
        self._meta = (model, attempt, max_attempts)
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        for thr in self._thresholds:
            wait = thr - (time.time() - self._t0)
            if wait > 0 and self._stop.wait(timeout=wait):
                return
            if self._stop.is_set():
                return
            elapsed = time.time() - self._t0
            model, attempt, max_attempts = self._meta
            print(
                f"[LLM WATCHDOG] still waiting model={model} "
                f"attempt={attempt}/{max_attempts} elapsed={elapsed:.0f}s",
                flush=True,
            )

    def __enter__(self) -> "_CallWatchdog":
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()


class LiteLLMGenerator:
    def __init__(
        self,
        name: str,
        provider: str = "openai",
        completion_method: Literal["openai", "litellm"] = "openai",
        retry_after_n_seconds: int | None = None,
        max_retries: int = 500,
        retry_backoff_factor: float = 2.0,
        max_retry_wait_seconds: float = 120.0,
        retry_on_empty_response: bool = True,
        on_failure: Literal["raise", "warn", "exit"] = "raise",
        failure_log_path: str | None = None,
        use_cache: bool = False,
        token_cost_data: dict | None = None,
        openrouter_provider: dict | None = None,
        **generation_kwargs: Any,
    ) -> None:
        self.model = name
        self.provider = provider
        default_custom_llm_provider = (
            "openai" if name not in litellm.model_cost and completion_method == "openai" else None
        )
        self.custom_llm_provider = generation_kwargs.get(
            "custom_llm_provider", default_custom_llm_provider
        )
        if token_cost_data:
            litellm.model_cost[name] = token_cost_data
        elif name not in litellm.model_cost:
            warning_message = (
                f"[yellow]litellm does not have token cost data for model '{name}'. "
                "So the cost tracking and logging will not work. If you need it, though, pass 'token_cost_data' "
                "in the config file in the same format as litellm.model_cost[name].[/yellow]"
            )
            rprint(
                Panel(warning_message, title="[bold red]Warning[/bold red]", border_style="yellow")
            )
        if completion_method not in ["openai", "litellm"]:
            raise ValueError(
                f"Invalid completion_method: {completion_method}. "
                "Valid values are: 'openai' or 'litellm'."
            )
        self.max_input_tokens = litellm.model_cost.get("name", {}).get("max_input_tokens", None)
        self.max_output_tokens = litellm.model_cost.get("name", {}).get("max_output_tokens", None)
        self.retry_after_n_seconds = retry_after_n_seconds
        self.max_retries = max_retries
        self.retry_backoff_factor = retry_backoff_factor
        self.max_retry_wait_seconds = max_retry_wait_seconds
        self.retry_on_empty_response = retry_on_empty_response
        self.on_failure = on_failure
        self.failure_log_path = failure_log_path
        # OpenRouter-specific provider routing preferences (e.g. min_throughput,
        # sort, allow_fallbacks). Passed via OpenAI SDK's `extra_body` as
        # {"provider": <dict>}. Only applied when provider == "openrouter".
        self.openrouter_provider = openrouter_provider or None
        self.chat_completion = {
            True: cached_chat_completion,
            False: non_cached_chat_completion,
        }[use_cache]
        if completion_method == "openai":
            # LiteLLM accepts these two arguments in completion function, whereas OpenAI
            # accepts them in the OpenAI constructor or in the environment variables.
            if "api_key" in generation_kwargs:
                os.environ["OPENAI_API_KEY"] = generation_kwargs.pop("api_key")
            if "base_url" in generation_kwargs:
                os.environ["OPENAI_BASE_URL"] = generation_kwargs.pop("base_url")
            generation_kwargs.pop("custom_llm_provider", None)
        valid_generation_kwargs_keys = set(
            inspect.signature(CHAT_COMPLETION[completion_method]()).parameters.keys()
        )
        invalid_keys = set(generation_kwargs.keys()) - valid_generation_kwargs_keys
        if "max_tokens" not in generation_kwargs and self.max_output_tokens:
            generation_kwargs["max_tokens"] = self.max_output_tokens
        # Drop None-valued params so the OpenAI SDK doesn't forward them as
        # "unsupported parameter" errors. Some models (e.g. OpenAI GPT-5 family)
        # reject legacy params like `stop`, `seed`, `logprobs`, `response_format`,
        # penalties, etc. Stage jsonnet overrides them with null so they are
        # merged away here.
        generation_kwargs = {k: v for k, v in generation_kwargs.items() if v is not None}
        generation_kwargs["completion_method"] = completion_method
        generation_kwargs["provider"] = provider
        self.generation_kwargs = generation_kwargs
        self.cost = 0
        self.log_file_path = None
        self._logger = None  # set externally if structured logging is desired

    def attach_logger(self, logger) -> None:
        """Attach a Logger instance for structured retry/failure logs."""
        self._logger = logger

    def generate(
        self,
        messages: list[dict[str, str]],
        tools: list[dict] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        used_num_tokens = token_counter(model=self.model, messages=messages)
        if self.max_input_tokens and used_num_tokens > self.max_input_tokens:
            print(
                "WARNING: Ran out of context limit of this model. "
                f"Model: {self.model}, used_num_tokens: {used_num_tokens}, "
                f"max_num_tokens: {self.max_input_tokens}"
            )
            return {"content": "", "tool_calls": [], "cost": 0}

        last_exception: BaseException | None = None
        last_error_type = "Unknown"
        last_error_msg = ""
        response: dict[str, Any] | None = None
        success = False
        attempt = 0
        for attempt in range(self.max_retries):
            t0 = time.time()
            if self._logger is not None:
                try:
                    self._logger.log_llm_request_start(
                        model=self.model,
                        attempt=attempt + 1,
                        max_attempts=self.max_retries,
                    )
                except AttributeError:
                    # Older Logger without the helper; fall back to a print.
                    print(
                        f"[LLM] -> {self.model} | attempt {attempt + 1}/{self.max_retries}",
                        flush=True,
                    )
            try:
                arguments = {
                    "model": self.model,
                    "messages": messages,
                    "tools": tools,
                    **(self.generation_kwargs | kwargs),
                }
                if (
                    self.provider.strip().lower() == "openrouter"
                    and self.openrouter_provider
                ):
                    existing_extra_body = arguments.get("extra_body") or {}
                    if not isinstance(existing_extra_body, dict):
                        existing_extra_body = {}
                    merged_provider = {
                        **(existing_extra_body.get("provider") or {}),
                        **self.openrouter_provider,
                    }
                    arguments["extra_body"] = {
                        **existing_extra_body,
                        "provider": merged_provider,
                    }
                with _CallWatchdog(
                    model=self.model,
                    attempt=attempt + 1,
                    max_attempts=self.max_retries,
                ):
                    response = self.chat_completion(**arguments)
                # Some providers (notably OpenRouter) return a date-stamped
                # snapshot id in the response (e.g. "deepseek/deepseek-v3.2"
                # -> "deepseek/deepseek-v3.2-20251201"). litellm's cost
                # lookup uses that response model id, which then fails with
                # "This model isn't mapped yet". Preserve the upstream value
                # under "upstream_model" for traceability and overwrite the
                # primary "model" field with the name we actually requested.
                if isinstance(response, dict):
                    upstream_model = response.get("model")
                    if upstream_model and upstream_model != self.model:
                        response["upstream_model"] = upstream_model
                    response["model"] = self.model
                response["cost"] = self.completion_cost(completion_response=response)
                self.may_log_call(arguments, response)
                # Empty response detection
                if self.retry_on_empty_response:
                    try:
                        choice_msg = response["choices"][0]["message"]
                        content = (choice_msg.get("content") or "").strip()
                        tool_calls = choice_msg.get("tool_calls") or []
                        if not content and not tool_calls:
                            raise LLMEmptyResponseError(
                                "LLM returned empty content and no tool_calls"
                            )
                    except (KeyError, IndexError, TypeError) as exc:
                        raise LLMEmptyResponseError(
                            f"LLM response missing expected fields: {exc}"
                        )
                duration = time.time() - t0
                if self._logger is not None:
                    self._logger.log_llm_call(
                        model=self.model, attempt=attempt + 1,
                        duration_s=duration, success=True,
                    )
                success = True
                break
            except FATAL_ERROR as exception:
                # Permanent errors: wrong API key, wrong model id, malformed
                # request, etc. Retrying never helps and only delays the obvious
                # fix. Surface immediately with a helpful message.
                duration = time.time() - t0
                last_exception = exception
                last_error_type = type(exception).__name__
                last_error_msg = (
                    getattr(exception, "message", None) or str(exception) or last_error_type
                )
                if self._logger is not None:
                    self._logger.log_llm_call(
                        model=self.model, attempt=attempt + 1,
                        duration_s=duration, success=False, error=last_error_msg,
                    )
                _write_failure_log(
                    self.failure_log_path, self.model, self.provider,
                    last_error_type, last_error_msg, attempt + 1,
                )
                hint = _build_fatal_hint(self.provider, self.model, last_error_type)
                rprint(
                    Panel(
                        f"{last_error_type}: {last_error_msg}\n\n{hint}",
                        title="[bold red]LLM FATAL (no retry)[/bold red]",
                        border_style="red",
                    )
                )
                if self._logger is not None:
                    self._logger.log_llm_failure(
                        model=self.model, attempts=attempt + 1, error=last_error_msg,
                    )
                raise LLMMaxRetriesError(
                    f"[FATAL] {last_error_type}: {last_error_msg[:200]}\n{hint}"
                ) from exception
            except RETRY_ERROR as exception:
                duration = time.time() - t0
                last_exception = exception
                last_error_type = type(exception).__name__
                last_error_msg = (
                    getattr(exception, "message", None) or str(exception) or last_error_type
                )
                if self._logger is not None:
                    self._logger.log_llm_call(
                        model=self.model, attempt=attempt + 1,
                        duration_s=duration, success=False, error=last_error_msg,
                    )
                if self.retry_after_n_seconds is None:
                    print(traceback.format_exc())
                    _write_failure_log(
                        self.failure_log_path, self.model, self.provider,
                        last_error_type, last_error_msg, attempt + 1,
                    )
                    raise LLMMaxRetriesError(
                        f"LLM call failed and retry_after_n_seconds is None: {last_error_msg}"
                    ) from exception
                wait = min(
                    self.retry_after_n_seconds * (self.retry_backoff_factor ** attempt),
                    self.max_retry_wait_seconds,
                )
                rprint(
                    Panel(
                        (
                            f"[bold yellow]reason[/bold yellow]: "
                            f"{last_error_type}: {last_error_msg[:300].strip()}\n"
                            f"[bold yellow]call_duration[/bold yellow]: {duration:.1f}s\n"
                            f"[bold yellow]sleeping[/bold yellow]: {wait:.1f}s "
                            f"(base={self.retry_after_n_seconds}s, "
                            f"backoff={self.retry_backoff_factor}, "
                            f"cap={self.max_retry_wait_seconds}s)\n"
                            f"[bold yellow]attempt[/bold yellow]: "
                            f"{attempt + 1}/{self.max_retries}"
                        ),
                        title=f"[yellow]LLM RETRY[/yellow] {self.model}",
                        border_style="yellow",
                    )
                )
                time.sleep(wait)

        if not success:
            attempts = attempt + 1
            _write_failure_log(
                self.failure_log_path, self.model, self.provider,
                last_error_type, last_error_msg, attempts,
            )
            failure_msg = (
                f"[LLM FAILURE] model={self.model} provider={self.provider} "
                f"attempts={attempts} last_error={last_error_type}: {last_error_msg[:200]}"
            )
            rprint(
                Panel(failure_msg, title="[bold red]LLM FAILURE[/bold red]", border_style="red")
            )
            if self._logger is not None:
                self._logger.log_llm_failure(
                    model=self.model, attempts=attempts, error=last_error_msg,
                )
            if self.on_failure == "raise":
                raise LLMMaxRetriesError(failure_msg) from last_exception
            elif self.on_failure == "warn":
                return {"content": "", "tool_calls": [], "cost": 0}
            elif self.on_failure == "exit":
                sys.exit(1)
            else:
                raise ValueError(f"Unknown on_failure mode: {self.on_failure}")

        if "chat_template_kwargs" in self.generation_kwargs:
            response["choices"][0]["message"]["content"] = response["choices"][0]["message"]["content"].split("<think>\n")[-1]

        output = {**response["choices"][0]["message"], "cost": response["cost"]}
        return output

    def may_log_call(self, arguments: dict, response: dict) -> None:
        log_data = {"id": uuid.uuid4().hex, "input": arguments, "output": response}
        if self.log_file_path:
            os.makedirs(os.path.dirname(self.log_file_path), exist_ok=True)
            write_jsonl([log_data], self.log_file_path, append=True, silent=True)

    def log_calls_to(self, file_path: str | None = None, world: AppWorld | None = None) -> None:
        if (world and file_path) or (not world and not file_path):
            raise ValueError("Either world or file_path must be provided.")
        if world:
            file_path = os.path.join(world.output_logs_directory, "lm_calls.jsonl")
        self.log_file_path = file_path

    def completion_cost(self, *args: Any, **kwargs: Any) -> float:
        if self.model in litellm.model_cost:
            if self.custom_llm_provider:
                kwargs["custom_llm_provider"] = self.custom_llm_provider
            return round(completion_cost(*args, **kwargs), 8)
        return 0.0


def to_dict(obj: Any) -> Any:
    if hasattr(obj, "json"):
        return {k: to_dict(v) for k, v in dict(obj).items()}
    elif isinstance(obj, list):
        return [to_dict(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}
    return obj
