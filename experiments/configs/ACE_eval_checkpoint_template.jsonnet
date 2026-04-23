// Template config used by CheckpointWatcher to evaluate a single playbook
// snapshot on test_normal.
//
// The watcher generates per-checkpoint wrapper configs at:
//   <run_dir>/configs/eval_ckpt_<N>.jsonnet
// and mirrors them into experiments/configs/ (so `appworld run eval_ckpt_<N>`
// can find them). Each wrapper overrides:
//   - agent.trained_playbook_file_path
//   - agent.{generator,reflector,curator}_model_config.failure_log_path
//
// The defaults below kick in only if you invoke this template directly via
// `appworld run ACE_eval_checkpoint_template` (rare; mostly used by watcher).

local project_home_path = std.extVar("APPWORLD_PROJECT_PATH");
local experiment_prompts_path = project_home_path + "/experiments/prompts";
local experiment_playbooks_path = project_home_path + "/experiments/playbooks";
// Default run-dir matches the orchestrator default so direct invocation finds
// a sensible location for the (template-only) failure log.
local run_dir = project_home_path + "/experiments/runs/ACE_offline_5epoch";

local model_config = {
    "name": "deepseek/deepseek-v3.2",
    "provider": "openrouter",
    "completion_method": "openai",
    "temperature": 0,
    "seed": 100,
    "stop": ["<|endoftext|>", "<|eot_id|>", "<|start_header_id|>"],
    "logprobs": false,
    "top_logprobs": null,
    "frequency_penalty": 0,
    "presence_penalty": 0,
    "n": 1,
    "response_format": {"type": "text"},
    "retry_after_n_seconds": 5,
    "retry_backoff_factor": 2.0,
    "max_retry_wait_seconds": 60.0,
    "retry_on_empty_response": true,
    "on_failure": "raise",  // eval should not crash on a single bad call
    "failure_log_path": run_dir + "/playbooks/llm_failures_eval.jsonl",  // overridden by watcher wrapper
    "use_cache": true,
    "max_retries": 5,
    "openrouter_provider": {
        // Mirror the adaptation config so eval routing is consistent.
        // Push past slow/dead providers before our 180s HTTP timeout fires.
        "sort": "throughput",
        "preferred_min_throughput": { "p90": 30 },   // tokens/sec
        "preferred_max_latency":    { "p90": 2 },   // seconds
        "allow_fallbacks": true,
        // "ignore": [],
    },
};

{
    "type": "ace",
    "config": {
        "run_type": "ace-evaluation",
        "agent": {
            "type": "ace_evaluation_react",
            "generator_model_config": model_config,
            "appworld_config": {
                "random_seed": 123,
            },
            "logger_config": {
                "color": true,
                "verbose": true,
            },
            "generator_prompt_file_path": experiment_prompts_path + "/appworld_react_generator_prompt.txt",
            // Overridden per-checkpoint by the wrapper jsonnet the watcher
            // creates. Default points at the latest trained playbook in the
            // default run-dir so direct invocation still works.
            "trained_playbook_file_path": run_dir + "/playbooks/trained_playbook.txt",
            "ignore_multiple_calls": true,
            "max_steps": 40,
            "max_cost_overall": 5000,
            "max_cost_per_task": 10,
            "log_lm_calls": true,
        },
        "dataset": "test_normal",
    }
}
