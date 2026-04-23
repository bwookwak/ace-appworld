// Offline adaptation with periodic playbook checkpointing.
// Designed to be paired with orchestrator.py which spawns a CheckpointWatcher
// to evaluate each *_snapshot_<N>.txt produced here.
//
// When run via orchestrator, the following paths are OVERRIDDEN to live under
// `--run-dir` (default: experiments/runs/<this_config_name>/):
//   - agent.trained_playbook_file_path
//   - agent.{generator,reflector,curator}_model_config.failure_log_path
// The defaults below are used only if you invoke `appworld run` directly.

local project_home_path = std.extVar("APPWORLD_PROJECT_PATH");
local experiment_prompts_path = project_home_path + "/experiments/prompts";
local experiment_playbooks_path = project_home_path + "/experiments/playbooks";
// Default run-dir for direct invocation (mirrors orchestrator's default).
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
    "retry_after_n_seconds": 3,
    "retry_backoff_factor": 2.0,
    "max_retry_wait_seconds": 30.0,
    "retry_on_empty_response": true,
    "on_failure": "raise",
    "failure_log_path": run_dir + "/playbooks/llm_failures.jsonl",  // overridden by orchestrator
    "use_cache": true,
    "max_retries": 5,
    "openrouter_provider": {
        // Prioritize throughput, but require providers to actually be fast at
        // P90 (90% of recent calls). This pushes routing away from providers
        // whose tail latency would otherwise trigger our 180s HTTP timeout.
        "sort": "throughput",
        "preferred_min_throughput": { "p90": 30 },   // tokens/sec
        "preferred_max_latency":    { "p90": 2 },   // seconds
        "allow_fallbacks": true,
        // "ignore": [],  // populate from llm_failures.jsonl when a specific
                          // provider is observed to time out repeatedly.
    },
};

{
    "type": "ace",
    "config": {
        "run_type": "ace-adaptation",
        "resume_from_checkpoint": false,
        "agent": {
            "type": "ace_adaptation_react",
            "generator_model_config": model_config,
            "reflector_model_config": model_config,
            "curator_model_config": model_config,
            "appworld_config": {
                "random_seed": 123,
            },
            "logger_config": {
                "color": true,
                "verbose": true,
            },
            "generator_prompt_file_path": experiment_prompts_path + "/appworld_react_generator_prompt.txt",
            "reflector_prompt_file_path": experiment_prompts_path + "/appworld_react_reflector_no_gt_prompt.txt",
            "curator_prompt_file_path": experiment_prompts_path + "/appworld_react_curator_prompt.txt",
            "initial_playbook_file_path": experiment_playbooks_path + "/appworld_initial_playbook.txt",
            // Overridden by orchestrator to <run_dir>/playbooks/trained_playbook.txt
            "trained_playbook_file_path": run_dir + "/playbooks/trained_playbook.txt",
            "ignore_multiple_calls": true,
            "max_steps": 40,
            "max_cost_overall": 5000,
            "max_cost_per_task": 10,
            "log_lm_calls": true,
            "use_gt_code": false,
            "checkpoint_every_n_tasks": 50,
        },
        "dataset": "train",
        "num_epochs": 10,
    }
}
