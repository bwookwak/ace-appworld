local base = import "/home/bwoo/workspace/ace-appworld/experiments/configs/ACE_eval_checkpoint_template.jsonnet";
base + {
  config+: {
    dataset: "dev",
    agent+: {
      trained_playbook_file_path: "/home/bwoo/workspace/ace-appworld/memory_utility/runs/dev_phaseB_snap200/stage2/full_memory_playbook.txt",
      generator_prompt_file_path: "/home/bwoo/workspace/ace-appworld/memory_utility/runs/dev_phaseB_snap200/stage2/generator_prompt.txt",
      generator_model_config+: {
        failure_log_path: "/home/bwoo/workspace/ace-appworld/memory_utility/runs/dev_phaseB_snap200/stage2/llm_failures_eval.jsonl",
        on_failure: "warn",
        max_retries: 10,
        name: "gpt-5-mini",
        temperature: 1,
        stop: null,
        seed: null,
        logprobs: null,
        top_logprobs: null,
        frequency_penalty: null,
        presence_penalty: null,
        response_format: null,
        provider: "openai",
        openrouter_provider: null,
      },
    },
  },
}
