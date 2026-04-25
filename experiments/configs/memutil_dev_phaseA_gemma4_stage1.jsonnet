local base = import "/home/bwoo/workspace/ace-appworld/experiments/configs/ACE_eval_checkpoint_template.jsonnet";
base + {
  config+: {
    dataset: "dev",
    agent+: {
      trained_playbook_file_path: "/home/bwoo/workspace/ace-appworld/memory_utility/runs/dev_phaseA_gemma4/stage1/baseline_playbook.txt",
      generator_model_config+: {
        failure_log_path: "/home/bwoo/workspace/ace-appworld/memory_utility/runs/dev_phaseA_gemma4/stage1/llm_failures_eval.jsonl",
        on_failure: "warn",
        max_retries: 10,
        name: "gemma-4-26b-a4b-it",
        provider: "gemini",
        openrouter_provider: null,
      },
    },
  },
}
