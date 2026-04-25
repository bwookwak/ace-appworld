local base = import "/home/bwoo/workspace/ace-appworld/experiments/configs/ACE_eval_checkpoint_template.jsonnet";
base + {
  config+: {
    dataset: "dev",
    agent+: {
      trained_playbook_file_path: "/home/bwoo/workspace/ace-appworld/memory_utility/runs/test_run_parallel/stage2/full_memory_playbook.txt",
      generator_prompt_file_path: "/home/bwoo/workspace/ace-appworld/memory_utility/runs/test_run_parallel/stage2/generator_prompt.txt",
      generator_model_config+: { failure_log_path: "/home/bwoo/workspace/ace-appworld/memory_utility/runs/test_run_parallel/stage2/llm_failures_eval.jsonl" },
    },
  },
}
