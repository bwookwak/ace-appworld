local base = import "/home/bwoo/workspace/ace-appworld/experiments/configs/ACE_eval_checkpoint_template.jsonnet";
base + {
  config+: {
    dataset: "test_normal",
    agent+: {
      trained_playbook_file_path: "/home/bwoo/workspace/ace-appworld/memory_utility/runs/baseline_test_normal/stage1/baseline_playbook.txt",
      generator_model_config+: {
        failure_log_path: "/home/bwoo/workspace/ace-appworld/memory_utility/runs/baseline_test_normal/stage1/llm_failures_eval.jsonl",
        on_failure: "warn",
        max_retries: 10,
      },
    },
  },
}
