local base = import "/home/bwoo/workspace/ace-appworld/experiments/configs/ACE_eval_checkpoint_template.jsonnet";
base + {
  config+: {
    agent+: {
      trained_playbook_file_path: "/home/bwoo/workspace/ace-appworld/experiments/runs/ACE_offline_5epoch/playbooks/trained_playbook_snapshot_550.txt",
      generator_model_config+: { failure_log_path: "/home/bwoo/workspace/ace-appworld/experiments/runs/ACE_offline_5epoch/playbooks/llm_failures_eval.jsonl" },
    },
  },
}
