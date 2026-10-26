"""Select, then execute the frozen test request without reparsing research settings."""

from pathlib import Path

from dexmani_policy.evaluation.config import frozen_test_request
from dexmani_policy.evaluation.rollout import evaluate_policy
from dexmani_policy.evaluation.selection import select_best_checkpoint


def run_pipeline(exp_dir, cfg, *, overrides=None):
    record, checkpoint = select_best_checkpoint(exp_dir, cfg, overrides=overrides)
    request = frozen_test_request(exp_dir, record, checkpoint)
    return evaluate_policy(request, output_root=Path(exp_dir) / "eval_dexsim")
