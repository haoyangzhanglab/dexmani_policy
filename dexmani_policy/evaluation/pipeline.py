"""Select, then execute the frozen test request without reparsing research settings."""

from pathlib import Path

from dexmani_policy.evaluation.config import resolve_evaluation_request
from dexmani_policy.evaluation.rollout import evaluate_policy
from dexmani_policy.evaluation.selection import select_best_checkpoint


def run_pipeline(exp_dir, cfg, *, overrides=None):
    record = select_best_checkpoint(exp_dir, cfg, overrides=overrides)
    request = resolve_evaluation_request(cfg, exp_dir, selection_record=record)
    return evaluate_policy(request, output_root=Path(exp_dir) / "eval_dexsim")
