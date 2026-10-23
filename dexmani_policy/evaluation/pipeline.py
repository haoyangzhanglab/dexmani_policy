"""Select a checkpoint and evaluate its saved test plan without video."""

from omegaconf import OmegaConf

from dexmani_policy.evaluation.rollout import evaluate_checkpoint
from dexmani_policy.evaluation.selection import select_best_checkpoint


def run_pipeline(
    exp_dir,
    cfg,
    *,
    episodes=None,
    eval_seed=None,
    inference_steps=None,
    action_steps=None,
    use_ema=None,
    overrides=None,
):
    cfg = OmegaConf.to_container(OmegaConf.create(cfg), resolve=True)
    selection_cfg = OmegaConf.merge(cfg, overrides if overrides is not None else {})
    if episodes is not None:
        OmegaConf.update(selection_cfg, "eval.offline.episodes", episodes)
    record = select_best_checkpoint(
        exp_dir,
        selection_cfg,
        inference_steps=inference_steps,
        action_steps=action_steps,
        use_ema=use_ema,
        eval_seed=eval_seed,
        record_video=False,
    )
    return evaluate_checkpoint(
        exp_dir,
        cfg,
        selection_record=record,
        action_steps=action_steps,
        inference_steps_list=[inference_steps] if inference_steps is not None else None,
        use_ema=use_ema,
        overrides=overrides,
        episodes=episodes,
        eval_seed=eval_seed,
        record_video=False,
    )
