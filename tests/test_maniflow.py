from dexmani_policy.common.checkpoint import load_checkpoint
from dexmani_policy.deployment import select_inference_weights
from dexmani_policy.training.config import load_policy_config

from tests.helpers import check_flow_recipe_defaults, check_recipe_composition


def test_recipe_composition():
    cfg = check_recipe_composition("maniflow")
    check_flow_recipe_defaults("maniflow", cfg)


def test_checkpoint_allows_one_flow_step(small_policy_checkpoint):
    checkpoint = load_checkpoint(small_policy_checkpoint)
    checkpoint.effective_config = load_policy_config("maniflow", ["eval.inference_steps=1"])
    assert select_inference_weights(checkpoint)[1] == 1
