from unittest.mock import patch

import pytest

from dexmani_policy.common.checkpoint import Checkpoint, save_checkpoint
from dexmani_policy.deployment import runtime


def test_load_policy_rejects_missing_data_processing_before_restore(tmp_path):
    checkpoint = Checkpoint(
        global_step=0, model_state={}, ema_model_state=None,
        optimizer_state={}, scheduler_state={},
        effective_config={"eval": {"use_ema": False, "inference_steps": 1}},
    )
    path = save_checkpoint(tmp_path / "policy.pt", checkpoint)
    with (
        patch.object(runtime, "restore_policy") as restore,
        pytest.raises(ValueError) as error,
    ):
        runtime.load_policy(path, device="cpu")
    assert str(error.value) == (
        "Checkpoint is missing required data_processing metadata. "
        "Please retrain with the current code."
    )
    restore.assert_not_called()
