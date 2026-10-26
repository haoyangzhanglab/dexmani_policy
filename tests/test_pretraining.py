import json

import pytest
import torch

from dexmani_policy.agents.hand_prior.hand_vae import load_hand_vae
from dexmani_policy.agents.hand_prior.motion_prior import load_motion_prior
from dexmani_policy.agents.hand_prior.vqvae import HandVQVAE
from dexmani_policy.common.config import load_policy_config
from dexmani_policy.common.hand_representation import validate_pretraining_data
from dexmani_policy.datasets.replay_buffer import ReplayBuffer
from dexmani_policy.datasets.split import split_episodes
from dexmani_policy.training.pretrain import hand_vae, lamp, vqvae
from dexmani_policy.training.pretrain.export_codebook import export_codebook


@pytest.fixture(params=[0.0, 0.25])
def split_path(demonstrations, tmp_path, request):
    _, _, manifest = split_episodes(ReplayBuffer(demonstrations, keys=[]), val_ratio=request.param)
    path = tmp_path / "split.json"
    path.write_text(json.dumps(manifest))
    return path


def test_hand_vae_pretraining_roundtrip(demonstrations, tmp_path, one_thread):
    output = tmp_path / "codec.pt"
    cfg = load_policy_config("hand_vae", [
        f"zarr_path={demonstrations}", f"codec.path={output}", "horizon=4",
        "codec.hidden_dim=8", "codec.n_layers=1", "codec.n_heads=2", "codec.latent_dim=3",
        "training.device=cpu", "training.steps=2", "training.batch_size=4",
        "training.kl_warmup=0", "training.kl_ramp=1", "training.eval_every=1",
    ])
    payload = hand_vae.train(cfg)
    restored, saved = load_hand_vae(output)
    torch.testing.assert_close(restored.state_dict(), payload["state_dict"])
    assert saved["metadata"]["completed_updates"] == 2
    assert saved["metadata"]["loss_reduction"]["kl"] == "mean_over_batch_time_latent"


def test_lamp_pretraining_roundtrip(demonstrations, split_path, tmp_path, one_thread):
    output = tmp_path / "prior.pt"
    cfg = load_policy_config("lamp", [
        f"zarr_path={demonstrations}", f"dataset.split_manifest={split_path}",
        f"lamp.prior_path={output}", "lamp.hidden_dim=8", "prior_training.device=cpu",
        "prior_training.steps=2", "prior_training.batch_size=8", "prior_training.eval_every=1",
        "prior_training.beta_warmup=0", "prior_training.warmup_steps=0",
    ])
    lamp.train(cfg)
    restored, saved = load_motion_prior(output)
    assert bool(restored.ready)
    dataset = lamp.build_prior_dataset(cfg)
    selection = saved["metadata"]["selection"]
    assert selection["split"] == ("validation" if dataset.val_mask.any() else "final_no_holdout")
    if not dataset.val_mask.any():
        assert selection["selection_step"] == cfg.prior_training.steps
    validate_pretraining_data(
        dataset.describe_hand_data(), dataset.data_recipe["split_manifest"],
        saved["metadata"]["hand_data"], saved["metadata"]["split_manifest"],
    )
    bad = dict(saved["metadata"]["hand_data"], dt=999)
    with pytest.raises(ValueError, match="data mismatch"):
        validate_pretraining_data(
            dataset.describe_hand_data(), dataset.data_recipe["split_manifest"],
            bad, saved["metadata"]["split_manifest"],
        )


def test_vq_pretraining_and_export(demonstrations, split_path, tmp_path, one_thread):
    output = tmp_path / "vq"
    cfg = load_policy_config("dqrise", [
        f"zarr_path={demonstrations}", f"dataset.split_manifest={split_path}",
        "horizon=4", "n_action_steps=2", "dataset.pad_after=1", f"vq_vae.output_dir={output}",
        "vq_vae.device=cpu", "vq_vae.num_epochs=1",
        "vq_vae.batch_size=8", "vq_vae.latent_dim=8", "vq_vae.hidden_dim=16",
        "vq_vae.num_layers=2", "vq_vae.warmup_steps=0",
    ])
    vqvae.train(cfg)
    checkpoint = output / "vqvae_hand_best.pt"
    payload = torch.load(checkpoint, weights_only=True)
    has_validation = bool(payload["split_metadata"]["split_manifest"]["val_indices"])
    assert payload["split_metadata"]["selection_split"] == (
        "validation" if has_validation else "train_no_holdout"
    )
    restored = HandVQVAE.from_checkpoint(payload, device="cpu")
    assert restored.hand_dim == 12
    codebook = export_codebook(checkpoint, output / "codebook.npz")
    assert codebook.num_codes == 16
    assert torch.isfinite(codebook.sorted_hand_poses).all()
    assert (output / "codebook_diagnostics.json").is_file()
