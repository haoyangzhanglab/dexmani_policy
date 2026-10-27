import json
import os
import subprocess

import pytest
import torch

from dexmani_policy.agents.hand_prior.hand_vae import load_hand_vae
from dexmani_policy.agents.hand_prior.motion_prior import load_motion_prior
from dexmani_policy.agents.hand_prior.vqvae import HandVQVAE
from dexmani_policy.common.config import PROJECT_ROOT, load_policy_config
from dexmani_policy.common.hand_representation import validate_pretraining_data
from dexmani_policy.datasets.common.replay_buffer import ReplayBuffer
from dexmani_policy.datasets.common.split import split_episodes
from dexmani_policy.training.common.data import build_dataset_and_normalizer
from dexmani_policy.training.pretrain.hand_prior import hand_vae, lamp, vqvae
from dexmani_policy.training.pretrain.hand_prior.export_codebook import export_codebook


@pytest.fixture(params=[0.0, 0.25])
def split_path(demonstrations, tmp_path, request):
    _, _, manifest = split_episodes(ReplayBuffer(demonstrations, keys=[]), val_ratio=request.param)
    path = tmp_path / "split.json"
    path.write_text(json.dumps(manifest))
    return path


@pytest.mark.parametrize("environment, overrides, task, source, output", [
    ({}, [], "pick_apple_messy", None, None),
    ({"TASK_NAME": "pick_apple_messy"}, ["stack_cups"], "stack_cups", None, None),
    ({"TASK_NAME": "stack_cups"}, [], "stack_cups", None, None),
    ({"TASK_NAME": "pick_apple_messy"}, ["task_name=stack_cups"], "stack_cups", None, None),
    ({}, ["pick_apple_messy", "task_name=stack_cups"], "stack_cups", None, None),
    (
        {"ZARR_PATH": "environment data.zarr", "OUTPUT_DIR": "environment output"},
        [], "pick_apple_messy", "environment data.zarr", "environment output",
    ),
    (
        {"ZARR_PATH": "environment data.zarr", "OUTPUT_DIR": "environment output"},
        ["zarr_path=command line data.zarr", "vq_vae.output_dir=command line output"],
        "pick_apple_messy", "command line data.zarr", "command line output",
    ),
])
def test_vq_launcher_overrides(tmp_path, monkeypatch, environment, overrides, task, source, output):
    launcher = tmp_path / "bin"
    launcher.mkdir()
    conda = launcher / "conda"
    conda.write_text('#!/bin/sh\nprintf \'%s\\n\' "$@"\n')
    conda.chmod(0o755)
    monkeypatch.setenv("PATH", f"{launcher}:{os.environ['PATH']}")
    for name in ("TASK_NAME", "ZARR_PATH", "OUTPUT_DIR"):
        monkeypatch.delenv(name, raising=False)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    result = subprocess.run(
        ["bash", str(PROJECT_ROOT / "scripts/pretrain/vqvae.sh"), *overrides],
        check=True, capture_output=True, text=True,
    )
    arguments = result.stdout.splitlines()
    index = arguments.index("--config")
    cfg = load_policy_config(arguments[index + 1], arguments[index + 2:])
    assert cfg.task_name == task
    assert cfg.dataset.zarr_path == cfg.zarr_path == (source or f"robot_data/{task}.zarr")
    assert cfg.vq_vae.output_dir == output


@pytest.mark.parametrize("name, prepare", [
    ("dqrise", vqvae.build_policy_dataset),
    ("lamp", lamp.build_prior_dataset),
])
def test_hand_pretraining_and_policy_require_shared_split(demonstrations, name, prepare):
    cfg = load_policy_config(name, [f"zarr_path={demonstrations}"])
    for build in (prepare, build_dataset_and_normalizer):
        with pytest.raises(ValueError, match="split_manifest"):
            build(cfg)


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


@pytest.mark.parametrize("num_groups, codebook_size", [(2, 4), (1, 3)])
def test_vq_pretraining_and_export(
    demonstrations, split_path, tmp_path, one_thread, num_groups, codebook_size,
):
    output = tmp_path / "vq"
    cfg = load_policy_config("dqrise", [
        f"zarr_path={demonstrations}", f"dataset.split_manifest={split_path}",
        f"vq_vae.output_dir={output}",
        f"vq_vae.num_groups={num_groups}", f"vq_vae.codebook_size={codebook_size}",
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
    assert codebook.num_codes == codebook_size ** num_groups
    assert torch.isfinite(codebook.sorted_hand_poses).all()
    assert (output / "codebook_diagnostics.json").is_file()
