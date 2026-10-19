"""Production hand-policy contracts on temporary, explicitly aligned synthetic episodes.

DEXMANI_TEST_DEVICE=cuda:0 enables the actual GPU training entry; default CPU uses
its shared builders and the same Trainer. Artifacts come from production pretraining.
"""

import copy
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import hydra
import numpy as np
import pytest
import torch
import zarr
from omegaconf import OmegaConf, open_dict
from torch.utils.data import default_collate

from dexmani_policy.smoke_test import load_config
from dexmani_policy.training.build_utils import (
    build_dataset_and_normalizer, build_model_and_ema, build_optimizer_and_scheduler,
    bind_action_space_artifact, validate_config, compile_models,
)
from dexmani_policy.training.checkpoint import CheckpointStore, restore_policy_agent
from dexmani_policy.training.resume import build_train_loader, restore_training_state
from dexmani_policy.training.workspace import TrainWorkspace
from dexmani_policy.train import TrainingComponents, build_train_components, build_trainer
from dexmani_policy.utils.hand_artifact import source_contract, policy_representation, state_sha256
from dexmani_policy.utils.tensor import dict_apply

DEVICE = os.environ.get("DEXMANI_TEST_DEVICE", "cpu")
ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(2)


def run_module(module, *args):
    env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    result = subprocess.run([sys.executable, "-m", module, *map(str, args)], cwd=ROOT,
                            env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    root = tmp_path_factory.mktemp("hand-contracts")
    data_path = root / "data.zarr"
    data = zarr.open_group(str(data_path), mode="w")
    rng = np.random.default_rng(7)
    frames = 72
    actions = (rng.normal(size=(frames, 19)) * np.linspace(.1, .5, 19) + np.linspace(-1, 2, 19)).astype("f4")
    states = actions + np.linspace(.2, .6, 19).astype("f4")
    cloud = rng.normal(size=(frames, 16, 6)).astype("f4")
    data.create_group("data")
    for key, values in (("action", actions), ("joint_state", states), ("point_cloud", cloud)):
        data["data"].create_dataset(key, data=values, chunks=(8, *values.shape[1:]))
    data.create_group("meta").create_dataset("episode_ends", data=np.array([24, 48, 72], dtype="i8"))
    data.attrs.update(dt=.05, obs_alignment="obs[t]_before_action[t]", data_revision="synthetic-v1",
                      task_name="pick_apple_messy", episode_ids=["e0", "e1", "e2"])
    manifest = dict(data_revision="synthetic-v1", episode_ids=["e0", "e1", "e2"],
                    trial_ids={f"e{i}": f"t{i}" for i in range(3)}, train_ids=["e0", "e1"],
                    val_ids=["e2"], exclusions=[], seed=7, group_unit="episode")
    split_path = root / "split.json"
    split_path.write_text(json.dumps(manifest))
    configs = {}
    for name in ("lamp", "temporal_codec_flow", "dqrise"):
        cfg = load_config(name)
        with open_dict(cfg):
            cfg.zarr_path = str(data_path)
            cfg.action_key = "action"
            cfg.env_runner.env_kwargs.control_mode = "joint"
            cfg.dataset.max_train_episodes = None
            cfg.dataset.val_ratio = 0
            cfg.dataset.split_manifest = str(split_path)
            cfg.dataset.augmentation_cfg = None
            cfg.dataloader.batch_size = 2
            cfg.dataloader.num_workers = 0
            cfg.dataloader.persistent_workers = False
            cfg.training.device = DEVICE
            cfg.training.use_compile = False
            cfg.training.use_bfloat16 = False
            cfg.training.lr_warmup_steps = 0
            cfg.training.loop.total_train_steps = 6
            cfg.training.loop.log_interval_steps = 1
            cfg.workspace.wandb_cfg = None
            cfg.workspace.output_dir = str(root / (name + "_run"))
            cfg.agent.num_points = 16
            cfg.agent.state_out_dim = 8
            cfg.agent.fps_random_config = dict(use_random=False, use_random_start=False,
                                                random_noise_scale=0., use_shuffle_output=False)
            if name != "dqrise":
                cfg.agent.pc_encoder_config = dict(channels=24, num_patches=4, group_size=4,
                                                  depth=1, num_heads=4, mlp_ratio=2.)
            if name == "lamp":
                cfg.lamp.hidden_dim = 16
                cfg.lamp.prior_path = str(root / "prior.pt")
            elif name == "temporal_codec_flow":
                cfg.codec.path = str(root / "codec.pt")
                cfg.codec.hidden_dim = 16
                cfg.codec.n_layers = 1
                cfg.agent.n_layers = 1
                cfg.agent.hidden_dim = 24
                cfg.agent.n_head = 4
                cfg.agent.p_drop_attn = 0.
            else:
                cfg.codebook_path = str(root / "codebook.npz")
                cfg.agent.encoder_type = "dp3"
                cfg.agent.pc_out_dim = 16
                cfg.agent.down_dims = [16, 32]
                cfg.agent.diffusion_step_embed_dim = 16
                cfg.vq_vae.hidden_dim = 16
                cfg.vq_vae.latent_dim = 8
                cfg.vq_vae.num_layers = 1
        # load_config resolves interpolations; rebind changed shared values explicitly.
        cfg.dataset.zarr_path = str(data_path)
        cfg.dataset.action_key = "action"
        cfg.agent.action_dim = cfg.action_dim = 19
        if name == "lamp":
            cfg.agent.prior_path = cfg.lamp.prior_path
            cfg.agent.prior_hidden_dim = cfg.lamp.hidden_dim
        elif name == "temporal_codec_flow":
            cfg.agent.codec_path = cfg.codec.path
            cfg.agent.codec_hidden_dim = cfg.codec.hidden_dim
            cfg.agent.codec_n_layers = cfg.codec.n_layers
        else:
            cfg.tcp_dim = cfg.agent.tcp_dim = 7
            cfg.agent.codebook_path = cfg.codebook_path
        path = root / (name + ".yaml")
        OmegaConf.save(cfg, path)
        configs[name] = cfg
    run_module("dexmani_policy.training.train_lamp_prior", "--policy-config", root / "lamp.yaml",
               "--device", "cpu", "--steps", 3, "--beta-warmup", 2, "--warmup-steps", 0,
               "--eval-every", 1, "--batch-size", 8)
    run_module("dexmani_policy.training.train_temporal_codec", "--policy-config", root / "temporal_codec_flow.yaml",
               "--device", "cpu", "--steps", 3, "--kl-warmup", 1, "--kl-ramp", 1,
               "--eval-every", 1, "--batch-size", 8)
    run_module("dexmani_policy.training.train_vq_hand", "--policy-config", root / "dqrise.yaml",
               "--output_dir", root / "vq", "--device", "cpu", "--num_epochs", 2,
               "--batch_size", 8, "--num_workers", 0, "--warmup_steps", 0)
    run_module("dexmani_policy.training.export_hand_codebook", "--checkpoint", root / "vq/vqvae_hand_best.pt",
               "--output", root / "codebook.npz", "--device", "cpu")
    return root, configs


def build(cfg, *, checkpoint=None):
    if DEVICE.startswith("cuda"):
        return build_train_components(cfg, resume_checkpoint=checkpoint)
    dataset, normalizer = build_dataset_and_normalizer(cfg, resume_checkpoint=checkpoint, verbose=False)
    loader = build_train_loader(cfg, dataset)
    model, ema, updater = build_model_and_ema(cfg, torch.device(DEVICE), normalizer,
                                            dataset=dataset, checkpoint=checkpoint)
    optimizer, scheduler = build_optimizer_and_scheduler(cfg, model, verbose=False)
    workspace = TrainWorkspace(cfg.workspace.output_dir)
    return TrainingComponents(torch.device(DEVICE), model, ema, updater, optimizer, scheduler,
                              loader, workspace, len(loader), checkpoint)


def batch_for(dataset, count=2):
    return dict_apply(default_collate([dataset[i] for i in range(count)]), lambda x: x.to(DEVICE))


@pytest.mark.parametrize("name,compiled", [("lamp", False), ("temporal_codec_flow", False),
                                           ("dqrise", False), ("lamp", True)])
def test_trainer_resume_raw_ema_without_external_artifact(prepared, tmp_path, name, compiled):
    root, configs = prepared
    cfg = copy.deepcopy(configs[name])
    cfg.workspace.output_dir = str(tmp_path / "run")
    cfg.training.use_compile = compiled
    validate_config(cfg)
    comp = build(cfg)
    trainer = build_trainer(cfg, comp)
    comp.workspace.save_hydra_config(cfg)
    if compiled:
        compile_models(comp.model, comp.ema_model, **({"backend": "eager", "mode": "default"} if DEVICE == "cpu" else {}))
    frozen = policy_representation(comp.model)[1]
    frozen_hash = state_sha256(frozen)
    trainable = [p for p in comp.model.parameters() if p.requires_grad]
    assigned = [p for g in comp.optimizer.param_groups for p in g["params"]]
    assert len(assigned) == len({id(p) for p in assigned}) == len(trainable)
    before = comp.model.action_decoder.model.state_dict()
    before = {k: v.clone() for k, v in before.items()}
    trainer.train(max_updates=2)
    assert trainer.global_step == 2
    assert any(not torch.equal(before[k], v) for k, v in comp.model.action_decoder.model.state_dict().items())
    assert frozen_hash == state_sha256(frozen)
    path = comp.workspace.checkpoint_dir / "latest.pt"
    checkpoint = CheckpointStore(path.parent).load(path)
    assert checkpoint.effective_config["action_space_artifact"] == OmegaConf.to_container(cfg.action_space_artifact)
    batch = batch_for(comp.train_loader.dataset)
    context = batch.get("action_context")
    artifact = Path(getattr(comp.model, "prior_path", getattr(comp.model, "codec_path", getattr(comp.model, "codebook_path", ""))))
    hidden = artifact.with_suffix(artifact.suffix + ".hidden")
    artifact.rename(hidden)
    try:
        data_path = Path(comp.train_loader.dataset.zarr_path)
        hidden_data = data_path.with_suffix(".hidden")
        data_path.rename(hidden_data)
        try:
            for use_ema, model in ((False, comp.model), (True, comp.ema_model)):
                model.eval()
                kwargs = {"action_context": context} if context is not None else {}
                torch.manual_seed(11)
                expected = model.predict_action(batch["obs"], **kwargs)
                restored = restore_policy_agent(checkpoint.effective_config, path, use_ema=use_ema, device=DEVICE)
                torch.manual_seed(11)
                actual = restored.predict_action(batch["obs"], **kwargs)
                torch.testing.assert_close(actual["pred_action"], expected["pred_action"])
                assert actual["control_action"].shape == (2, cfg.n_action_steps, 19)
                assert torch.isfinite(actual["pred_action"]).all()
                assert state_sha256(policy_representation(restored)[1]) == frozen_hash
        finally:
            hidden_data.rename(data_path)
        resume_cfg = OmegaConf.create(checkpoint.effective_config)
        with open_dict(resume_cfg):
            resume_cfg.resume_from = str(path)
            resume_cfg.workspace.output_dir = str(tmp_path / "resume")
        resumed = build(resume_cfg, checkpoint=checkpoint)
        resumed_trainer = build_trainer(resume_cfg, resumed)
        state = restore_training_state(checkpoint, resume_contract=resumed_trainer.resume_contract,
            model=resumed.model, ema_model=resumed.ema_model, ema_updater=resumed.ema_updater,
            optimizer=resumed.optimizer, scheduler=resumed.scheduler, device=DEVICE)
        assert state == (2, trainer.current_epoch, trainer.next_micro_step)
        if compiled:
            compile_models(resumed.model, resumed.ema_model, **({"backend": "eager", "mode": "default"} if DEVICE == "cpu" else {}))
        resumed_trainer.train(resume_state=state, max_updates=1)
        assert resumed_trainer.global_step == 3
        resumed.workspace.close()
        changed_path = tmp_path / "changed.zarr"
        shutil.copytree(comp.train_loader.dataset.zarr_path, changed_path)
        changed_data = zarr.open_group(str(changed_path), mode="a")
        changed_data["data/action"][:24] = changed_data["data/action"][:24][::-1]
        bad_cfg = OmegaConf.create(checkpoint.effective_config)
        with open_dict(bad_cfg):
            bad_cfg.resume_from = str(path)
            bad_cfg.dataset.zarr_path = str(changed_path)
            bad_cfg.workspace.output_dir = str(tmp_path / "rejected")
        with pytest.raises(ValueError, match="source mismatch"):
            build(bad_cfg, checkpoint=checkpoint)
    finally:
        hidden.rename(artifact)
        comp.workspace.close()


def test_lamp_gradients_coordinates_and_compile(prepared):
    cfg = copy.deepcopy(prepared[1]["lamp"])
    dataset, normalizer = build_dataset_and_normalizer(cfg, verbose=False)
    model, _, _ = build_model_and_ema(cfg, DEVICE, normalizer, dataset=dataset)
    batch = batch_for(dataset)
    assert "hand_latent_target" not in batch
    final = model.action_decoder.model[-1]
    with torch.no_grad():
        final.weight.zero_()
        final.bias.zero_()
    model.eval()
    condition, mu = model.encode_condition(batch["obs"], batch["action_context"])
    loss, metrics = model.compute_loss(batch)
    assert metrics["offset_l2"] == 0
    assert metrics["loss_offset"] == 0
    prediction = model.predict_action(batch["obs"], action_context=batch["action_context"])["pred_action"]
    affine = model.normalizer["action"].params_dict
    torch.testing.assert_close(prediction[:, 0, :7], (-affine["offset"][:7] / affine["scale"][:7]).expand(2, -1))
    torch.testing.assert_close(prediction[:, 0, 7:], model.prior.unnormalize_hand(model.prior.decode(mu)))
    with torch.no_grad():
        final.bias[7:] = torch.tensor([.2, -.3], device=DEVICE)
    model.zero_grad(set_to_none=True)
    loss, metrics = model.compute_loss(batch)
    assert metrics["offset_l2"].item() == pytest.approx(.13)
    grad = torch.autograd.grad(metrics["loss_offset"], final.bias, retain_graph=True)[0][7:]
    torch.testing.assert_close(grad, torch.tensor([.0004, -.0006], device=DEVICE))
    decoded_grad = torch.autograd.grad(metrics["loss_decoded_hand"], final.bias, retain_graph=True)[0][7:]
    assert decoded_grad.abs().sum() > 0
    loss.backward()
    assert all(p.grad is None for p in model.prior.parameters())
    model.train(); assert not model.prior.training
    model.eval()
    eager = model.predict_action_from_cond(condition, mu=mu)["pred_action"]
    calls = []
    def backend(graph, example_inputs):
        calls.append(graph)
        return graph.forward
    model.compile_backbone(backend=backend)
    compiled = model.predict_action_from_cond(condition, mu=mu)["pred_action"]
    assert calls and hasattr(model.action_decoder.model, "_orig_mod")
    torch.testing.assert_close(compiled, eager)
    with pytest.raises(ValueError, match="inference_steps"):
        model.predict_action(batch["obs"], inference_steps=10, action_context=batch["action_context"])


def test_history_reference_and_episode_padding(prepared):
    lamp = hydra.utils.instantiate(prepared[1]["lamp"].dataset)
    history = lamp.history_for_frame(24)
    np.testing.assert_array_equal(history, np.repeat(lamp.hand_state_bootstrap[1:2], 8, axis=0))
    before = lamp.history_for_frame(27).copy()
    lamp.hand_actions[27:] += 100
    np.testing.assert_array_equal(before, lamp.history_for_frame(27))
    cfg = copy.deepcopy(prepared[1]["temporal_codec_flow"])
    cfg.dataset.augmentation_cfg = {"state": {"noise": {"noise_std": 1., "prob": 1.}}}
    dataset = hydra.utils.instantiate(cfg.dataset)
    for i in (0, 1, len(dataset) - 1):
        rows = dataset.sampler.source_rows(dataset.sampler.indices[i:i + 1])[0]
        sample = dataset[i]
        q0 = sample["action_context"]["hand_reference"].numpy()
        np.testing.assert_array_equal(q0, dataset.hand_states[rows[1]])
        assert not np.array_equal(q0, sample["obs"]["joint_state"][-1, 7:].numpy())
        np.testing.assert_array_equal(sample["action"].numpy(), dataset.replay_buffer.read("action", rows))
        assert np.all(np.searchsorted(dataset.replay_buffer.episode_ends, rows, side="right") ==
                      np.searchsorted(dataset.replay_buffer.episode_ends, rows[1], side="right"))
    rows = dataset.sampler.source_rows(dataset.sampler.indices[:1])[0]
    np.testing.assert_array_equal(rows[1:9], np.arange(8))


def test_temporal_mean_full_chunk_gradients(prepared):
    cfg = copy.deepcopy(prepared[1]["temporal_codec_flow"])
    dataset, norm = build_dataset_and_normalizer(cfg, verbose=False)
    model, _, _ = build_model_and_ema(cfg, DEVICE, norm, dataset=dataset)
    batch = batch_for(dataset)
    q0 = batch["action_context"]["hand_reference"]
    codec = model.codec
    actions = batch["action"][..., 7:]
    normalized = codec.normalize_actions(actions, q0)
    torch.testing.assert_close(normalized * codec.action_std + codec.action_mean + q0[:, None], actions)
    a, _ = codec.encode(actions, q0); b, _ = codec.encode(actions, q0)
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    terms = codec.reconstruction_terms(normalized + .25, normalized)
    assert terms["velocity"] < 1e-12 and terms["acceleration"] < 1e-12
    seen = []
    handle = codec.decoder_input.register_forward_pre_hook(lambda module, args: seen.append(args[0].shape[1]))
    loss, metrics = model.compute_loss(batch)
    grads = torch.autograd.grad(metrics["loss_decoded_hand"], list(model.action_decoder.model.parameters()), retain_graph=True, allow_unused=True)
    assert any(g is not None and g.abs().sum() > 0 for g in grads)
    loss.backward()
    assert all(p.grad is None for p in codec.parameters())
    model.train(); assert not codec.training
    model.eval()
    result = model.predict_action(batch["obs"])
    handle.remove()
    assert seen and all(h == 16 for h in seen)
    torch.testing.assert_close(result["control_action"], result["pred_action"][:, 1:9])
    with pytest.raises(ValueError, match="hand_reference"):
        model.compute_loss({"obs": batch["obs"], "action": batch["action"]})


def test_dq_nn_rounding_export_and_physical_decode(prepared, monkeypatch):
    from dexmani_policy.agents.hand_prior.codebook import HandStateCodebook
    from dexmani_policy.agents.hand_prior.vqvae import HandVQVAE
    root, configs = prepared
    book = HandStateCodebook(12)
    book.load(root / "codebook.npz")
    indices = torch.arange(16)
    scalar = 2 * indices / 15 - 1
    poses, roundtrip = book.decode(scalar)
    torch.testing.assert_close(roundtrip, indices)
    _, boundaries = book.decode(torch.tensor([-10., -1., 1., 10.]))
    assert boundaries.tolist() == [0, 0, 15, 15]
    assert book.decode(torch.tensor(0.))[1].item() == 8  # Exactly representable index 7.5.
    for index in (0, 4, 7, 14):
        midpoint = 2 * (index + .5) / 15 - 1
        assert book.decode(torch.tensor(midpoint - 1e-6, dtype=torch.float64))[1].item() == index
        assert book.decode(torch.tensor(midpoint + 1e-6, dtype=torch.float64))[1].item() == index + 1
    probe = poses + .0123
    scalar, labels = book.encode(probe)
    expected = ((probe[:, None] - poses[None]) ** 2).sum(-1).argmin(-1)
    torch.testing.assert_close(labels, expected)
    payload = torch.load(root / "vq/vqvae_hand_best.pt", weights_only=False)
    vq = HandVQVAE.from_checkpoint(payload, device="cpu")
    torch.testing.assert_close(book.layer_weights, vq.quantizer.layer_weights.softmax(0))
    combinations = torch.cartesian_prod(torch.arange(4), torch.arange(4))
    latent = sum(vq.quantizer.layer_weights.softmax(0)[i] * vq.codebooks[i, combinations[:, i]]
                 for i in range(2))
    expected_poses = vq.decode(latent).detach().clamp(-1, 1)[book.pca_permutation]
    torch.testing.assert_close(poses, expected_poses)
    cfg = copy.deepcopy(configs["dqrise"])
    dataset, norm = build_dataset_and_normalizer(cfg, verbose=False)
    model, _, _ = build_model_and_ema(cfg, DEVICE, norm, dataset=dataset)
    native = torch.zeros(1, 16, 8, device=DEVICE)
    native[0, :, -1] = torch.linspace(-1, 1, 16, device=DEVICE)
    monkeypatch.setattr(model.action_decoder, "predict_action", lambda *args, **kwargs: native)
    result = model.predict_action_from_cond(torch.zeros(1, 1, device=DEVICE))["pred_action"]
    params = norm["action"].params_dict
    physical = (poses - params["offset"][7:]) / params["scale"][7:]
    torch.testing.assert_close(result[0, :, 7:].cpu(), physical)


@pytest.mark.parametrize("name", ["lamp", "temporal_codec_flow", "dqrise"])
def test_source_binding_relocation_and_rejection(prepared, tmp_path, name):
    root, configs = prepared
    cfg = copy.deepcopy(configs[name])
    dataset, norm = build_dataset_and_normalizer(cfg, verbose=False)
    model, _, _ = build_model_and_ema(cfg, DEVICE, norm, dataset=dataset)
    copied = tmp_path / "same.zarr"
    shutil.copytree(root / "data.zarr", copied)
    moved_cfg = copy.deepcopy(cfg)
    moved_cfg.dataset.zarr_path = str(copied)
    moved = hydra.utils.instantiate(moved_cfg.dataset)
    bind_action_space_artifact(moved_cfg, model, moved)
    assert source_contract(moved) == source_contract(dataset)
    data = zarr.open_group(str(copied), mode="a")
    # Permutation preserves every affine statistic while changing command chronology.
    data["data/action"][:24] = data["data/action"][:24][::-1]
    changed = hydra.utils.instantiate(moved_cfg.dataset)
    with pytest.raises(ValueError, match="source mismatch"):
        bind_action_space_artifact(moved_cfg, model, changed)
    swapped = copy.deepcopy(cfg.dataset)
    split = json.loads((root / "split.json").read_text())
    split["train_ids"], split["val_ids"] = ["e0", "e2"], ["e1"]
    path = tmp_path / "split.json"; path.write_text(json.dumps(split))
    swapped.split_manifest = str(path)
    with pytest.raises(ValueError, match="source mismatch"):
        bind_action_space_artifact(cfg, model, hydra.utils.instantiate(swapped))
    if name != "lamp":
        swapped = copy.deepcopy(cfg.dataset)
        swapped.pad_after += 1
        with pytest.raises(ValueError, match="source mismatch"):
            bind_action_space_artifact(cfg, model, hydra.utils.instantiate(swapped))


@pytest.mark.parametrize("name", ["lamp", "temporal_codec_flow"])
def test_selection_and_legacy_artifacts(prepared, tmp_path, name):
    from dexmani_policy.utils.hand_artifact import validate_selection_schedule
    from dexmani_policy.agents.hand_prior.motion_prior import load_motion_prior
    root, configs = prepared
    for steps, beta, eligible in ((1, .001, 2), (10, 0., 2), (10, float("nan"), 2)):
        with pytest.raises(ValueError, match="KL warmup/ramp"):
            validate_selection_schedule(steps, beta, eligible)
    source = root / ("prior.pt" if name == "lamp" else "codec.pt")
    payload = torch.load(source, weights_only=True)
    selection = payload["metadata"]["selection"]
    assert selection["selection_step"] >= 2
    assert selection["beta"] == selection["target_beta"]
    assert selection["split"] == "validation"
    cfg = copy.deepcopy(configs[name])
    cfg.dataset.split_manifest = None
    cfg.dataset.val_ratio = 0
    output = tmp_path / "no-holdout.pt"
    if name == "lamp":
        cfg.lamp.prior_path = cfg.agent.prior_path = str(output)
        args = ["--beta-warmup", 2, "--warmup-steps", 0]
        module = "train_lamp_prior"
    else:
        cfg.codec.path = cfg.agent.codec_path = str(output)
        args = ["--kl-warmup", 1, "--kl-ramp", 1]
        module = "train_temporal_codec"
    path = tmp_path / "config.yaml"; OmegaConf.save(cfg, path)
    run_module("dexmani_policy.training." + module, "--policy-config", path, "--device", "cpu",
               "--steps", 3, "--eval-every", 1, "--batch-size", 8, *args)
    final = torch.load(output, weights_only=True)
    assert final["metadata"]["selection"]["split"] == "final_no_holdout"
    assert final["metadata"]["selection"]["selection_step"] == 3
    if name == "lamp":
        payload["_format"] = "lamp_prior.v1"
        old = tmp_path / "old.pt"; torch.save(payload, old)
        with pytest.raises(ValueError, match="historical"):
            load_motion_prior(old)
    else:
        payload["metadata"].pop("source_contract")
        bad = tmp_path / "old.pt"; torch.save(payload, bad)
        cfg.agent.codec_path = str(bad)
        model = hydra.utils.instantiate(cfg.agent)
        with pytest.raises(ValueError, match="regenerate"):
            model.initialize_training()


def test_runtime_context_sync_and_future_protocol(prepared, tmp_path):
    from dexmani_policy.deployment.runtime import inspect_policy, load_policy
    for name in ("lamp", "temporal_codec_flow", "dqrise"):
        cfg = copy.deepcopy(prepared[1][name])
        cfg.workspace.output_dir = str(tmp_path / name)
        comp = build(cfg)
        comp.workspace.save_hydra_config(cfg)
        trainer = build_trainer(cfg, comp); trainer.train(max_updates=1)
        info = inspect_policy(cfg.workspace.output_dir, checkpoint="latest", weights="raw")
        runtime = load_policy(OmegaConf.to_container(cfg), info, device=DEVICE)
        sample = comp.train_loader.dataset[0]
        obs = {k: v.numpy() for k, v in sample["obs"].items()}
        context = None
        if name == "lamp":
            context = {"hand_history": sample["action_context"]["hand_history"].numpy()}
            original = context["hand_history"].copy()
            for invalid in (None, {}, {"unknown": np.zeros((8, 12))},
                            {"hand_history": np.zeros((7, 12))},
                            {"hand_history": np.full((8, 12), np.nan)}):
                with pytest.raises(ValueError):
                    runtime.predict(obs, action_context=invalid)
            with pytest.raises(ValueError, match="inference_steps"):
                inspect_policy(cfg.workspace.output_dir, checkpoint="latest", weights="raw", inference_steps=10)
        result = runtime.predict(obs, action_context=context)
        assert result.shape == (cfg.horizon - cfg.n_obs_steps + 1, 19)
        runtime.configure_execution("sync")
        for mode in ("async", "rtc"):
            if name != "dqrise":
                with pytest.raises(NotImplementedError, match="sync"):
                    runtime.configure_execution(mode, guidance_cap=1.)
        runtime.warmup(samples=1)
        if context is not None:
            np.testing.assert_array_equal(context["hand_history"], original)
        runtime.close(); comp.workspace.close()


def test_runner_only_appends_successful_executed_commands():
    from dexmani_policy.env_runner.base_runner import BaseRunner
    runner = BaseRunner(n_obs_steps=1, default_eval_episodes=1, sensor_modalities=["joint_state"])
    seen = []
    class Agent(torch.nn.Module):
        history_steps, hand_dim, action_dim = 8, 12, 19
        def __init__(self):
            super().__init__(); self.weight = torch.nn.Parameter(torch.zeros(1))
        def predict_action(self, obs_dict, inference_steps=None, *, action_context):
            seen.append(action_context["hand_history"].cpu().clone())
            # A deliberately unused tail tests the runner's executed-step boundary.
            return {"control_action": torch.ones(1, 3, 19)}
    class Env:
        def reset(self, **kwargs):
            self.action_cnt = 0
            return {"joint_state": np.full(19, 2., dtype="f4")}, {}
        def step(self, action):
            self.action_cnt += 1
            return ({"joint_state": np.full(19, 9., dtype="f4")}, 0., True, False,
                    {"executed_joint_action": np.full(19, .7, dtype="f4")})
    agent = Agent(); env = Env()
    for _ in range(2):
        runner.run_one_episode(agent, env, 5)
        np.testing.assert_array_equal(runner._hand_history[:-1], np.full((7, 12), 2.))
        np.testing.assert_allclose(runner._hand_history[-1], .7)
        torch.testing.assert_close(seen[-1], torch.full((1, 8, 12), 2.))
    original = runner._hand_history.copy()
    runner.get_action_chunk(runner.get_obs_batch(device="cpu"), agent)
    np.testing.assert_array_equal(runner._hand_history, original)


@pytest.mark.parametrize("time", [0., 1.])
def test_temporal_noise_to_data_endpoints(prepared, monkeypatch, time):
    cfg = copy.deepcopy(prepared[1]["temporal_codec_flow"])
    dataset, norm = build_dataset_and_normalizer(cfg, verbose=False)
    model, _, _ = build_model_and_ema(cfg, DEVICE, norm, dataset=dataset)
    model.eval()
    batch = batch_for(dataset)
    q0 = batch["action_context"]["hand_reference"]
    with torch.no_grad():
        mu, _ = model.codec.encode(batch["action"][..., 7:], q0)
        arm = model.normalizer["action"].normalize(batch["action"])[..., :7]
        target = torch.cat((arm, mu), dim=-1)
    noise = torch.full_like(target, .3)
    captured = []
    class Oracle(torch.nn.Module):
        def forward(self, x, timestep, context):
            captured.append(x.detach())
            return target - noise
    model.action_decoder.model = Oracle()
    monkeypatch.setattr(torch, "randn_like", lambda value: torch.full_like(value, .3))
    monkeypatch.setattr(model.action_decoder.time_sampler, "sample",
                        lambda count, mode, device: torch.full((count,), time, device=device))
    decoded = []
    hook = model.codec.decoder_input.register_forward_pre_hook(lambda module, args: decoded.append(args[0].detach()))
    _, metrics = model.compute_loss(batch)
    hook.remove()
    assert metrics["loss_arm"] == 0 and metrics["loss_latent"] == 0
    torch.testing.assert_close(captured[0], noise if time == 0 else target)
    torch.testing.assert_close(decoded[0], mu)


def test_historical_lamp_policy_has_explicit_diagnostic(tmp_path):
    path = tmp_path / "historical.pt"
    torch.save({"_format": "simple.v4", "weights": {}, "state": {},
                "effective_config": {
                    "dataset": {"_target_": "dexmani_policy.datasets.lamp_dataset.LampDataset"},
                    "agent": {"_target_": "historical.unavailable.Policy"},
                }}, path)
    with pytest.raises(ValueError, match="historical checkout or retrain"):
        CheckpointStore(tmp_path).load_payload(path)
