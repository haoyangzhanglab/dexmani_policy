"""Actual agent/decoder integration and model-driven training dependencies."""

import copy

import hydra
import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from dexmani_policy.agents.action_decoders.consistency_flow import ConsistencyFlowMatch
from dexmani_policy.agents.action_decoders.rectified_flow import RectifiedFlow
from dexmani_policy.common.normalizer import LinearNormalizer
from dexmani_policy.training.build_utils import build_model_and_ema


def small_config(policy):
    common = dict(
        horizon=4,
        n_obs_steps=2,
        n_action_steps=2,
        action_dim=3,
        pc_dim=6,
        state_dim=3,
        num_points=16,
        state_out_dim=8,
        n_layers=1,
        hidden_dim=32,
        n_head=4,
        mlp_ratio=2,
        p_drop_attn=0.0,
        num_inference_steps=2,
        fps_random_config={
            "use_random": False,
            "use_random_start": False,
            "use_shuffle_output": False,
            "random_noise_scale": 0.0,
        },
    )
    if policy == "maniflow":
        common.update(
            _target_="dexmani_policy.agents.core.maniflow.ManiFlowAgent",
            encoder_type="pointnet_dense",
            timestep_embed_dim=16,
            target_t_embed_dim=16,
            pc_encoder_config={"out_channels": 8, "num_points": 16, "hidden_dims": [8]},
        )
    else:
        common.update(
            _target_="dexmani_policy.agents.core.sat.SATAgent",
            encoder_type="pointnext_tokenizer",
            pc_encoder_config={
                "stem_channels": 8,
                "token_channels": 8,
                "num_patches": 4,
                "patch_radii": [0.5],
                "patch_neighbors": [4],
            },
        )
    return OmegaConf.create(
        {
            "agent": common,
            "action_key": "action",
            "normalization": {
                "joint_state": "limits",
                "point_cloud": "identity",
                "action": "limits",
            },
            "training": {"use_ema": True},
            "ema": {"_target_": "dexmani_policy.training.ema_model.EMAModel"},
        }
    )


def sample_batch(device="cpu"):
    return {
        "obs": {
            "point_cloud": torch.randn(2, 2, 16, 6, device=device),
            "joint_state": torch.randn(2, 2, 3, device=device),
        },
        "action": torch.randn(2, 4, 3, device=device),
    }


def normalizer_for(batch):
    normalizer = LinearNormalizer()
    normalizer.fit(
        {"joint_state": batch["obs"]["joint_state"], "action": batch["action"]}
    )
    return normalizer


@pytest.mark.parametrize("policy", ["sat", "maniflow"])
@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("use_ema", [False, True])
def test_model_driven_ema_builder(policy, rank, use_ema):
    cfg = small_config(policy)
    cfg.training.use_ema = use_ema
    normalizer = normalizer_for(sample_batch())
    if policy == "maniflow" and not use_ema:
        with pytest.raises(ValueError, match="requires training.use_ema"):
            build_model_and_ema(cfg, "cpu", normalizer, rank=rank)
        return
    agent, ema, updater = build_model_and_ema(cfg, "cpu", normalizer, rank=rank)
    assert agent.requires_ema_for_loss == (policy == "maniflow")
    expected = use_ema and (rank == 0 or policy == "maniflow")
    assert (ema is not None) == expected
    assert (updater is not None) == expected
    if policy == "maniflow":
        with pytest.raises(RuntimeError, match="EMA"):
            agent.get_training_loss_kwargs(None)
        assert agent.get_training_loss_kwargs(ema) == {
            "ema_decoder": ema.action_decoder
        }
    else:
        assert agent.get_training_loss_kwargs(None) == {}
    if ema is not None:
        assert not ema.training and not any(p.requires_grad for p in ema.parameters())
        for key, value in agent.state_dict().items():
            torch.testing.assert_close(value, ema.state_dict()[key], rtol=0, atol=0)


@pytest.mark.parametrize("policy", ["sat", "maniflow"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_real_flow_agent_train_predict_checkpoint(policy, device, tmp_path):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    cfg, batch = small_config(policy), sample_batch(device)
    agent, ema, updater = build_model_and_ema(cfg, device, normalizer_for(batch))
    optimizer = torch.optim.AdamW(agent.get_optim_param_groups(1e-3, 1e-3, 0.0, 0.0))
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss, logs = agent(batch, **agent.get_training_loss_kwargs(ema))
        assert torch.isfinite(loss)
        if policy == "maniflow":
            assert logs["has_consistency"] == 1 and logs["consistency_batch_size"] == 1
        loss.backward()
        assert all(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in agent.parameters()
            if p.requires_grad
        )
        optimizer.step()
        updater.step(agent)
    agent.eval()
    for nfe in (1, 2, 4):
        result = agent.predict_action(batch["obs"], denoise_timesteps=nfe)
        assert result["pred_action"].shape == (2, 4, 3)
        assert result["control_action"].shape == (2, 2, 3)
        torch.testing.assert_close(
            result["control_action"], result["pred_action"][:, 1:3]
        )
        assert torch.isfinite(result["pred_action"]).all()
    path = tmp_path / "agent.pt"
    torch.save(agent.state_dict(), path)
    restored = hydra.utils.instantiate(cfg.agent).to(device)
    restored.load_state_dict(torch.load(path, weights_only=True), strict=True)
    restored.eval()
    torch.manual_seed(13)
    expected = agent.predict_action(batch["obs"])
    torch.manual_seed(13)
    actual = restored.predict_action(batch["obs"])
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key])


def test_sat_shuffle_and_transpose(monkeypatch):
    cfg, batch = small_config("sat"), sample_batch()
    agent, _, _ = build_model_and_ema(cfg, "cpu", normalizer_for(batch))
    backbone = agent.action_decoder.model
    forward = backbone.forward
    seen = []

    def record(*args, **kwargs):
        seen.append((kwargs["x"].shape, kwargs.get("shuffle", False)))
        return forward(*args, **kwargs)

    monkeypatch.setattr(backbone, "forward", record)
    agent.train()
    agent.compute_loss(batch)
    assert seen.pop() == (torch.Size([2, 3, 4]), True)
    agent.eval()
    agent.compute_loss(batch)
    assert seen.pop()[1] is False
    agent.predict_action(batch["obs"], denoise_timesteps=2)
    assert len(seen) == 2 and all(not shuffled for _, shuffled in seen)
    modes = []
    monkeypatch.setattr(
        torch, "compile", lambda model, **kwargs: modes.append(kwargs["mode"]) or model
    )
    agent.compile_backbone(mode="reduce-overhead")
    assert modes == ["default"]


@pytest.mark.parametrize("decoder_type", ["diffusion", "rectified_flow"])
def test_multitask_actual_decoder_branch(monkeypatch, decoder_type):
    from dexmani_policy.agents.core import multi_task

    # Replace only the external pretrained encoders. The agent, text projection,
    # normalization, DiT, decoder, optimizer and predictions remain production code.
    class TextEncoder(nn.Module):
        embed_dim = 6

        def __init__(self, **kwargs):
            super().__init__()

        def forward(self, texts):
            return torch.arange(6, dtype=torch.float32).expand(len(texts), -1)

    class ObsEncoder(nn.Module):
        out_dim = 4

        def __init__(self, **kwargs):
            super().__init__()
            self.proj = nn.Linear(3, 4)

        def forward(self, obs):
            return self.proj(obs["joint_state"]).reshape(2, -1), {}

    monkeypatch.setattr(multi_task, "CLIPTextEncoder", TextEncoder)
    monkeypatch.setattr(multi_task, "DPObsEncoder", ObsEncoder)
    agent = multi_task.MultiTaskAgent(
        rgb_backbone_name="resnet",
        state_dim=3,
        n_emb=16,
        num_heads=4,
        n_layers=1,
        mlp_ratio=2,
        attn_drop=0,
        proj_drop=0,
        action_decoder_type=decoder_type,
        horizon=4,
        n_obs_steps=2,
        n_action_steps=2,
        action_dim=3,
        num_training_steps=4,
        num_inference_steps=2,
        flow_num_inference_steps=2,
        task_texts=["pick", "place"],
    )
    assert isinstance(agent.action_decoder, RectifiedFlow) == (
        decoder_type == "rectified_flow"
    )
    batch = sample_batch()
    batch["obs"] = {
        "joint_state": batch["obs"]["joint_state"],
        "task_text": ["pick", "place"],
    }
    agent.load_normalizer_from_dataset(normalizer_for(batch))
    optimizer = torch.optim.AdamW(agent.get_optim_param_groups(1e-3, 1e-3, 0.0, 0.0))
    loss, _ = agent.compute_loss(batch)
    loss.backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in agent.parameters()
        if p.requires_grad
    )
    optimizer.step()
    agent.eval()
    result = agent.predict_action(batch["obs"])
    assert result["control_action"].shape == (2, 2, 3)
    assert torch.isfinite(result["pred_action"]).all()
    with pytest.raises(AssertionError, match="action_decoder_type"):
        multi_task.MultiTaskAgent(action_decoder_type="flowmatch")
