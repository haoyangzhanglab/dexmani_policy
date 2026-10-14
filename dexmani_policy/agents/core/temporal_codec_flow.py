"""Flow matching in a frozen, state-conditioned temporal hand codec.

Paper: https://arxiv.org/abs/2609.19666, Steps 1–2. The codec keeps all
horizon frames; only the hand coordinates are compressed.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from dexmani_policy.agents.action_decoders.backbone.ditx import DiTX
from dexmani_policy.agents.action_decoders.rectified_flow import RectifiedFlow
from dexmani_policy.agents.action_decoders.time_sampler import shift_time_to_noise
from dexmani_policy.agents.core.base import BaseAgent
from dexmani_policy.agents.core.sat import SATObsEncoder
from dexmani_policy.agents.hand_prior.temporal_codec import TemporalHandCodec, load_codec
from dexmani_policy.utils.config import validate_window_contract


class TemporalCodecFlowAgent(BaseAgent):
    def __init__(
        self,
        horizon: int,
        n_obs_steps: int,
        n_action_steps: int,
        action_dim: int,
        codec_path: str,
        latent_dim: int = 9,
        codec_hidden_dim: int = 128,
        codec_n_layers: int = 3,
        codec_n_heads: int = 4,
        codec_mlp_ratio: float = 2.0,
        pc_dim: int = 6,
        state_dim: int = 19,
        num_points: int = 1024,
        state_out_dim: int = 64,
        pc_encoder_config: dict | None = None,
        fps_random_config: dict | None = None,
        n_layers: int = 8,
        hidden_dim: int = 768,
        n_head: int = 8,
        mlp_ratio: float = 4.0,
        p_drop_attn: float = 0.1,
        qkv_bias: bool = True,
        qk_norm: bool = True,
        num_inference_steps: int = 10,
        t_sample_mode_for_flow: str = "beta",
        beta_s: float = 0.999,
        beta_alpha: float = 1.0,
        beta_beta: float = 1.5,
        num_flow_train_timesteps: int = 10,
        arm_loss_weight: float = 1.0,
        latent_loss_weight: float = 0.25,
        decoded_loss_weight: float = 0.75,
        modality_dropout_probs: dict | None = None,
        decoded_loss_space: str = "standardized",
    ):
        validate_window_contract(horizon, n_obs_steps, n_action_steps)
        if action_dim != 19 or state_dim != 19:
            raise ValueError("Temporal codec currently requires joint actions/state: 7 arm + 12 hand")
        if decoded_loss_space not in ("standardized", "physical"):
            raise ValueError("decoded_loss_space must be 'standardized' or 'physical'")
        for weight in (arm_loss_weight, latent_loss_weight, decoded_loss_weight):
            if not 0 <= weight < float("inf"):
                raise ValueError("Temporal codec loss weights must be finite and nonnegative")
        obs_encoder = SATObsEncoder(
            encoder_type="pointnext_tokenizer", pc_dim=pc_dim, state_dim=state_dim,
            num_points=num_points, n_obs_steps=n_obs_steps, state_out_dim=state_out_dim,
            pc_encoder_config=pc_encoder_config, fps_random_config=fps_random_config,
        )
        backbone = DiTX(
            horizon=horizon, action_dim=7 + latent_dim, n_obs_steps=n_obs_steps,
            obs_token_dim=obs_encoder.obs_token_dim // n_obs_steps,
            n_layers=n_layers, hidden_dim=hidden_dim, n_head=n_head, mlp_ratio=mlp_ratio,
            p_drop_attn=p_drop_attn, qkv_bias=qkv_bias, qk_norm=qk_norm,
        )
        flow = RectifiedFlow(
            backbone, num_inference_steps=num_inference_steps,
            t_sample_mode=t_sample_mode_for_flow, beta_s=beta_s,
            beta_alpha=beta_alpha, beta_beta=beta_beta,
            num_flow_train_timesteps=num_flow_train_timesteps,
        )
        super().__init__(obs_encoder, flow, horizon, n_obs_steps, n_action_steps,
                         action_dim, modality_dropout_probs)
        self.arm_dim, self.hand_dim = 7, 12
        self.latent_dim = latent_dim
        self.model_action_dim = self.arm_dim + latent_dim
        self.codec_path = codec_path
        self.arm_loss_weight = arm_loss_weight
        self.latent_loss_weight = latent_loss_weight
        self.decoded_loss_weight = decoded_loss_weight
        self.decoded_loss_space = decoded_loss_space
        self.codec = TemporalHandCodec(
            hand_dim=12, horizon=horizon, latent_dim=latent_dim,
            hidden_dim=codec_hidden_dim, n_layers=codec_n_layers,
            n_heads=codec_n_heads, mlp_ratio=codec_mlp_ratio,
        )
        self.codec.freeze()
        self.register_load_state_dict_post_hook(self._check_restored_codec)

    def initialize_training(self):
        codec, _ = load_codec(self.codec_path)
        if codec.model_config != self.codec.model_config:
            raise ValueError("Temporal codec architecture differs from the policy configuration")
        self.codec.load_state_dict(codec.state_dict(), strict=True)

    def _check_restored_codec(self, module, incompatible_keys):
        self.codec.validate_state()

    def train(self, mode=True):
        super().train(mode)
        self.codec.eval()
        return self

    def _condition(self, obs):
        if not self.codec.ready.item():
            raise RuntimeError("Initialize the temporal codec or restore a complete policy checkpoint")
        cond, aux = self._build_cond(obs)
        b, patches, _ = cond.shape
        cond = cond.reshape(b, patches, self.n_obs_steps, -1).transpose(1, 2)
        return cond.reshape(b, self.n_obs_steps * patches, -1), aux

    def _hand_reference(self, obs, action_context, *, required=False):
        reference = (action_context or {}).get("hand_reference")
        if reference is None and not required:
            reference = obs["joint_state"][:, self.n_obs_steps - 1, self.arm_dim:]
        if (not isinstance(reference, torch.Tensor)
                or reference.shape != (obs["joint_state"].shape[0], self.hand_dim)
                or not torch.isfinite(reference).all()):
            raise ValueError("Temporal codec requires raw action_context.hand_reference [B,12]")
        return reference.float()

    def compute_loss(self, batch, **kwargs):
        self._validate_batch(batch)
        # Dataset supplies the measured query state before observation augmentation.
        reference = self._hand_reference(batch["obs"], batch.get("action_context"), required=True)
        cond, aux = self._condition(batch["obs"])
        actions = self.normalizer["action"].normalize(batch["action"])
        if not torch.isfinite(actions).all():
            raise ValueError("Temporal codec action targets must be finite")
        with torch.no_grad(), torch.autocast(device_type=actions.device.type, enabled=False):
            mu, _ = self.codec.encode(batch["action"][..., self.arm_dim:].float(), reference)
        target = torch.cat((actions[..., :self.arm_dim], mu), dim=-1)

        flow = self.action_decoder
        noise = torch.randn_like(target)
        t = flow.time_sampler.sample(len(target), flow.t_sample_mode, device=target.device)
        t = shift_time_to_noise(t, flow.time_shift_alpha)
        weight = t[:, None, None]
        xt = (1 - weight) * noise + weight * target
        pred_v = flow.model(x=xt, timestep=t, context=cond).float()
        target_v = target - noise
        arm_loss = F.mse_loss(pred_v[..., :self.arm_dim], target_v[..., :self.arm_dim])
        latent_loss = F.mse_loss(pred_v[..., self.arm_dim:], target_v[..., self.arm_dim:])
        flow_loss = self.arm_loss_weight * arm_loss + self.latent_loss_weight * latent_loss
        # Our flow goes noise -> data, reversing the paper's time coordinate.
        clean = xt + (1 - weight) * pred_v
        with torch.autocast(device_type=actions.device.type, enabled=False):
            decoded = self.codec.decode(clean[..., self.arm_dim:].float(), reference)
            hand_target = self.codec.normalize_actions(batch["action"][..., self.arm_dim:].float(), reference)
            if self.decoded_loss_space == "physical":
                # The common q0 and mean cancel in physical command error.
                error = (decoded - hand_target) * self.codec.action_std
                decoded_loss = error.square().mean()
            else:
                decoded_loss = F.mse_loss(decoded, hand_target)
        loss = flow_loss + self.decoded_loss_weight * decoded_loss
        metrics = dict(loss=loss, loss_action=flow_loss, loss_arm=arm_loss,
                       loss_latent=latent_loss, loss_decoded_hand=decoded_loss,
                       t_mean=t.mean())
        return self._merge_aux_loss(loss, metrics, aux)

    @torch.no_grad()
    def predict_action(self, obs_dict, inference_steps=None, *, action_context=None,
                       rtc_prefix=None, delay_steps=0, rtc_guidance_cap=0.0):
        if rtc_prefix is not None or delay_steps or rtc_guidance_cap:
            raise NotImplementedError("Temporal codec currently supports synchronous chunk execution")
        self._validate_obs_dict(obs_dict)
        reference = self._hand_reference(obs_dict, action_context)
        cond, _ = self._condition(obs_dict)
        return self.predict_action_from_cond(cond, inference_steps, hand_reference=reference)

    @torch.no_grad()
    def predict_action_from_cond(self, cond, inference_steps=None, *, hand_reference):
        template = torch.zeros(len(cond), self.horizon, self.model_action_dim,
                               device=cond.device, dtype=torch.float32)
        native = self.action_decoder.predict_action(cond, template, inference_steps)
        affine = self.normalizer["action"].params_dict
        scale = affine["scale"].expand(self.action_dim)[:self.arm_dim]
        offset = affine["offset"].expand(self.action_dim)[:self.arm_dim]
        arm = (native[..., :self.arm_dim] - offset) / scale
        with torch.autocast(device_type=native.device.type, enabled=False):
            hand = self.codec.decode_actions(native[..., self.arm_dim:].float(), hand_reference.float())
        pred = torch.cat((arm, hand), dim=-1)
        # Bidirectional codec attention requires decoding the full horizon first.
        start = self.n_obs_steps - 1
        return {"pred_action": pred,
                "control_action": pred[:, start:start + self.n_action_steps],
                "tail": pred[:, start + self.n_action_steps:]}
