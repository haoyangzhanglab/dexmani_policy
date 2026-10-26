from __future__ import annotations

import torch
from omegaconf import DictConfig
from torch import nn

from dexmani_policy.agents.action_decoders.backbones.dit import DiT
from dexmani_policy.agents.action_decoders.diffusion import Diffusion
from dexmani_policy.agents.action_decoders.flow import RectifiedFlow
from dexmani_policy.agents.obs_encoder.proprio import StateEncoder
from dexmani_policy.agents.obs_encoder.rgb.build import build_rgb_encoder
from dexmani_policy.agents.obs_encoder.rgb.resnet import ResNetEncoder
from dexmani_policy.agents.obs_encoder.text import CLIPTextEncoder
from dexmani_policy.agents.policies.base import BasePolicy, parameter_groups
from dexmani_policy.common.config import positive_int


def encode_task_embeddings(encoder: CLIPTextEncoder, task_texts: list[str]) -> torch.Tensor:
    return torch.cat(
        [
            encoder(task_texts[start : start + 32]).squeeze(1)
            for start in range(0, len(task_texts), 32)
        ]
    )


class MultiTaskDiTObservationEncoder(nn.Module):
    consumed_observation_fields = ("joint_state", "rgb")

    def __init__(
        self,
        rgb_backbone_name: str,
        state_dim: int,
        n_obs_steps: int,
        state_out_dim: int = 64,
        rgb_backbone_config: dict | None = None,
    ) -> None:
        super().__init__()
        self.backbone, self.image_processor = build_rgb_encoder(
            rgb_backbone_name, config=rgb_backbone_config
        )
        self.state_mlp = StateEncoder(state_dim, state_out_dim)
        self.n_obs_steps = n_obs_steps
        self.out_dim = self.backbone.out_dim + self.state_mlp.out_dim

    @property
    def effective_config(self):
        return {"rgb_backbone_config": self.backbone.effective_config}

    def forward(self, obs: dict) -> torch.Tensor:
        rgb = self.image_processor.process_images(obs["rgb"])

        if isinstance(self.backbone, ResNetEncoder):
            rgb = rgb.to(memory_format=torch.channels_last)

        features = torch.cat(
            [
                self.backbone(rgb)["global_token"],
                self.state_mlp(obs["joint_state"]),
            ],
            dim=-1,
        )
        return features.unflatten(0, (-1, self.n_obs_steps)).flatten(1)


class MultiTaskDiTPolicy(BasePolicy):
    """Language-conditioned RGB policy with cached task embeddings or online frozen CLIP."""

    @classmethod
    def prepare_restore_config(cls, config) -> None:
        rgb = dict(config.rgb_backbone_config)
        if config.rgb_backbone_name == "resnet" and not {"tune_mode", "norm_mode"} <= rgb.keys():
            raise ValueError("Saved ResNet requires explicit tune_mode and norm_mode")
        if config.rgb_backbone_name in {"dino", "siglip"} and not isinstance(
            rgb.get("architecture"), (dict, DictConfig)
        ):
            raise ValueError("Saved HF backbone requires an architecture mapping")
        rgb["load_pretrained"] = False
        config.rgb_backbone_config = rgb
        if config.get("task_texts") is not None:
            positive_int(config.text_embed_dim, "saved text_embed_dim")

    def __init__(
        self,
        text_encoder_model: str = "openai/clip-vit-base-patch16",
        text_embed_dim: int | None = None,
        rgb_backbone_name: str = "dino",
        rgb_backbone_config: dict | None = None,
        state_dim: int = 19,
        state_out_dim: int = 64,
        n_emb: int = 512,
        num_heads: int = 8,
        n_layers: int = 6,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        attn_drop: float = 0.1,
        proj_drop: float = 0.1,
        action_decoder_type: str = "diffusion",
        num_training_steps: int = 100,
        num_inference_steps: int = 10,
        prediction_type: str = "sample",
        flow_num_inference_steps: int = 10,
        flow_t_sample_mode: str = "beta",
        flow_beta_s: float = 0.999,
        flow_beta_alpha: float = 1.0,
        flow_beta_beta: float = 1.5,
        horizon: int = 16,
        n_obs_steps: int = 2,
        n_action_steps: int = 8,
        action_dim: int = 19,
        modality_dropout_probs: dict | None = None,
        task_texts: list[str] | None = None,
        num_flow_train_timesteps: int = 10,
        clip_sample: bool = True,
    ) -> None:
        if action_decoder_type not in ("diffusion", "rectified_flow"):
            raise ValueError("action_decoder_type must be 'diffusion' or 'rectified_flow'")

        text_encoder = (
            CLIPTextEncoder(model_name=text_encoder_model)
            if task_texts is None or text_embed_dim is None
            else None
        )
        text_dim = text_encoder.embed_dim if text_encoder is not None else text_embed_dim
        obs_encoder = MultiTaskDiTObservationEncoder(
            rgb_backbone_name=rgb_backbone_name,
            state_dim=state_dim,
            n_obs_steps=n_obs_steps,
            state_out_dim=state_out_dim,
            rgb_backbone_config=rgb_backbone_config,
        )

        obs_cond_dim = obs_encoder.out_dim * n_obs_steps
        full_cond_dim = obs_cond_dim + n_emb

        backbone = DiT(
            horizon=horizon,
            action_dim=action_dim,
            cond_dim=full_cond_dim,
            n_emb=n_emb,
            num_heads=num_heads,
            n_layers=n_layers,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )

        if action_decoder_type == "diffusion":
            action_decoder: nn.Module = Diffusion(
                backbone,
                num_training_steps=num_training_steps,
                num_inference_steps=num_inference_steps,
                prediction_type=prediction_type,
                clip_sample=clip_sample,
            )
        else:
            action_decoder = RectifiedFlow(
                backbone,
                num_inference_steps=flow_num_inference_steps,
                num_flow_train_timesteps=num_flow_train_timesteps,
                t_sample_mode=flow_t_sample_mode,
                beta_s=flow_beta_s,
                beta_alpha=flow_beta_alpha,
                beta_beta=flow_beta_beta,
            )

        super().__init__(
            obs_encoder,
            action_decoder,
            horizon,
            n_obs_steps,
            n_action_steps,
            action_dim,
            modality_dropout_probs=modality_dropout_probs,
        )

        self.supported_execution_modes = (
            ("sync", "async", "rtc") if action_decoder_type == "diffusion" else ("sync", "async")
        )
        self.text_embed_dim = text_dim
        self.text_encoder_model = text_encoder_model
        self.needs_text_initialization = text_encoder is None
        self.text_proj = nn.Linear(text_dim, n_emb)
        if task_texts is None:
            self.text_encoder = text_encoder
            self.register_buffer("task_emb_table", None)
        else:
            unique = list(dict.fromkeys(task_texts))
            if not unique:
                raise ValueError("Closed task_texts must not be empty")
            if text_encoder is None:
                table = torch.zeros(len(unique), text_dim)
            else:
                table = encode_task_embeddings(text_encoder, unique)
            self.register_buffer("task_emb_table", table)
            self.task_to_idx = {t: i for i, t in enumerate(unique)}

    @property
    def effective_config(self):
        config = super().effective_config
        if self.task_emb_table is not None:
            config["text_embed_dim"] = self.text_embed_dim
        return config

    def initialize_training(self, data_description) -> None:
        if self.needs_text_initialization:
            encoder = CLIPTextEncoder(model_name=self.text_encoder_model)
            if encoder.embed_dim != self.text_embed_dim:
                raise ValueError("Saved text dimension differs from initialization model")
            table = encode_task_embeddings(encoder, list(self.task_to_idx))
            self.task_emb_table.copy_(table.to(self.task_emb_table))
            self.needs_text_initialization = False

    @property
    def consumed_observation_fields(self) -> tuple[str, ...]:
        return (*self.obs_encoder.consumed_observation_fields, "task_text")

    def encode_text(self, task_texts: list[str]) -> torch.Tensor:
        if self.task_emb_table is not None:
            indices = [self.task_to_idx[text] for text in task_texts]
            indices = torch.tensor(indices, device=self.task_emb_table.device)
            embedding = self.task_emb_table[indices]
        else:
            embedding = self.text_encoder(task_texts).squeeze(1)
        return self.text_proj(embedding.to(dtype=self.text_proj.weight.dtype))

    def encode_observation(self, obs_dict: dict) -> torch.Tensor:
        obs_numerical = {k: v for k, v in obs_dict.items() if k not in ("task_text", "task_name")}
        condition = super().encode_observation(obs_numerical)
        text_emb = self.encode_text(obs_dict["task_text"])
        return torch.cat([condition, text_emb.to(condition)], dim=-1)

    def optimizer_groups(
        self, lr: float, obs_lr: float, weight_decay: float, obs_weight_decay: float
    ) -> list[dict]:
        groups = super().optimizer_groups(lr, obs_lr, weight_decay, obs_weight_decay)
        return groups + parameter_groups(self.text_proj, lr=lr, weight_decay=weight_decay)
