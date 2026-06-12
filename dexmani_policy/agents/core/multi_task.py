import torch
import torch.nn as nn
from dexmani_policy.agents.core.base import BaseAgent
from dexmani_policy.agents.core.dp import DPObsEncoder
from dexmani_policy.agents.obs_encoder.text.clip import CLIPTextEncoder
from dexmani_policy.agents.action_decoders.backbone.dit import DiT_Diffusion
from dexmani_policy.agents.action_decoders.diffusion import Diffusion
from dexmani_policy.agents.action_decoders.flowmatch import FlowMatch


class MultiTaskAgent(BaseAgent):
    """多任务 Agent：RGB + joint_state + 自然语言 task_text → DiT backbone。

    obs_dict 中需包含 'task_text' key（字符串列表），由 MultiTaskDataset 注入。

    如果提供了 task_texts 列表，会在 __init__ 中预计算所有 CLIP text embedding
    并存入 buffer，训练/推理时按字符串 key 查表，仅 text_proj（Linear）可训练。
    未缓存的文本退回实时 CLIP 编码。
    """

    def __init__(
        self,
        # text encoder
        text_encoder_model: str = "openai/clip-vit-base-patch16",
        # obs encoder (RGB + state)
        rgb_backbone_name: str = "dino",
        rgb_backbone_config: dict = None,
        state_dim: int = 19,
        state_out_dim: int = 64,
        # backbone (DiT)
        n_emb: int = 512,
        num_heads: int = 8,
        n_layers: int = 6,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        # action decoder type
        action_decoder_type: str = "diffusion",  # "diffusion" | "flowmatch"
        # action decoder (Diffusion)
        num_training_steps: int = 100,
        num_inference_steps: int = 10,
        prediction_type: str = "sample",
        # action decoder (FlowMatch)
        flow_num_inference_steps: int = 10,
        flow_t_sample_mode: str = "beta",
        flow_beta_s: float = 0.999,
        flow_beta_alpha: float = 1.0,
        flow_beta_beta: float = 1.5,
        # common
        horizon: int = 16,
        n_obs_steps: int = 2,
        n_action_steps: int = 8,
        action_dim: int = 19,
        modality_dropout_probs: dict = None,
        # text embedding cache (可选)
        task_texts: list = None,
    ):
        assert rgb_backbone_name in ("resnet", "clip", "dino", "siglip", "r3m"), \
            f"rgb_backbone_name must be one of resnet/clip/dino/siglip/r3m, got {rgb_backbone_name}"
        assert action_decoder_type in ("diffusion", "flowmatch"), \
            f"action_decoder_type must be 'diffusion' or 'flowmatch', got {action_decoder_type}"

        text_encoder = CLIPTextEncoder(model_name=text_encoder_model)
        obs_encoder = DPObsEncoder(
            rgb_backbone_name=rgb_backbone_name,
            state_dim=state_dim,
            n_obs_steps=n_obs_steps,
            state_out_dim=state_out_dim,
            rgb_backbone_config=rgb_backbone_config,
        )

        obs_cond_dim = obs_encoder.out_dim * n_obs_steps
        full_cond_dim = obs_cond_dim + n_emb

        backbone = DiT_Diffusion(
            horizon=horizon,
            action_dim=action_dim,
            cond_dim=full_cond_dim,
            n_emb=n_emb,
            num_heads=num_heads,
            n_layers=n_layers,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
        )

        if action_decoder_type == "diffusion":
            action_decoder: nn.Module = Diffusion(
                backbone,
                num_training_steps=num_training_steps,
                num_inference_steps=num_inference_steps,
                prediction_type=prediction_type,
            )
        elif action_decoder_type == "flowmatch":
            action_decoder = FlowMatch(
                backbone,
                num_inference_steps=flow_num_inference_steps,
                t_sample_mode=flow_t_sample_mode,
                beta_s=flow_beta_s,
                beta_alpha=flow_beta_alpha,
                beta_beta=flow_beta_beta,
            )

        super().__init__(
            obs_encoder, action_decoder, horizon,
            n_obs_steps, n_action_steps, action_dim,
            modality_dropout_probs=modality_dropout_probs,
        )

        self.text_encoder = text_encoder
        self.text_encoder.requires_grad_(False)
        self.text_proj = nn.Linear(text_encoder.embed_dim, n_emb)

        if task_texts is not None:
            self.init_text_cache(task_texts)
        else:
            self.register_buffer("task_emb_table", None)

    def init_text_cache(self, task_texts: list):
        unique = list(dict.fromkeys(task_texts))
        with torch.no_grad():
            embs = [self.text_encoder([t]).squeeze() for t in unique]
        self.register_buffer("task_emb_table", torch.stack(embs))
        self.task_to_idx = {t: i for i, t in enumerate(unique)}

    def get_text_emb(self, task_texts):
        if self.task_emb_table is not None:
            indices = [self.task_to_idx.get(t) for t in task_texts]
            if all(i is not None for i in indices):
                idx = torch.tensor(indices, device=self.task_emb_table.device)
                return self.text_proj(self.task_emb_table[idx].to(dtype=self.text_proj.weight.dtype))
        emb = self.text_encoder(task_texts).squeeze(1)
        return self.text_proj(emb.to(dtype=self.text_proj.weight.dtype))

    def _build_cond(self, obs_dict):
        task_texts = obs_dict["task_text"]
        obs_numerical = {k: v for k, v in obs_dict.items() if k not in ("task_text", "task_name")}
        obs = self.preprocess(obs_numerical)
        cond, aux = self.obs_encoder(obs)
        text_emb = self.get_text_emb(task_texts)
        cond = torch.cat([cond, text_emb.to(device=cond.device, dtype=cond.dtype)], dim=-1)
        return cond, aux

    def get_optim_param_groups(self, lr, obs_lr, weight_decay, obs_wd):
        groups = super().get_optim_param_groups(lr, obs_lr, weight_decay, obs_wd)
        groups.append({"params": [self.text_proj.weight], "weight_decay": weight_decay, "lr": lr})
        if self.text_proj.bias is not None:
            groups.append({"params": [self.text_proj.bias], "weight_decay": 0.0, "lr": lr})
        return groups


def example():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, T, H, A = 2, 2, 16, 19

    task_texts = ["pick apple", "place cube"]

    agent = MultiTaskAgent(
        rgb_backbone_name="resnet",
        state_dim=A,
        n_emb=128, num_heads=4, n_layers=2, mlp_ratio=2.0,
        num_training_steps=10, num_inference_steps=3,
        horizon=H, n_obs_steps=T, n_action_steps=8, action_dim=A,
    ).to(device)

    obs = {
        "rgb": torch.rand(B * T, 3, 224, 224, device=device),
        "joint_state": torch.randn(B * T, A, device=device),
    }
    action = torch.randn(B, H, A, device=device)

    cond, _ = agent.obs_encoder(obs)
    print(f"obs_cond shape: {cond.shape}")

    text_emb = agent.get_text_emb(task_texts)
    print(f"text_emb shape: {text_emb.shape}")
    merged = torch.cat([cond, text_emb.to(device=cond.device, dtype=cond.dtype)], dim=-1)
    print(f"merged_cond shape: {merged.shape}")

    from dexmani_policy.common.normalizer import LinearNormalizer
    normalizer = LinearNormalizer()
    normalizer.fit({"action": action, "joint_state": obs["joint_state"].reshape(B, T, A)}, mode="limits")
    agent.load_normalizer_from_dataset(normalizer)

    batch = {
        "obs": {
            "rgb": obs["rgb"].reshape(B, T, 3, 224, 224),
            "joint_state": obs["joint_state"].reshape(B, T, A),
            "task_text": task_texts,
            "task_name": task_texts,
        },
        "action": action,
    }
    loss, loss_dict = agent.compute_loss(batch)
    print(f"loss: {loss.item():.4f}  keys={list(loss_dict.keys())}")

    result = agent.predict_action({
        "rgb": obs["rgb"].reshape(B, T, 3, 224, 224),
        "joint_state": obs["joint_state"].reshape(B, T, A),
        "task_text": task_texts,
    })
    print(f"pred_action: {result['pred_action'].shape}")
    print(f"control_action: {result['control_action'].shape}")
    print("=== MultiTaskAgent PASSED ===")


if __name__ == "__main__":
    example()
