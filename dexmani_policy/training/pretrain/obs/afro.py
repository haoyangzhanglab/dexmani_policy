# AFRO: KolaKivy/AFRO@283024a49e2c14bc030debf15383a4e7ee73836d.

import torch
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from torch import nn
from torch.nn import functional as F
from timm.models.vision_transformer import Mlp

from dexmani_policy.agents.action_decoders.backbones.modules import (
    DiTTimestepEmbedding, approximate_gelu, initialize_linear,
)
from dexmani_policy.agents.modules import build_mlp
from dexmani_policy.agents.obs_encoder.pointcloud.pointnet import PointNet
from dexmani_policy.common.normalizer import LinearNormalizer


def vicreg_loss(prediction, target):
    with torch.autocast(device_type=prediction.device.type, enabled=False):
        prediction, target = prediction.float(), target.float()
        invariance = F.mse_loss(prediction, target)
        centered = [value - value.mean(0, keepdim=True) for value in (prediction, target)]
        std = [(value.var(0, unbiased=False) + 1e-4).sqrt() for value in centered]
        variance = 0.5 * (F.relu(1 - std[0]).mean() + F.relu(1 - std[1]).mean())
        covariance = prediction.new_zeros(())
        for value in centered:
            n, dim = value.shape
            if n > 1:
                cov = value.T @ value / (n - 1.0)
                covariance = covariance + (cov.square().sum() - cov.diagonal().square().sum()) / dim
        return 25 * invariance + 25 * variance + covariance, invariance, variance, covariance


class DynamicsBlock(nn.Module):
    def __init__(self, hidden_size, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.value_proj = nn.Linear(hidden_size, hidden_size)
        self.output_proj = nn.Linear(hidden_size, hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
            act_layer=approximate_gelu,
            drop=0,
        )
        self.adaln_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(self, x, c):
        shift_value, scale_value, gate_value, shift_mlp, scale_mlp, gate_mlp = self.adaln_modulation(c).chunk(6, dim=1)
        value = self.value_proj(self.norm1(x) * (1 + scale_value) + shift_value)
        x = x + gate_value * self.output_proj(value)
        x = x + gate_mlp * self.mlp(self.norm2(x) * (1 + scale_mlp) + shift_mlp)
        return x


class ForwardDynamics(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_size=256,
        depth=4,
        mlp_ratio=4.0,
        timestep_embed_dim=256,
    ):
        super().__init__()
        self.noise_proj = nn.Linear(input_dim, hidden_size)
        self.vis_proj = nn.Linear(input_dim, hidden_size)
        self.act_proj = nn.Linear(16, hidden_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, hidden_size))
        self.t_embedder = DiTTimestepEmbedding(hidden_size, frequency_embedding_size=timestep_embed_dim)
        self.ca_proj = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

        self.blocks = nn.ModuleList([
            DynamicsBlock(hidden_size, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.output_layer = nn.Linear(hidden_size, input_dim)

        self.initialize_weights()

    def initialize_weights(self):
        self.apply(initialize_linear)
        for block in self.blocks:
            # Match the V slice's Xavier variance in the former [3H,H] QKV matrix.
            nn.init.xavier_uniform_(block.value_proj.weight, gain=2 ** -0.5)
            nn.init.zeros_(block.adaln_modulation[-1].weight)
            nn.init.zeros_(block.adaln_modulation[-1].bias)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, z, t, vis, act):
        x = self.noise_proj(z) + self.pos_embed

        t_emb = self.t_embedder(t)
        v = self.vis_proj(vis)
        a = self.act_proj(act)
        c = self.ca_proj(torch.cat([t_emb, v, a], dim=-1))

        for block in self.blocks:
            x = block(x, c)

        return self.output_layer(x)


class AFRO(nn.Module):
    def __init__(self, normalization):
        super().__init__()
        self.encoder = PointNet(3, 64, pooling="max")
        self.teacher = PointNet(3, 64, pooling="max")
        self.teacher.requires_grad_(False)
        self.idm = build_mlp([64, 256, 128, 16], activation=nn.GELU, normalize=False)
        self.fdm = ForwardDynamics(64)
        self.normalizer = LinearNormalizer(
            {"point_cloud": (3,)} if normalization["point_cloud"] != "identity" else {}
        )
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=100, beta_schedule="squaredcos_cap_v2",
        )

    def forward(self, batch):
        points = batch["obs"]["point_cloud"]
        if "point_cloud" in self.normalizer.fields:
            points = self.normalizer["point_cloud"].normalize(points)
        batch_size, frames = points.shape[:2]
        points = points.flatten(0, 1)
        with torch.no_grad():
            teacher = self.teacher(points).reshape(batch_size, frames, 64)
        online = self.encoder(points).reshape(batch_size, frames, 64)

        current = torch.cat((online[:, :-1], online[:, 1:])).reshape(-1, 64)
        following = torch.cat((online[:, 1:], online[:, :-1])).reshape(-1, 64)
        latent = self.idm(following - current)
        targets = (teacher[:, 1:].reshape(-1, 64), teacher[:, :-1].reshape(-1, 64))
        noisy, timesteps = [], []
        for target in targets:
            timestep = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps, (len(target),), device=target.device,
            )
            noise = torch.randn_like(target)
            noisy.append(self.noise_scheduler.add_noise(target, noise, timestep))
            timesteps.append(timestep)
        predictions = self.fdm(torch.cat(noisy), torch.cat(timesteps), current, latent).chunk(2)

        # VICReg statistics remain separate for the two prediction directions.
        losses, metrics = [], {}
        for name, prediction, target in zip(("forward", "reverse"), predictions, targets):
            loss, inv, var, cov = vicreg_loss(prediction, target)
            losses.append(loss)
            metrics.update({
                f"{name}_{key}": value.detach()
                for key, value in zip(("invariance", "variance", "covariance"), (inv, var, cov))
            })
        loss = losses[0] + losses[1]
        return loss, {"loss": loss.detach(), **metrics}

    @torch.no_grad()
    def update_teacher(self, epoch):
        momentum = 0.996 + 0.004 * min(epoch, 300) / 300
        for target, online in zip(self.teacher.parameters(), self.encoder.parameters()):
            target.mul_(momentum).add_(online, alpha=1 - momentum)


def main():
    from dexmani_policy.training.pretrain.obs.common import train

    train("afro")


if __name__ == "__main__":
    main()
