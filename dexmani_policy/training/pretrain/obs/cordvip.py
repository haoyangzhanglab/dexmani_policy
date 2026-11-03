# CordViP: xuanxuanzzzii/cordvip@ad6d441abe7b0480cf79b0311db266965c1313b6.

import math

import torch
from torch import nn
from torch.nn import functional as F

from dexmani_policy.agents.modules import build_mlp
from dexmani_policy.agents.obs_encoder.interaction.correspondence import ContactCoordinationEncoder
from dexmani_policy.common.normalizer import LinearNormalizer


class CordViP(nn.Module):
    def __init__(
        self, normalization, n_obs_steps=4,
        contact_loss_weight=1.0, arm_loss_weight=1.0, hand_loss_weight=1.0,
    ):
        super().__init__()
        weights = (contact_loss_weight, arm_loss_weight, hand_loss_weight)
        if any(not math.isfinite(w) or w < 0 for w in weights) or not any(w > 0 for w in weights):
            raise ValueError("Loss weights must be finite, nonnegative, and at least one positive")
        self.encoder = ContactCoordinationEncoder()
        self.n_obs_steps = n_obs_steps
        shapes = {"hand_pointcloud": (3,), "object_pointcloud": (3,), "joint_state": (19,)}
        self.normalizer = LinearNormalizer({
            key: shape for key, shape in shapes.items() if normalization[key] != "identity"
        })

        robot_dim, object_dim, hand_dim, arm_dim = self.encoder.output_dims
        visual_dim = robot_dim + object_dim
        # The pretraining recipe cross-conditions arm and hand prediction.
        self.arm_head = build_mlp(
            [n_obs_steps * (visual_dim + hand_dim), 512, 256, 64, 7], activation=nn.ReLU, normalize=False,
        )
        self.hand_head = build_mlp(
            [n_obs_steps * (visual_dim + arm_dim), 512, 256, 64, 12], activation=nn.ReLU, normalize=False,
        )
        self.contact_head = build_mlp(
            [visual_dim, 2048, 1024, 1024, 1024], activation=nn.ReLU, normalize=False,
        )
        self.contact_loss_weight = contact_loss_weight
        self.arm_loss_weight = arm_loss_weight
        self.hand_loss_weight = hand_loss_weight

    def forward(self, batch):
        obs = {
            key: self.normalizer[key].normalize(value) if key in self.normalizer.fields else value
            for key, value in batch["obs"].items()
        }
        robot_pc, object_pc, hand_state, arm_state = self.encoder(obs)
        loss = robot_pc.new_zeros(())
        metrics = {}
        if self.arm_loss_weight > 0:
            condition = torch.cat((robot_pc.flatten(1), object_pc.flatten(1), hand_state.flatten(1)), dim=1)
            arm_loss = F.mse_loss(self.arm_head(condition), obs["joint_state"][:, self.n_obs_steps - 1, :7])
            loss = loss + self.arm_loss_weight * arm_loss
            metrics["arm"] = arm_loss.detach()
        if self.hand_loss_weight > 0:
            condition = torch.cat((robot_pc.flatten(1), object_pc.flatten(1), arm_state.flatten(1)), dim=1)
            hand_loss = F.mse_loss(self.hand_head(condition), obs["joint_state"][:, self.n_obs_steps - 1, 7:19])
            loss = loss + self.hand_loss_weight * hand_loss
            metrics["hand"] = hand_loss.detach()
        if self.contact_loss_weight > 0:
            contact = self.contact_head(torch.cat((robot_pc, object_pc), dim=-1))
            contact_loss = F.mse_loss(contact, batch["contact_map"])
            loss = loss + self.contact_loss_weight * contact_loss
            metrics["contact"] = contact_loss.detach()
        return loss, {"loss": loss.detach(), **metrics}


def main():
    from dexmani_policy.training.pretrain.obs.common import train

    train("cordvip")


if __name__ == "__main__":
    main()
