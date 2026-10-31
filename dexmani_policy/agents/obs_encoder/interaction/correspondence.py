"""CordViP contact/coordination encoding and cross-attention correspondences."""

import copy

import torch
from torch import nn
from torch.nn import functional as F

from dexmani_policy.agents.obs_encoder.pointcloud.pointnet import PointNet


# Source: xuanxuanzzzii/cordvip@ad6d441abe7b0480cf79b0311db266965c1313b6, train_policy/policy/CordViP/CordViP/CordViP/model/vision/transformer.py.
# Upstream: https://github.com/r-pad/taxpose and https://github.com/WangYueFt/dcp/blob/master/model.py.
class CrossAttentionTransformer(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        attention = MultiHeadedAttention(4, emb_dim)
        feed_forward = PositionwiseFeedForward(emb_dim, 1024)
        self.encoder_attention = attention
        self.encoder_ffn = feed_forward
        self.decoder_attention = copy.deepcopy(attention)
        self.cross_attention = copy.deepcopy(attention)
        self.decoder_ffn = copy.deepcopy(feed_forward)
        self.encoder_norm1 = LayerNorm(emb_dim)
        self.encoder_norm2 = LayerNorm(emb_dim)
        self.encoder_norm = LayerNorm(emb_dim)
        self.decoder_norm1 = LayerNorm(emb_dim)
        self.decoder_norm2 = LayerNorm(emb_dim)
        self.decoder_norm3 = LayerNorm(emb_dim)
        self.decoder_norm = LayerNorm(emb_dim)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        normalized = self.encoder_norm1(context)
        context = context + self.encoder_attention(normalized, normalized, normalized)
        context = self.encoder_norm(context + self.encoder_ffn(self.encoder_norm2(context)))
        normalized = self.decoder_norm1(query)
        query = query + self.decoder_attention(normalized, normalized, normalized)
        query = query + self.cross_attention(self.decoder_norm2(query), context, context)
        return self.decoder_norm(query + self.decoder_ffn(self.decoder_norm3(query)))


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        # Sample std and eps outside the square root are part of the original model.
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class MultiHeadedAttention(nn.Module):
    def __init__(self, heads, features):
        super().__init__()
        self.head_dim = features // heads
        self.heads = heads
        linear = nn.Linear(features, features)
        self.linears = nn.ModuleList([linear, *(copy.deepcopy(linear) for _ in range(3))])

    def forward(self, query, key, value):
        batch_size = query.size(0)
        query, key, value = [
            linear(x).view(batch_size, -1, self.heads, self.head_dim).transpose(1, 2)
            for linear, x in zip(self.linears, (query, key, value))
        ]
        x = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0)
        x = x.transpose(1, 2).reshape(batch_size, -1, self.heads * self.head_dim)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, features, hidden_dim):
        super().__init__()
        self.w_1 = nn.Linear(features, hidden_dim)
        self.w_2 = nn.Linear(hidden_dim, features)

    def forward(self, x):
        return self.w_2(F.relu(self.w_1(x)))


# Source: xuanxuanzzzii/cordvip@ad6d441abe7b0480cf79b0311db266965c1313b6, train_policy/policy/CordViP/CordViP/CordViP/model/vision/encoder.py.
class ContactCoordinationEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_robot = PointNet(3, 1024, pooling="max")
        self.encoder_object = PointNet(3, 1024, pooling="max")
        self.transformer_robot = CrossAttentionTransformer(emb_dim=1024)
        self.transformer_object = CrossAttentionTransformer(emb_dim=1024)
        self.hand_projection = nn.Linear(12, 64)
        self.arm_projection = nn.Linear(7, 64)
        self.attn_hand = CrossAttentionTransformer(emb_dim=64)
        self.attn_arm = CrossAttentionTransformer(emb_dim=64)

    @property
    def output_dims(self) -> tuple[int, int, int, int]:
        return (
            self.encoder_robot.out_dim, self.encoder_object.out_dim,
            self.hand_projection.out_features, self.arm_projection.out_features,
        )

    @property
    def out_dim(self) -> int:
        return sum(self.output_dims)

    def forward(self, observations):
        robot_points = observations["hand_pointcloud"]
        object_points = observations["object_pointcloud"]
        joint_state = observations["joint_state"]
        batch_size = robot_points.shape[0]
        robot_pc = self.encoder_robot(robot_points.flatten(0, 1)).unflatten(0, (batch_size, -1))
        object_pc = self.encoder_object(object_points.flatten(0, 1)).unflatten(0, (batch_size, -1))
        robot_embedding = robot_pc + self.transformer_robot(robot_pc, object_pc)
        object_embedding = object_pc + self.transformer_object(object_pc, robot_pc)
        joint_hand = self.hand_projection(joint_state[:, :, 7:19])
        joint_arm = self.arm_projection(joint_state[:, :, :7])
        hand_embedding = joint_hand + self.attn_hand(joint_hand, joint_arm)
        arm_embedding = joint_arm + self.attn_arm(joint_arm, joint_hand)
        return robot_embedding, object_embedding, hand_embedding, arm_embedding
