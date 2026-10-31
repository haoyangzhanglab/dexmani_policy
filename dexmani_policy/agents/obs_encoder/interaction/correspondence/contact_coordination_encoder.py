"""CordViP Section III-C with local PointNet and arm7-before-hand12 state layout."""
# Source: xuanxuanzzzii/cordvip@ad6d441abe7b0480cf79b0311db266965c1313b6, train_policy/policy/CordViP/CordViP/CordViP/model/vision/encoder.py.

from torch import nn

from dexmani_policy.agents.obs_encoder.interaction.correspondence.cross_attention_transformer import (
    CrossAttentionTransformer,
)
from dexmani_policy.agents.obs_encoder.pointcloud.pointnet import PointNet


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
