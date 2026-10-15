"""身体锚定的几何读取与逐指触觉证据，输出逐帧策略条件。"""

import torch
from torch import nn

from dexmani_policy.agents.obs_encoder.interaction.attention import (
    InteractionSelfAttentionBlock,
)
from dexmani_policy.agents.obs_encoder.interaction.finger_evidence import (
    FingerEvidenceEncoder,
)
from dexmani_policy.agents.obs_encoder.interaction.hand_scene_relation import (
    HandSceneRelationEncoder,
)
from dexmani_policy.agents.obs_encoder.interaction.modality_gate import (
    FingerModalityGate,
)
from dexmani_policy.agents.obs_encoder.interaction.scene_context import SceneContextPool
from dexmani_policy.agents.obs_encoder.pointcloud.geometry_patch import (
    GeometryAwarePointTransformer,
)
from dexmani_policy.agents.obs_encoder.proprio.state_mlp import create_state_mlp
from dexmani_policy.agents.obs_encoder.tactile.xhand_tactile_encoder import (
    XHandForceEncoder,
    XHandTaxelEncoder,
)


class InteractionObsEncoder(nn.Module):
    """Flattened [B*T,...] observations -> [B,T*(S+6+1),D].

    query/late 共享所有参数与 residual/norm 顺序，唯一区别是几何读取
    query 是否含同指触觉。none 保持 token 数，去掉触觉证据。
    输入点云、腕部和指尖必须处于同一米制基坐标系；上游不得独立归一化。
    """

    def __init__(
        self,
        n_obs_steps: int,
        state_dim: int = 19,
        pc_dim: int = 6,
        token_dim: int = 192,
        num_patches: int = 256,
        group_size: int = 32,
        patch_depth: int = 6,
        num_scene_tokens: int = 16,
        tactile_input_key: str = "finger_tactile_force",
        tactile_fusion: str = "query",
        edge_frame: str = "wrist",
        fps_random_config: dict | None = None,
        tactile_dropout_prob: float = 0.0,
        use_tactile_valid: bool = False,
        use_modality_gate: bool = True,
    ):
        super().__init__()
        if n_obs_steps < 1 or num_scene_tokens > num_patches:
            raise ValueError(
                "positive history and num_scene_tokens <= num_patches required"
            )
        if tactile_fusion not in {"query", "late", "none"}:
            raise ValueError("tactile_fusion must be query, late or none")
        if not 0 <= tactile_dropout_prob <= 1:
            raise ValueError("tactile_dropout_prob must be in [0,1]")
        self.n_obs_steps = n_obs_steps
        self.obs_token_dim = token_dim
        self.tokens_per_frame = num_scene_tokens + 7
        self.tactile_input_key = tactile_input_key
        self.tactile_fusion = tactile_fusion
        self.tactile_dropout_prob = tactile_dropout_prob
        self.use_tactile_valid = use_tactile_valid
        self.use_modality_gate = use_modality_gate

        self.geometry = GeometryAwarePointTransformer(
            input_channels=pc_dim,
            num_patches=num_patches,
            group_size=group_size,
            depth=patch_depth,
            fps_random_config=fps_random_config,
        )
        # Project the same spatial memory once for scene and hand readers.
        self.geometry_projection = nn.Sequential(
            nn.Linear(self.geometry.out_dim, token_dim), nn.LayerNorm(token_dim)
        )
        self.state = create_state_mlp(state_dim, token_dim)
        if tactile_input_key == "finger_tactile_force":
            self.tactile = XHandForceEncoder()
        elif tactile_input_key == "taxel_tactile_force":
            self.tactile = XHandTaxelEncoder()
        else:
            raise ValueError("tactile_input_key must be 'finger_tactile_force' or 'taxel_tactile_force'")
        self.finger_evidence = FingerEvidenceEncoder(self.tactile.out_dim, token_dim)
        self.modality_gate = FingerModalityGate(token_dim)
        self.relation = HandSceneRelationEncoder(
            token_channels=token_dim, edge_frame=edge_frame
        )
        self.scene = SceneContextPool(
            token_channels=token_dim, num_scene_tokens=num_scene_tokens
        )
        self.hand_input_norm = nn.LayerNorm(token_dim)
        self.hand_coordination = InteractionSelfAttentionBlock(
            token_dim, 4, 2 * token_dim
        )
        self.hand_output_norm = nn.LayerNorm(token_dim)
        self.token_type = nn.Parameter(torch.empty(3, token_dim))
        nn.init.normal_(self.token_type, std=0.02)

    @property
    def consumed_observation_fields(self) -> tuple[str, ...]:
        fields = [
            "joint_state",
            "point_cloud",
            "fingertip_xyz",
            "eef_pose",
            self.tactile_input_key,
            "finger_tactile_force",
        ]
        if self.use_tactile_valid:
            fields.append("tactile_valid")
        return tuple(dict.fromkeys(fields))

    def forward(self, obs: dict, *, return_intermediate: bool = False):
        pointcloud = obs["point_cloud"]
        bt = pointcloud.shape[0]
        batch = bt // self.n_obs_steps
        state = obs["joint_state"]
        if not torch.isfinite(state).all():
            raise ValueError("joint_state must be finite")
        fingers = obs["fingertip_xyz"]
        wrist = obs["eef_pose"]
        force = obs["finger_tactile_force"]
        # Aggregate force also enters the gate when dense tactile arrays are used.
        if force.shape != (bt, 5, 3):
            raise ValueError("finger_tactile_force must be [BT,5,3]")
        valid = torch.ones(bt, 5, dtype=torch.bool, device=pointcloud.device)
        if self.use_tactile_valid:
            valid = obs["tactile_valid"]
            if valid.dtype != torch.bool or valid.shape != (bt, 5):
                raise ValueError("tactile_valid must be bool [BT,5]")
        if self.training and self.tactile_dropout_prob > 0:
            # One sensor-drop mask per sample/finger, shared by its history frames.
            keep = (
                torch.rand(batch, 1, 5, device=valid.device)
                >= self.tactile_dropout_prob
            )
            valid = valid & keep.expand(-1, self.n_obs_steps, -1).reshape(bt, 5)
        if self.tactile_fusion == "none":
            valid = torch.zeros_like(valid)

        tactile_input = (
            force
            if self.tactile_input_key == "finger_tactile_force"
            else obs[self.tactile_input_key]
        )
        frames = self.tactile(tactile_input.unsqueeze(1), valid.unsqueeze(1)).squeeze(1)
        evidence = self.finger_evidence(frames, force, valid)

        state_token = self.state(state)
        patches = self.geometry(pointcloud, return_geometry=True)
        patches["tokens"] = self.geometry_projection(patches["tokens"])
        scene = self.scene(
            patches["tokens"],
            patches["xyz"],
        )["scene_token"]
        hand_encoding = self.relation.hand_encoder(wrist, fingers)

        # 门控只看读取前的共同证据，绝不使用 Q/L 不同的 relation_update。
        gains = self.modality_gate(
            hand_encoding["hand_token"] + state_token[:, None],
            evidence,
            scene.mean(dim=1),
            force,
            valid,
            enabled=self.use_modality_gate,
        )
        gated_evidence = gains["tactile_gain"] * evidence
        query_context = state_token[:, None].expand(-1, 6, -1)
        if self.tactile_fusion == "query":
            query_context = query_context + gated_evidence
        relation = self.relation(
            pointcloud,
            patches,
            wrist,
            fingers,
            query_context=query_context,
            hand_encoding=hand_encoding,
            return_intermediate=return_intermediate,
        )

        # Q/L share the same direct evidence, normalization and coordination.
        hand = (
            relation["hand_query"]
            + state_token[:, None]
            + gated_evidence
            + gains["geometry_gain"] * relation["relation_update"]
        )
        hand = self.hand_output_norm(self.hand_coordination(self.hand_input_norm(hand)))

        frame_tokens = torch.cat(
            (
                scene + self.token_type[0],
                hand + self.token_type[1],
                state_token[:, None] + self.token_type[2],
            ),
            dim=1,
        )
        cond = frame_tokens.reshape(
            batch, self.n_obs_steps * self.tokens_per_frame, self.obs_token_dim
        )

        aux = {}
        if return_intermediate:
            aux = dict(
                relation,
                **gains,
                finger_evidence=evidence,
                gated_finger_evidence=gated_evidence,
                tactile_valid=valid,
                frame_tokens=frame_tokens,
            )
        return cond, aux
