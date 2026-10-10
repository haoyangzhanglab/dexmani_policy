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
    GeometryPatchEncoder,
)
from dexmani_policy.agents.obs_encoder.proprio.state_mlp import create_state_mlp
from dexmani_policy.agents.obs_encoder.tactile.xhand_frame import (
    XHandTactileFrameEncoder,
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
        num_patches: int = 128,
        group_size: int = 32,
        patch_depth: int = 4,
        num_scene_tokens: int = 16,
        tactile_input_key: str = "contact_force",
        tactile_fusion: str = "query",
        edge_frame: str = "wrist",
        wrist_pose_key: str = "eef_pose",
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
        if wrist_pose_key not in {"eef_pose", "split"}:
            raise ValueError("wrist_pose_key must be eef_pose (Real) or split (Sim)")
        if not 0 <= tactile_dropout_prob <= 1:
            raise ValueError("tactile_dropout_prob must be in [0,1]")
        self.n_obs_steps = n_obs_steps
        self.obs_token_dim = token_dim
        self.tokens_per_frame = num_scene_tokens + 7
        self.state_dim = state_dim
        self.pc_dim = pc_dim
        self.tactile_input_key = tactile_input_key
        self.tactile_fusion = tactile_fusion
        self.wrist_pose_key = wrist_pose_key
        self.tactile_dropout_prob = tactile_dropout_prob
        self.use_tactile_valid = use_tactile_valid
        self.use_modality_gate = use_modality_gate

        self.geometry = GeometryPatchEncoder(
            input_channels=pc_dim,
            token_channels=token_dim,
            num_patches=num_patches,
            group_size=group_size,
            depth=patch_depth,
            fps_random_config=fps_random_config,
        )
        self.state = create_state_mlp(state_dim, token_dim)
        self.tactile = XHandTactileFrameEncoder(tactile_input_key)
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
        wrist = (
            ("eef_pose",)
            if self.wrist_pose_key == "eef_pose"
            else ("eef_pos", "eef_rot6d")
        )
        fields = [
            "joint_state",
            "point_cloud",
            "fingertip_points",
            *wrist,
            self.tactile_input_key,
            "contact_force",
        ]
        if self.use_tactile_valid:
            fields.append("tactile_valid")
        return tuple(dict.fromkeys(fields))

    @staticmethod
    def _finger_vectors(value, batch, name):
        if value.shape not in {(batch, 15), (batch, 5, 3)}:
            raise ValueError(f"{name} must be [BT,15] or [BT,5,3]")
        return value.reshape(batch, 5, 3)

    def forward(self, obs: dict, *, return_intermediate: bool = False):
        pointcloud = obs["point_cloud"]
        if pointcloud.ndim != 3 or pointcloud.shape[-1] < self.pc_dim:
            raise ValueError("point_cloud must be [BT,N,C] before interaction encoding")
        bt = pointcloud.shape[0]
        if not bt or bt % self.n_obs_steps:
            raise ValueError(
                "observation batch must be nonempty and divisible by n_obs_steps"
            )
        batch = bt // self.n_obs_steps
        state = obs["joint_state"]
        if state.shape != (bt, self.state_dim) or not torch.isfinite(state).all():
            raise ValueError("joint_state must be finite [BT,state_dim]")
        fingers = self._finger_vectors(obs["fingertip_points"], bt, "fingertip_points")
        if self.wrist_pose_key == "eef_pose":
            wrist = obs["eef_pose"]
        else:
            if obs["eef_pos"].shape != (bt, 3) or obs["eef_rot6d"].shape != (bt, 6):
                raise ValueError(
                    "split wrist pose requires eef_pos [BT,3] and eef_rot6d [BT,6]"
                )
            wrist = torch.cat((obs["eef_pos"], obs["eef_rot6d"]), dim=-1)
        if (
            wrist.shape != (bt, 9)
            or not torch.isfinite(wrist).all()
            or not torch.isfinite(fingers).all()
        ):
            raise ValueError(
                "wrist and fingertip geometry must be finite in the common metric frame"
            )

        force = self._finger_vectors(obs["contact_force"], bt, "contact_force")
        valid = torch.ones(bt, 5, dtype=torch.bool, device=pointcloud.device)
        if self.use_tactile_valid:
            valid = obs["tactile_valid"]
            if valid.dtype != torch.bool or valid.shape != (bt, 5):
                raise ValueError(
                    "tactile_valid must be bool [BT,5], not normalized contact scores"
                )
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
            if self.tactile_input_key == "contact_force"
            else obs[self.tactile_input_key]
        )
        frames = self.tactile(tactile_input.unsqueeze(1), valid.unsqueeze(1)).squeeze(1)
        evidence = self.finger_evidence(frames, force, valid)

        state_token = self.state(state)
        patches = self.geometry(pointcloud, return_intermediate=True)
        scene = self.scene(
            patches["patch_token"],
            patches["patch_center"],
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
