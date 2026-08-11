"""Hand FK Generator — pytorch_kinematics FK + mesh sampling.

Generates the hand surface point cloud and multi-scale KPConv data structure
from joint angles.  Uses ``pytorch_kinematics`` (Chain, not SerialChain) for
forward kinematics, matching the official OPFA GaLR training pipeline exactly.

**GPU support**: the ``pytorch_kinematics`` chain lives on GPU via ``.to(device)``.
``forward(angles)`` accepts ``(12,)`` (single frame) or ``(B, 12)`` (batched).

=== XHand joint order (VAE order = DexMani native order) ===
  0: right_hand_thumb_bend_joint   (thumb abduction)
  1: right_hand_thumb_rota_joint1  (thumb rotation prox)
  2: right_hand_thumb_rota_joint2  (thumb rotation dist)
  3: right_hand_index_bend_joint   (index abduction)
  4: right_hand_index_joint1       (index prox)
  5: right_hand_index_joint2       (index dist)
  6: right_hand_mid_joint1         (middle prox)
  7: right_hand_mid_joint2         (middle dist)
  8: right_hand_ring_joint1        (ring prox)
  9: right_hand_ring_joint2        (ring dist)
 10: right_hand_pinky_joint1       (pinky prox)
 11: right_hand_pinky_joint2       (pinky dist)

=== Finger/link PE mapping (from OPFA hands.py) ===
  finger 0 = palm:  right_hand_link, ee_link, back_link
  finger 1 = thumb: bend_link(0), rota_link1(1), rotaback_link1(1),
                     rota_link2(1), rotaback_link2(2), rota_tip(2)
  finger 2 = index: bend(0), rota_link1(1), rotaback_link1(1),
                     rota_link2(2), rotaback_link2(2), rota_tip(2)
  finger 3 = middle: link1(0), back_link1(0), link2(1), back_link2(1), tip(1)
  finger 4 = ring:   link1(0), back_link1(0), link2(1), back_link2(1), tip(1)
  finger 5 = pinky:  link1(0), back_link1(0), link2(1), back_link2(1), tip(1)
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import pytorch_kinematics as pk

# Use official OPFA C++ extension when available; fall back to pure-PyTorch.
try:
    from dexmani_policy.agents.opfa._geotransformer_bridge import (
        _ensure_geotransformer,
        grid_subsample,
        radius_search,
    )
    _ensure_geotransformer()
except ImportError:
    from dexmani_policy.agents.opfa.point_ops import (  # noqa: F811
        grid_subsample,
        radius_search,
    )


# =============================================================================
# URDF joint parameter
# =============================================================================


@dataclass
class JointParam:
    """Parsed URDF joint parameters needed for FK."""

    name: str
    parent: str
    child: str
    joint_type: str  # "revolute" | "fixed"
    origin_xyz: np.ndarray  # (3,) translation
    origin_rpy: np.ndarray  # (3,) roll-pitch-yaw in radians
    axis_xyz: np.ndarray  # (3,) rotation axis


# =============================================================================
# URDF parser (stdlib only)
# =============================================================================


def _parse_vector(s: str) -> np.ndarray:
    return np.array([float(x) for x in s.split()], dtype=np.float32)


def parse_xhand_urdf(urdf_path: str) -> OrderedDict[str, JointParam]:
    """Parse URDF and return joint parameters in topological order (root→leaves).

    Uses ``xml.etree.ElementTree`` — no external deps.
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    # Build joint parameter mapping
    joints: dict[str, JointParam] = {}

    for j_elem in root.findall("joint"):
        name = j_elem.attrib["name"]
        jtype = j_elem.attrib["type"]

        origin = j_elem.find("origin")
        if origin is not None:
            xyz = _parse_vector(origin.attrib.get("xyz", "0 0 0"))
            rpy = _parse_vector(origin.attrib.get("rpy", "0 0 0"))
        else:
            xyz = np.zeros(3, dtype=np.float32)
            rpy = np.zeros(3, dtype=np.float32)

        axis_elem = j_elem.find("axis")
        if axis_elem is not None:
            axis = _parse_vector(axis_elem.attrib.get("xyz", "1 0 0"))
        else:
            axis = np.zeros(3, dtype=np.float32)

        parent = j_elem.find("parent").attrib["link"]
        child = j_elem.find("child").attrib["link"]

        joints[name] = JointParam(
            name=name,
            parent=parent,
            child=child,
            joint_type=jtype,
            origin_xyz=xyz,
            origin_rpy=rpy,
            axis_xyz=axis,
        )

    # Topological sort: start from root link, BFS to leaves
    root_link = "right_hand_link"
    # Build adjacency: parent → [(joint_name, child_link)]
    edges: dict[str, list[tuple[str, str]]] = {root_link: []}
    for j in joints.values():
        edges.setdefault(j.parent, []).append((j.name, j.child))
        edges.setdefault(j.child, [])

    ordered: OrderedDict[str, JointParam] = OrderedDict()
    visited = {root_link}
    queue = [root_link]

    while queue:
        parent = queue.pop(0)
        for jname, child in edges.get(parent, []):
            if child not in visited:
                visited.add(child)
                queue.append(child)
                if jname in joints:
                    ordered[jname] = joints[jname]

    return ordered


# =============================================================================
# HandFKGenerator
# =============================================================================


class HandFKGenerator(nn.Module):
    """Generate hand surface point cloud + KPConv data from joint angles.

    Path: joint angles → pytorch_kinematics FK → transform canonical link PCs
          → finger/link PE → grid subsample + radius search → KPConv data dict

    Uses ``pytorch_kinematics.Chain`` for FK, matching the official OPFA GaLR
    training pipeline exactly.  KPConv parameters default to the GaLR training
    values (config.py:133-143).

    **GPU-ready**: chain and all internal tensors are on GPU via ``.to(device)``.
    ``forward()`` supports ``(12,)`` (single frame) or ``(B, 12)`` (batched).

    **Cache**: an optional ``angle → data_dict`` cache avoids recomputing
    FK + KPConv data for repeated joint configurations (common in trajectories).

    Args:
        urdf_path: path to ``xhand_right.urdf``.
        mesh_dir: directory containing ``right_hand_*.STL`` files.
        num_stages: KPConv pyramid depth (default 4).
        init_voxel_size: initial grid sub-sampling size (doubled per stage).
        init_radius: initial ball-query radius (doubled per stage).
        neighbor_limits: max neighbours per KPConv stage.
        sample_points: total points per link for canonical PC.
        cache_size: max entries in angle→data_dict cache (0 = disabled).
        cache_tolerance: rounding tolerance in radians for cache key (default 0.001).
    """

    _DEFAULT_VAE_ORDER = [
        "right_hand_thumb_bend_joint",
        "right_hand_thumb_rota_joint1",
        "right_hand_thumb_rota_joint2",
        "right_hand_index_bend_joint",
        "right_hand_index_joint1",
        "right_hand_index_joint2",
        "right_hand_mid_joint1",
        "right_hand_mid_joint2",
        "right_hand_ring_joint1",
        "right_hand_ring_joint2",
        "right_hand_pinky_joint1",
        "right_hand_pinky_joint2",
    ]

    _FINGER_LINK_INDICES = {
        "right_hand_link": (0, 0),
        "right_hand_ee_link": (0, 0),
        "right_hand_back_link": (0, 0),
        "right_hand_thumb_bend_link": (1, 0),
        "right_hand_thumb_rota_link1": (1, 1),
        "right_hand_thumb_rotaback_link1": (1, 1),
        "right_hand_thumb_rota_link2": (1, 1),
        "right_hand_thumb_rotaback_link2": (1, 2),
        "right_hand_thumb_rota_tip": (1, 2),
        "right_hand_index_bend_link": (2, 0),
        "right_hand_index_rota_link1": (2, 1),
        "right_hand_index_rotaback_link1": (2, 1),
        "right_hand_index_rota_link2": (2, 2),
        "right_hand_index_rotaback_link2": (2, 2),
        "right_hand_index_rota_tip": (2, 2),
        "right_hand_mid_link1": (3, 0),
        "right_hand_midback_link1": (3, 0),
        "right_hand_mid_link2": (3, 1),
        "right_hand_midback_link2": (3, 1),
        "right_hand_mid_tip": (3, 1),
        "right_hand_ring_link1": (4, 0),
        "right_hand_ringback_link1": (4, 0),
        "right_hand_ring_link2": (4, 1),
        "right_hand_ringback_link2": (4, 1),
        "right_hand_ring_tip": (4, 1),
        "right_hand_pinky_link1": (5, 0),
        "right_hand_pinkyback_link1": (5, 0),
        "right_hand_pinky_link2": (5, 1),
        "right_hand_pinkyback_link2": (5, 1),
        "right_hand_pinky_tip": (5, 1),
    }

    _LINK_NAMES = [
        "right_hand_link",
        "right_hand_ee_link",
        "right_hand_back_link",
        "right_hand_thumb_bend_link",
        "right_hand_thumb_rota_link1",
        "right_hand_thumb_rotaback_link1",
        "right_hand_thumb_rota_link2",
        "right_hand_thumb_rotaback_link2",
        "right_hand_thumb_rota_tip",
        "right_hand_index_bend_link",
        "right_hand_index_rota_link1",
        "right_hand_index_rotaback_link1",
        "right_hand_index_rota_link2",
        "right_hand_index_rotaback_link2",
        "right_hand_index_rota_tip",
        "right_hand_mid_link1",
        "right_hand_midback_link1",
        "right_hand_mid_link2",
        "right_hand_midback_link2",
        "right_hand_mid_tip",
        "right_hand_ring_link1",
        "right_hand_ringback_link1",
        "right_hand_ring_link2",
        "right_hand_ringback_link2",
        "right_hand_ring_tip",
        "right_hand_pinky_link1",
        "right_hand_pinkyback_link1",
        "right_hand_pinky_link2",
        "right_hand_pinkyback_link2",
        "right_hand_pinky_tip",
    ]

    def __init__(
        self,
        urdf_path: str | None = None,
        mesh_dir: str | None = None,
        num_stages: int = 4,
        init_voxel_size: float = 0.01,
        init_radius: float = 0.025,
        neighbor_limits: list[int] | None = None,
        sample_points: int = 256,
        cache_size: int = 0,
        cache_tolerance: float = 0.001,
    ):
        super().__init__()

        if neighbor_limits is None:
            neighbor_limits = [30, 15, 10, 5]

        self.num_stages = num_stages
        self.init_voxel_size = init_voxel_size
        self.init_radius = init_radius
        self.neighbor_limits = neighbor_limits
        self.cache_tolerance = cache_tolerance

        # Resolve default paths
        if urdf_path is None:
            urdf_path = self._default_urdf_path()
        if mesh_dir is None:
            mesh_dir = os.path.join(os.path.dirname(urdf_path), "meshes")

        self.urdf_path = urdf_path
        self.mesh_dir = mesh_dir

        # Build pytorch_kinematics chain (matching official OPFA):
        #   pk.build_chain_from_urdf(data) → Chain (NOT SerialChain)
        #   Chain.forward_kinematics(th) → dict[str, Transform3D] for ALL 30 frames
        with open(urdf_path, 'rb') as f:
            self._pk_chain = pk.build_chain_from_urdf(f.read())

        # Parse URDF → FK bookkeeping (joint order, topology — needed by other methods)
        joint_params = parse_xhand_urdf(urdf_path)
        self._init_fk_chain(joint_params)

        # Load canonical link point clouds from STL meshes
        self._init_canonical_pcs(mesh_dir, sample_points)

        # Angle → data_dict cache (LRU via OrderedDict, with maxsize)
        self._cache_enabled = cache_size > 0
        self._cache_maxsize = cache_size
        self._cache: OrderedDict | None = OrderedDict() if self._cache_enabled else None
        self._cache_hits = 0
        self._cache_misses = 0

    # -----------------------------------------------------------------
    # Device movement — pk chain is a plain attribute, not a sub-module
    # -----------------------------------------------------------------

    # -----------------------------------------------------------------
    # Device movement — pk chain is a plain attribute, not a sub-module.
    # PyTorch's nn.Module.cpu() / cuda() call _apply() directly, bypassing
    # to(), so we must override all three.
    # -----------------------------------------------------------------

    @staticmethod
    def _resolve_to_args(*args, **kwargs) -> tuple[torch.device | None, torch.dtype | None]:
        """Extract (device, dtype) from ``to(*args, **kwargs)`` arguments."""
        device = kwargs.get("device", None)
        dtype = kwargs.get("dtype", None)
        for a in args:
            if isinstance(a, torch.device):
                device = a
            elif isinstance(a, torch.dtype):
                dtype = a
            elif isinstance(a, str):
                try:
                    device = torch.device(a)
                except Exception:
                    pass
        return device, dtype

    def _move_pk_chain(self, device: torch.device | None, dtype: torch.dtype | None = None):
        """Move pk chain to *device* (and optionally *dtype*) if not already there."""
        if device is not None:
            self._pk_chain = self._pk_chain.to(device=device, dtype=dtype)

    def to(self, *args, **kwargs):
        """Move pk chain to target device alongside registered buffers."""
        device, dtype = self._resolve_to_args(*args, **kwargs)
        super().to(*args, **kwargs)
        self._move_pk_chain(device, dtype)
        return self

    def cpu(self):
        """Move pk chain to CPU alongside registered buffers."""
        super().cpu()
        self._move_pk_chain(torch.device("cpu"))
        return self

    def cuda(self, device: int | torch.device | None = None):
        """Move pk chain to CUDA alongside registered buffers."""
        target = torch.device(f"cuda:{device}") if isinstance(device, int) else (
            device if isinstance(device, torch.device) else torch.device("cuda")
        )
        super().cuda(device)
        self._move_pk_chain(target)
        return self

    # -----------------------------------------------------------------
    # Default path resolution
    # -----------------------------------------------------------------

    @staticmethod
    def _default_urdf_path() -> str:
        """Auto-resolve xhand_right.urdf from environment or known locations."""
        candidates = []
        if env_base := os.environ.get("DEXMANI_SIM_PATH"):
            candidates.append(
                os.path.join(env_base, "dexmani_sim/assets/robots/xhand/xhand_right.urdf")
            )
        candidates.append(
            os.path.expanduser("~/Desktop/DexMani_Sim/dexmani_sim/assets/robots/xhand/xhand_right.urdf")
        )
        for p in candidates:
            if os.path.isfile(p):
                return p
        raise FileNotFoundError(
            "Cannot auto-resolve xhand_right.urdf. "
            "Set DEXMANI_SIM_PATH env var or pass urdf_path explicitly."
        )

    # -----------------------------------------------------------------
    # FK chain init (extract from URDF)
    # -----------------------------------------------------------------

    def _init_fk_chain(self, joint_params: OrderedDict[str, JointParam]):
        """Extract joint order and link topology from parsed URDF.

        FK itself is handled by ``pytorch_kinematics.Chain`` — this method
        only stores the bookkeeping needed by other methods (point cloud
        transform dispatch, descriptive fields, ``__repr__``).
        """
        # Map: joint_name → index in revolute joint order (0..11)
        self.revolute_indices: dict[str, int] = {}
        for idx, jname in enumerate(self._DEFAULT_VAE_ORDER):
            self.revolute_indices[jname] = idx

        # Topological order of ALL joints (including fixed)
        self.joint_order: list[str] = []
        self.joint_parent_link: dict[str, str] = {}
        self.joint_child_link: dict[str, str] = {}
        self.joint_is_revolute: dict[str, bool] = {}

        for jname, jp in joint_params.items():
            self.joint_order.append(jname)
            self.joint_parent_link[jname] = jp.parent
            self.joint_child_link[jname] = jp.child
            self.joint_is_revolute[jname] = (jp.joint_type == "revolute")

        # Map: child_link → revolute index (0-11) for active joints
        self._link_to_rv_idx: dict[str, int] = {}
        for jname, jp in joint_params.items():
            if jp.joint_type == "revolute":
                rv_idx = self.revolute_indices.get(jname)
                if rv_idx is not None:
                    self._link_to_rv_idx[jp.child] = rv_idx

        # Build topological order of links (root → leaves)
        self._link_topological_order: list[str] = []
        visited_links = set()
        queue = ["right_hand_link"]

        while queue:
            link = queue.pop(0)
            if link in visited_links:
                continue
            visited_links.add(link)
            self._link_topological_order.append(link)

            for jname in self.joint_order:
                if self.joint_parent_link[jname] == link:
                    child = self.joint_child_link[jname]
                    if child not in visited_links:
                        queue.append(child)

    # -----------------------------------------------------------------
    # Batched FK computation
    # -----------------------------------------------------------------

    def _compute_link_transforms(self, angles: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute per-link world transforms via pytorch_kinematics.

        Matches official OPFA: ``Chain.forward_kinematics(angles)`` returns
        ``dict[str, Transform3D]`` for ALL 30 link frames.

        Args:
            angles: ``(12,)`` (single) or ``(B, 12)`` (batched) in VAE order.

        Returns:
            ``dict[str, Tensor]`` mapping link_name → ``(B, 4, 4)`` world transform.
            Always includes all 30 frames (fixed joints included).
        """
        if angles.dim() == 1:
            angles = angles.unsqueeze(0)  # (1, 12)

        # pk Chain.forward_kinematics → dict[str, Transform3D]
        # Works on GPU: chain and angles on same device, output on same device
        ret = self._pk_chain.forward_kinematics(angles)
        return {k: v.get_matrix() for k, v in ret.items()}

    # -----------------------------------------------------------------
    # Canonical point clouds
    # -----------------------------------------------------------------

    def _init_canonical_pcs(self, mesh_dir: str, sample_points: int):
        """Load STL meshes, sample canonical link PCs, register as buffers.

        Pre-computes:
          - ``_pc_all``: ``(N_total, 3)`` — concatenated link point clouds.
          - ``_finger_ids``: ``(N_total,)`` int64 — per-point finger label.
          - ``_link_ids``: ``(N_total,)`` int64 — per-point link label.
          - ``_link_ranges``: ``dict[str, (int, int)]`` — (start, end) in _pc_all.
          - ``_link_list_loaded``: list of link names with canonical PCs.
        """
        try:
            from stl.mesh import Mesh
        except ImportError:
            raise ImportError(
                "numpy-stl is required for mesh loading. Install with: pip install numpy-stl"
            )

        pc_parts = []
        finger_id_parts = []
        link_id_parts = []
        self._link_ranges: dict[str, tuple[int, int]] = {}
        self._link_list_loaded: list[str] = []

        offset = 0
        for link_name in self._LINK_NAMES:
            stl_path = os.path.join(mesh_dir, f"{link_name}.STL")
            if not os.path.isfile(stl_path):
                continue

            mesh = Mesh.from_file(stl_path)
            vertices = np.unique(mesh.vectors.reshape(-1, 3), axis=0).astype(np.float32)

            # Sample to fixed count
            n_verts = len(vertices)
            if n_verts > sample_points:
                idx = np.random.RandomState(42).choice(n_verts, sample_points, replace=False)
                vertices = vertices[idx]
            elif n_verts < sample_points:
                repeats = sample_points // n_verts + 1
                vertices = np.tile(vertices, (repeats, 1))[:sample_points]
                vertices += np.random.RandomState(42).randn(*vertices.shape).astype(np.float32) * 1e-4

            n = len(vertices)
            pc_parts.append(torch.from_numpy(vertices))
            finger_id, link_id = self._FINGER_LINK_INDICES[link_name]
            finger_id_parts.append(torch.full((n,), finger_id, dtype=torch.long))
            link_id_parts.append(torch.full((n,), link_id, dtype=torch.long))

            self._link_ranges[link_name] = (offset, offset + n)
            self._link_list_loaded.append(link_name)
            offset += n

        self.register_buffer("_pc_all", torch.cat(pc_parts, dim=0))  # (N_total, 3)
        self.register_buffer("_finger_ids", torch.cat(finger_id_parts, dim=0))  # (N_total,)
        self.register_buffer("_link_ids", torch.cat(link_id_parts, dim=0))  # (N_total,)

        # Build per-link transform index: for each point in _pc_all, which link index
        # in _link_list_loaded does it belong to?
        link_to_idx = {name: i for i, name in enumerate(self._link_list_loaded)}
        link_idx_per_point = torch.zeros(offset, dtype=torch.long)
        for link_name, (s, e) in self._link_ranges.items():
            link_idx_per_point[s:e] = link_to_idx[link_name]
        self.register_buffer("_link_idx_per_point", link_idx_per_point)  # (N_total,)

    # -----------------------------------------------------------------
    # Cache
    # -----------------------------------------------------------------

    def _cache_key(self, angles: torch.Tensor) -> tuple[float, ...]:
        """Quantize angles to produce a hashable cache key."""
        return tuple(round(float(a), -int(np.log10(self.cache_tolerance))) for a in angles.cpu())

    @property
    def cache_stats(self) -> dict:
        """Return cache hit/miss statistics."""
        return {"hits": self._cache_hits, "misses": self._cache_misses,
                "size": len(self._cache) if self._cache else 0}

    # -----------------------------------------------------------------
    # Forward — FK → PC → KPConv data
    # -----------------------------------------------------------------

    @torch.no_grad()
    def forward(
        self,
        joint_angles: torch.Tensor,
        hand_type: str = "xhand",
        precompute_data: bool = True,
    ) -> dict:
        """Generate hand point cloud + KPConv data from joint angles.

        Args:
            joint_angles: ``(12,)`` or ``(B, 12)`` in VAE order.
            hand_type: always ``"xhand"``.
            precompute_data: if True, build multi-scale KPConv neighbours/subsampling.

        Returns:
            dict with keys: ``features``, ``points``, ``lengths``, ``neighbors``,
            ``subsampling``, ``upsampling``, ``hand_type``, ``angles``.
            Always returns the FIRST sample's data in per-sample format
            (consistent with OPFA's batch_size=1 convention).
        """
        if joint_angles.dim() == 1:
            joint_angles = joint_angles.unsqueeze(0)  # (1, 12)
        B = joint_angles.shape[0]

        # ── Cache lookup (full batch key) ──
        if self._cache_enabled and B == 1:
            key = self._cache_key(joint_angles[0])
            if key in self._cache:
                self._cache_hits += 1
                return self._cache[key]
            self._cache_misses += 1

        # ── Batched FK: compute all link transforms ──
        T_links = self._compute_link_transforms(joint_angles)  # dict[link] → (B, 4, 4)

        # ── Batched point cloud transform ──
        # For each link, transform its canonical points by T_links[link_name]
        # _pc_all: (N_total, 3), _link_idx_per_point: (N_total,) → index in _link_list_loaded
        N = self._pc_all.shape[0]
        pts_h = torch.cat([self._pc_all, torch.ones(N, 1, dtype=self._pc_all.dtype, device=self._pc_all.device)], dim=-1)  # (N, 4)

        # Gather per-link transforms for each point: (B, num_links, 4, 4) → (B, N, 4, 4)
        T_per_point = torch.stack(
            [T_links[name] for name in self._link_list_loaded], dim=1
        )[:, self._link_idx_per_point, :, :]  # (B, N, 4, 4)

        # Transform all points at once: (B, N, 4, 4) @ (1, N, 4, 1) → (B, N, 3)
        pts_h_b = pts_h.unsqueeze(0)  # (1, N, 4)
        pts_world = (T_per_point @ pts_h_b.unsqueeze(-1)).squeeze(-1)[..., :3]  # (B, N, 3)

        # ── Build features: (finger_id, link_id) per point ──
        feats = torch.stack([self._finger_ids, self._link_ids], dim=-1)  # (N, 2) on device

        # ── Per-link point counts ──
        lengths = [self._link_ranges[name][1] - self._link_ranges[name][0] for name in self._link_list_loaded]

        # ── Build result for first sample (OPFA convention) ──
        pc0 = pts_world[0]  # (N, 3)

        result: dict = {
            "features": feats,
            "angles": joint_angles[0].cpu().numpy() if B == 1 else joint_angles.cpu().numpy(),
            "lengths": lengths,
            "hand_type": hand_type,
        }

        if precompute_data:
            kpconv_data = self._build_kpconv_data(pc0, lengths)
            result.update(kpconv_data)
        else:
            result["points"] = pc0

        # ── Cache result (single sample only) ──
        if self._cache_enabled and B == 1:
            key = self._cache_key(joint_angles[0])
            if len(self._cache) >= self._cache_maxsize:
                self._cache.popitem(last=False)  # FIFO eviction
            self._cache[key] = result

        return result

    # -----------------------------------------------------------------
    # Multi-scale KPConv data structure
    # -----------------------------------------------------------------

    def _build_kpconv_data(self, points: torch.Tensor, lengths: list[int]) -> dict:
        """Build grid subsampling + radius search data at all pyramid levels.

        Faithful to OPFA ``hand_precompute_data_stack_mode``.
        """
        lengths_t = torch.tensor(lengths, dtype=torch.long, device=points.device)
        voxel_size = self.init_voxel_size
        radius = self.init_radius

        points_list = []
        lengths_list = []
        neighbors_list = []
        subsampling_list = []
        upsampling_list = []

        cur_points = points
        cur_lengths = lengths_t

        for i in range(self.num_stages):
            if i > 0:
                cur_points, cur_lengths = grid_subsample(cur_points, cur_lengths, voxel_size=voxel_size)
            points_list.append(cur_points)
            lengths_list.append(cur_lengths)
            voxel_size *= 2

        for i in range(self.num_stages):
            cur_points = points_list[i]
            cur_lengths = lengths_list[i]  # per-link counts
            cur_total = cur_lengths.sum().unsqueeze(0)  # (1,) — flatten links

            neighbors = radius_search(
                cur_points, cur_points,
                cur_total, cur_total,
                radius,
                self.neighbor_limits[i],
            )
            neighbors_list.append(neighbors)

            if i < self.num_stages - 1:
                sub_points = points_list[i + 1]
                sub_total = lengths_list[i + 1].sum().unsqueeze(0)

                subsampling = radius_search(
                    sub_points, cur_points,
                    sub_total, cur_total,
                    radius,
                    self.neighbor_limits[i],
                )
                subsampling_list.append(subsampling)

                upsampling = radius_search(
                    cur_points, sub_points,
                    cur_total, sub_total,
                    radius * 2,
                    self.neighbor_limits[i + 1],
                )
                upsampling_list.append(upsampling)

            radius *= 2

        return {
            "points": points_list,
            "lengths": lengths_list,
            "neighbors": neighbors_list,
            "subsampling": subsampling_list,
            "upsampling": upsampling_list,
        }

    def __repr__(self) -> str:
        n_rev = sum(1 for v in self.joint_is_revolute.values() if v)
        n_fixed = sum(1 for v in self.joint_is_revolute.values() if not v)
        cache_info = f" cache={len(self._cache)}/{self._cache_maxsize}" if self._cache_enabled else ""
        return (
            f"HandFKGenerator(urdf={self.urdf_path}, "
            f"links={len(self._link_list_loaded)}, "
            f"revolute_joints={n_rev}, fixed_joints={n_fixed}, "
            f"stages={self.num_stages}{cache_info})"
        )


# =============================================================================
# Convenience functions
# =============================================================================

# For DexMani (VAE-native order), these are identity:
def vae_to_isaac(angles: torch.Tensor) -> torch.Tensor:
    """Convert VAE joint order → IsaacLab joint order (identity for DexMani)."""
    return angles  # identity: DexMani uses VAE order natively


def isaac_to_vae(angles: torch.Tensor) -> torch.Tensor:
    """Convert IsaacLab joint order → VAE joint order (identity for DexMani)."""
    return angles  # identity
