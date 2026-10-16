# HandSpatialEncoder: Wrist–Fingertip Spatial Encoding

## Purpose and input contract

The module maps six measured 3D anchors to six semantic observation tokens:

- Input: **[BT, 6, 3]** float XYZ in a *shared metric world/base frame*.
- Order: **wrist, thumb, index, middle, ring, pinky**.
- Output: **[BT, 6, D]**, with D=192 in InteractionFlow.

No forward kinematics, rotation reconstruction, coordinate-frame transformation, contact estimation, or hand–scene attention is performed here. The observed wrist and fingertip positions are already computed upstream. Position vectors and relative displacements stay aligned with the input world axes.

## Encoding

For each point `p_i` and wrist point `p_0`, define a fixed-scale input `u_i=p_i/s_g` and

```text
gamma(u) = [u, sin(pi*u), sin(2*pi*u), cos(pi*u), cos(2*pi*u)]
delta_i = (p_i - p_0) / s_h
z_i = LayerNorm(MLP(gamma(u_i)) + W_delta * delta_i + ID_i)
```

The sinusoidal part uses two frequencies by default; the implementation reuses the repository's axis-major `NeRFSinusoidalPosEmb3D`. The shared MLP is 15→128→192 with GELU. `W_delta` is a bias-free 3→192 projection initialized to zero. This optional residual supplies *each* fingertip with its own wrist-to-tip displacement, rather than broadcasting a flattened whole-hand layout vector to every anchor. Finger identity embeddings distinguish coincident points.

Unlike a wrist-frame coordinate, `p_i-p_0` is still expressed along world axes. Only this displacement term is invariant to shared translation. The complete representation is not translation-invariant, rotation-invariant or SE(3)-equivariant. These features contain no contact force or finger orientation.

Default scales: `s_g=0.25m`, `s_h=0.10m`. They change **feature units only**. They never alter original XYZ used for metric hand–scene distances. Position inputs should remain FP32 in mixed-precision training; the implementation forms Fourier phases and additive residuals in FP32 to preserve small corrections.

With D=192, the position-only model has 28,352 trainable parameters; enabling the offset residual adds 576, totaling **28,928**. This is a module parameter reduction relative to the original `HandKinematicsEncoder`, not a claim of proportional end-to-end speedup.

## Relation to interaction modeling

InteractionFlow first constructs `hand_xyz = cat([eef_pose[..., :3], fingertip_xyz], dim=1)` and encodes it **once**. The output is reused in tactile/geometry gates, geometry retrieval queries, and hand coordination. The unscaled `hand_xyz` separately anchors `HandSceneRelationEncoder`'s context and near-field geometric reads. Scene points, anchors and displacement vectors share one metric frame.

The geometry reader, tactile evidence branch, fusion gates and hand coordination remain responsible for hand–object interaction. A nearby visible point is not a verified contact; a missing tactile measurement is not equivalent to zero contact force. No new attention layer, Voxel module, temporal encoder or auxiliary head is introduced.

The dataset still stores `eef_pose [BT,9]` (XYZ + rot6d) and `fingertip_xyz [BT,5,3]`; only the spatial encoding path consumes XYZ. Removing the old rotation pathway is a research-model change, not a format or checkpoint-compatible rename.

## Controlled ablations

Set `agent.hand_spatial` in `configs/interaction_flow.yaml`:

| Variant | num_frequencies | use_wrist_offset |
|---|---:|---|
| A: scaled raw XYZ + point identity | 0 | false |
| B: raw XYZ + low-frequency Fourier | 2 | false |
| C: B + per-tip wrist displacement | 2 | true |

Hold the scene backbone, tactile path, fusion variant, token dimension, data split and training budget fixed. Compare both interaction success and robustness to hand/world placement, positional noise and sensor calibration. Performance improvements remain hypotheses until measured in closed-loop evaluations.

## Compatibility and validation

`hand_kinematics.py` and its `HandKinematicsEncoder` are removed; so are the `edge_frame` switch and the reader's `wrist_rotation` dependence. Existing checkpoints **cannot be strictly resumed** under the new module structure. Historical results should be evaluated with the original source and saved configuration.

Focused CPU tests on the source snapshot: **75 passed, 22 skipped** using

```bash
python -m pytest -q tests/test_hand_spatial_encoder.py tests/test_geometry_aware_point_transformer.py
```

The skips include native CUDA/PyTorch3D checks. This does **not** establish full-repository, CUDA, DDP, on-robot or training-run validation. The manuscript should not claim the spatial encoder is a novel contact detector or a complete geometry-equivariant policy.
