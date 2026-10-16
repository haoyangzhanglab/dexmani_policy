# InteractionFlow: body-anchored hand–scene interaction

This document describes the current spatial path following the 2026-10-16 refactor. For the earlier experimental discussion and module semantics, refer to [the pre-refactor source](https://github.com/haoyangzhanglab/dexmani_policy/blob/aefbddc0f561e1f57689dcc66705a2fe84f243b4/docs/interaction_representation.md).

## Research story

Dexterous manipulation requires the policy to jointly answer **where the visible surface is**, **which part of the hand is near it**, and **what the fingertip currently feels**. Scene point clouds encode observed geometry; six semantic hand anchors locate possible effectors; per-finger tactile streams reflect local interaction. Proximity alone does not prove contact, nor does an attention weight represent calibrated contact probability.

The architecture tests whether fingertip-local tactile signals should condition geometric retrieval (query fusion) or enter only after retrieval (late fusion). Its spatial encoder is a deliberately small *conditioning component*, not the central hand–scene interaction mechanism.

## Actual observation path

Per frame: 16 scene tokens, 6 hand tokens and 1 state token, each D=192. With two observed frames, DiTX receives [B,46,192] tokens.

1. **Scene memory.** GAPT encodes XYZRGB point clouds with a single shared 3D geometry backbone; a shared projection supplies `SceneContextPool` and both hand–scene readers.
2. **Hand anchors.** The observed wrist XYZ from `eef_pose[..., :3]` is concatenated with the 5 ordered `fingertip_xyz` to form world-frame `[BT,6,3]`. [HandSpatialEncoder](hand_spatial_encoding.md) maps it to `[BT,6,D]` using scaled raw XYZ, 2-frequency Fourier features, six learnable identities and an optional zero-initialized per-tip wrist-displacement residual.
3. **Geometry retrieval.** `HandSceneRelationEncoder` separately receives unscaled `anchor_xyz` and already encoded `hand_tokens`. Context/near attention operates on **world-axis** `scene_xyz - anchor_xyz`, meter distances, multi-scale Gaussian biases and learned null entries. The hand encoder runs only once.
4. **Interaction evidence.** Per-finger tactile features and valid flags form `FingerEvidenceEncoder` outputs; `FingerModalityGate` computes geometry/tactile gains from pre-retrieval context. The query variant adds gated tactile evidence before the geometry read; the late variant uses it only in the common residual.
5. **Coordination and actions.** One six-token hand self-attention block combines spatial, joint-state, tactile and geometry evidence. A standard DiTX/RectifiedFlow decoder predicts the action trajectory. No additional hand-local attention or contact prediction module is added.

All physical XYZ, point-cloud centers and relative geometry use the same input meter-scale axes. The former `edge_frame=wrist` and `wrist_rotation` pathways are removed by design; there is no FK or frame rotation in the spatial encoder. The retained `eef_pose` data field still has nine components, but its rotation portion is not consumed by this particular spatial/reader path.

## Configuration and evidence

The single-card policy config remains `configs/interaction_flow.yaml`. Its new `agent.hand_spatial` block controls `hidden_dim`, `position_scale`, `hand_scale`, `num_frequencies`, and `use_wrist_offset`. The scene model, point-cloud geometry, tactile modality gate, optimizer, action backbone, and training budget are unchanged by this refactor.

The dataset/observation schema remains `eef_pose [BT,9]`, `fingertip_xyz [BT,5,3]`, `joint_state`, `point_cloud`, and `finger_tactile_force` (plus optional taxels and validity masks). Point-cloud, wrist and fingertip XYZ must be metric and in the same base frame. Real-robot base-frame coordinates and simulated world-frame numbers need not match across datasets; they are not mixed within a sample.

New configs/weights are **not** compatible with strictly restoring old `HandKinematicsEncoder` checkpoints. Use new runs; preserve old results with their original source and effective configuration. Independent CPU unit and focused integration tests have run, but CUDA kernels, full policy setup, closed-loop success, and on-robot performance are not yet validated.

Recommended comparison: A (raw XYZ), B (+ Fourier), C (+ per-finger wrist offset) under a fixed Q/L tactile design and fixed seeds/splits. The purpose is to test an inductive bias, not claim that Fourier, point identity or pose conditioning are novel primitives.
