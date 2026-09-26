# ManiFlow XYZRGB + R3D Augmentation Fix Task

## 0. Task objective

Update the ManiFlow implementation in this repository so that it is:

1. **Faithful to the official ManiFlow core architecture and training semantics**.
2. **Explicitly adapted to XYZRGB dense point-cloud input**.
3. **Compatible with R3D-style point-cloud/data augmentation**.
4. Uses the final observation representation:
   **PointNet(XYZRGB) + projected Adapt3R-style XYZ PE + frame PE**.
5. Keeps the implementation minimal, local to ManiFlow where possible, easy to audit, and free of unnecessary changes to other policies.

Do **not** create or switch branches. Work on the current branch.

This is a correctness/fidelity task, not a broad refactor. Do not introduce unrelated architecture changes.

---

## 1. Reference implementations and validated baseline

Use the following references when resolving implementation details:

### Official ManiFlow
Repository:
https://github.com/geyan21/ManiFlow_Policy

Validated reference commit:
`ef2f116f1f90163ed36e657b8c5503740bb468af`

Relevant files:

- `ManiFlow/maniflow/model/vision_3d/pointnet_extractor.py`
- `ManiFlow/maniflow/model/diffusion/ditx.py`
- `ManiFlow/maniflow/model/diffusion/ditx_block.py`
- `ManiFlow/maniflow/policy/maniflow_pointcloud_policy.py`
- `ManiFlow/maniflow/config/maniflow_pointcloud_policy.yaml`

### Official R3D
Repository:
https://github.com/Wushr-Lance/R3D-Policy

Validated reference commit:
`e637c0148376ddc4b5e667fa8f8e108cb8ff7a85`

Relevant files:

- `R3D/r3d/config/r3d_robotwin2.yaml`
- `R3D/r3d/dataset/robotwin_dataset.py`
- `R3D/r3d/model/vision/pointnet_extractor.py`
- `R3D/r3d/policy/dp3.py`

### Adapt3R positional encoding reference
Repository:
https://github.com/pairlab/Adapt3R

Validated reference commit:
`9563f068ee34b6a6cc6e760808bb219620f93af4`

Relevant files:

- `adapt3r/algos/utils/position_encodings.py`
- `adapt3r/algos/encoders/adapt3r.py`

Use the **mathematical idea** of Adapt3R's continuous NeRF-style XYZ encoding. Implement it in this repository's own style rather than copying unrelated Adapt3R code structure.

### Current dexmani_policy baseline used for this task
The plan was validated against:

`b97651bd90df3d2f4d73986b12ce7c70d0057e26`

Before editing, inspect the current HEAD. If relevant code has changed since that commit, preserve the intent below and adapt the implementation to the current interfaces rather than blindly applying stale line-level edits.

---

## 2. Final intended architecture

The final ManiFlow observation path must be:

```text
XYZRGB point cloud [B,T,N,6]
joint_state        [B,T,19]
        |
        v
R3D-style dataset augmentation
- XYZ noise
- RGB brightness / contrast / saturation
- joint-state noise
        |
        v
normalization
        |
        v
[B*T,N,6]
        |
        +------------------------------+
        |                              |
        v                              v
Official ManiFlow                normalized XYZ
RGB PointNetDense                    |
6->64->128->256->512->128            v
(no pooling)                  NeRF-style 3D PE
        |                     K=8 frequencies/axis
        |                         48 dimensions
        |                              |
        |                        Linear 48->128
        |                              |
        |                        LayerNorm(128)
        |                              |
        +------------- add ------------+
                      |
                 spatial point feature
                      128
                      |
                  StateMLP
                 19 -> 64
                      |
              broadcast per point
                      |
           concat(point128, state64)
                      |
                  192D token
                      |
               [B,T*N,192]
                      |
                Linear 192->768
                      |
              + frame-only PE
                [1,T,768], repeated
                across points in frame
                      |
                      v
             observation K/V memory

action [B,16,19]
        |
Linear 19->768 + official action temporal PE
        |
        v
12 x official ManiFlow DiT-X
- action self-attention
- action -> observation cross-attention
- MLP
- dual (t, target_t) conditioning
- AdaLN-Zero
        |
        v
velocity [B,16,19]
        |
Consistency Flow + EMA teacher
        |
Euler integration
        |
prediction [B,16,19]
        |
slice [1:9]
        |
control [B,8,19]
```

The implementation must preserve:

- horizon = 16
- n_obs_steps = 2
- n_action_steps = 8
- action_dim = 19 for joint-action mode
- pc_dim = 6
- 1024 dense point tokens per frame in the current dataset
- PointNet output dim = 128
- state feature dim = 64
- observation token dim = 192
- DiT-X hidden dim = 768
- 12 DiT-X blocks
- 8 heads
- action self-attention followed by action-query / observation-key-value cross-attention
- flow/consistency split ratio = 0.75 / 0.25
- EMA consistency target
- Euler sampling
- current action execution alignment `pred[:, 1:9]`

---

## 3. Scope and non-goals

### Required changes

Only make changes needed for:

1. Official XYZRGB PointNet fidelity.
2. Adapt3R-style continuous XYZ positional encoding.
3. Frame-only observation temporal positional encoding.
4. Official ManiFlow DiT-X self-attention semantics.
5. Separation of consistency training time-grid resolution from inference NFE.
6. R3D-compatible color/coordinate/state augmentation semantics.
7. Tests and assertions proving the above behavior.

### Explicit non-goals

Do **not** introduce in this task:

- SAT-style action decomposition.
- wrist/fingertip action tokens.
- hand/arm-specific attention.
- stage-aware attention.
- observation self-attention.
- 3D RoPE in DiT-X.
- learned wrist/fingertip spatial anchors.
- point-token pruning.
- global pooling.
- a global point-cloud token.
- R3D Uni3D.
- R3D OneWayTransformer.
- R3D auxiliary EE action head.
- aggressive R3D point dropout.
- changes to action horizon or execution slicing.
- changes to EMA policy.
- changes to optimizer/training budgets unless strictly required for correctness.
- broad renaming/refactoring of shared modules.
- checkpoint migration logic for old ManiFlow checkpoints.

Old ManiFlow checkpoints are allowed to be architecture-incompatible with this corrected baseline. Do not add fragile heuristic checkpoint migration code.

---

## 4. Change 1 — restore the official ManiFlow XYZRGB PointNet

### Current problem

Current `PointNetDense` applies a generic hidden-dimension loop:

```text
6 -> 64 LN ReLU
  -> 128 LN ReLU
  -> 256 LN ReLU
  -> 128 LN
```

This is smaller than the official ManiFlow XYZRGB encoder.

### Required official XYZRGB topology

Implement exactly:

```text
6
-> Linear(6, 64)
-> LayerNorm(64)
-> ReLU
-> Linear(64, 128)
-> LayerNorm(128)
-> ReLU
-> Linear(128, 256)
-> LayerNorm(256)
-> ReLU
-> Linear(256, 512)
-> Linear(512, out_channels)
-> LayerNorm(out_channels)
```

There must be **no LayerNorm or ReLU between 256->512 and 512->out_channels**.

Keep:

- class name `PointNetDense`
- no pooling
- no patch aggregation
- no global token
- one output token per input point

### XYZ compatibility

If the class intentionally continues supporting `input_channels == 3`, preserve the official XYZ branch semantics rather than forcing the RGB branch onto XYZ input.

Before changing the constructor API, perform a repository-wide call-site search.

Preferred outcome:

- remove arbitrary `hidden_dims` control from the ManiFlow configuration;
- avoid allowing YAML changes to silently alter the official ManiFlow PointNet topology;
- update all in-repo call sites/examples/tests consistently.

Do not rename the class to `DP3PointNet` or similar.

### Files

Primary:

- `dexmani_policy/agents/obs_encoder/pointcloud/pointnet_dense.py`
- `dexmani_policy/agents/obs_encoder/pointcloud/registry.py`
- `dexmani_policy/configs/maniflow.yaml`

Also update any example/test that still passes `hidden_dims`.

---

## 5. Change 2 — add Adapt3R-style continuous XYZ positional encoding

### Goal

Make each dense point token explicitly spatial while remaining permutation-equivariant.

The point feature must become:

```text
point_feature
=
PointNetDense(XYZRGB)
+
projected_NeRF_XYZ_PE(XYZ)
```

### Positional encoding definition

Add a small reusable module in:

`dexmani_policy/agents/position_encodings.py`

Suggested class name:

`NeRFSinusoidalPosEmb3D`

Do not modify the semantics of the existing `SinusoidalPosEmb3D`, because other point-cloud models already use it.

Use continuous XYZ and fixed NeRF/Fourier frequencies:

```text
frequencies = 2^[0, 1, ..., K-1]
K = 8
```

For each axis:

```text
sin(freq * coordinate)
cos(freq * coordinate)
```

Concatenate x/y/z encodings.

For `K=8`:

```text
3 axes * 8 frequencies * 2 = 48 dimensions
```

Expected contract:

```text
input:  [..., 3]
output: [..., 48]
```

The frequency tensor should be a non-trainable registered buffer so device/dtype movement works correctly and no per-forward tensor construction is required.

No gradient is required with respect to the fixed frequency basis, but do not unnecessarily detach the XYZ tensor in a way that complicates tracing/compile.

### Projection

In `ManiFlowObsEncoder`, add a ManiFlow-specific projection:

```text
NeRFSinusoidalPosEmb3D(K=8)
-> Linear(48, 128)
-> LayerNorm(128)
```

Then:

```text
pc_feat = PointNetDense(pc)
xyz_pe  = xyz_pe_proj(xyz_pe(pc[..., :3]))
pc_feat = pc_feat + xyz_pe
```

Do not concatenate the PE and increase `obs_token_dim`.

The final point feature remains 128D.

### Coordinate source

The PE must use the **same point XYZ tensor seen by PointNetDense** after the normal training preprocessing path:

```text
raw XYZ
-> dataset augmentation
-> normalizer
-> point-cloud preprocessing / sampling
-> PointNet and XYZ PE
```

Do not encode stale pre-augmentation coordinates.

Do not encode RGB.

### Configuration

Expose only the meaningful hyperparameter:

```yaml
xyz_pe_num_frequencies: 8
```

Do not over-generalize this into a large configurable PE framework in this task.

---

## 6. Change 3 — replace flat point-index context PE with frame-only PE

### Current problem

Current DiT-X uses:

```text
context_pos_embed: [1, T*N, hidden_dim]
```

This assigns learned positional identity to arbitrary point indices.

That conflicts with randomized/FPS point-set semantics.

### Required behavior

Keep temporal observation-frame identity, remove point-index identity.

In `ConsistencyDiTX`:

- remove `context_pos_embed`;
- add `n_obs_steps`;
- add `context_frame_pos_embed` with shape:

```text
[1, n_obs_steps, hidden_dim]
```

Use it **after** the official-style context projection:

```text
context_c = context_embedder(context)
context_c = context_c + frame_pe_repeated_over_points
```

At runtime:

```text
N_total = context.shape[1]
assert N_total % n_obs_steps == 0
tokens_per_frame = N_total // n_obs_steps
frame_pe = context_frame_pos_embed.repeat_interleave(tokens_per_frame, dim=1)
```

All points from the same observation frame receive the same frame PE.

### Initialization

Initialize `context_frame_pos_embed` to zeros.

This is the conservative choice closest to the official ManiFlow observation positional embedding initialization.

Keep the action positional embedding initialization unchanged.

### Optimizer grouping

Update no-weight-decay parameter names:

- remove `context_pos_embed`;
- add `context_frame_pos_embed`.

### API cleanup

Once flat `num_obs_tokens` PE allocation is removed, `ConsistencyDiTX` should no longer require `num_obs_tokens` merely to allocate position embeddings.

Prefer passing:

- `n_obs_steps`
- `obs_token_dim`

and deriving per-frame token count dynamically at forward time.

Keep `ManiFlowObsEncoder.num_obs_tokens` only if it is still useful elsewhere; do not keep dead API purely for historical compatibility.

---

## 7. Change 4 — restore official ManiFlow DiT-X self-attention semantics

### Current problem

Current `DiTXBlock` routes `qkv_bias` and `qk_norm` into both self-attention and cross-attention.

Official ManiFlow does not.

Official self-attention is:

```python
nn.MultiheadAttention(
    hidden_size,
    num_heads,
    batch_first=True,
    dropout=p_drop_attn,
)
```

Therefore official self-attention semantics are:

- QKV bias = true
- no qk norm

The config-level `qkv_bias` / `qk_norm` belongs to custom cross-attention.

### Required change

In `DiTXBlock`:

- use `nn.MultiheadAttention` for self-attention;
- keep the existing custom `CrossAttention` for action-query -> observation-K/V cross-attention;
- keep `qkv_bias` and `qk_norm` controlling **cross-attention only**.

Forward:

```text
normed_x
-> self_attn(normed_x, normed_x, normed_x, need_weights=False)[0]
-> residual/gate
-> cross-attention
-> residual/gate
-> MLP
```

Do not modify the shared `dit.py::Attention` to force ManiFlow-specific behavior.

### Initialization

Update ManiFlow initialization to the official MHA parameter names:

- Xavier uniform `in_proj_weight`
- zero `in_proj_bias`
- Xavier uniform `out_proj.weight`
- zero `out_proj.bias`

Keep the rest of the official-style initialization logic intact.

---

## 8. Change 5 — decouple consistency training time grid from inference NFE

### Current problem

Current code constructs:

```python
TimeSampler(num_steps=num_inference_steps)
```

This incorrectly couples:

- consistency-training discrete time-grid resolution;
- inference Euler step count.

Official ManiFlow separates them.

### Required API

Add:

```text
denoise_timesteps = 10
num_inference_steps = 10
```

Semantics:

```text
denoise_timesteps
    -> discrete consistency-training grid

num_inference_steps
    -> default Euler inference NFE
```

Construct:

```python
self.time_sampler = TimeSampler(num_steps=denoise_timesteps)
```

Do not let an evaluation NFE override mutate the training sampler.

Current evaluation behavior such as:

```text
eval denoise_steps = 4
```

must remain a runtime inference override only.

### Consistency denominator

Remove the dependency:

```text
clamp(min=1 / num_inference_steps)
```

The consistency target must not depend on inference NFE.

Use:

```python
denominator = (1.0 - t_view).clamp_min(1e-6)
```

or the exact mathematically equivalent implementation appropriate to the existing code.

For the standard discrete grid with `K=10`, valid training `t` values are below 1, so the clamp is only a defensive numerical guard.

---

## 9. Change 6 — make R3D-style augmentation semantics correct

### Keep enabled

For ManiFlow:

```yaml
coord_noise:
  noise_std: 0.002
  prob: 1.0

color:
  brightness: 0.125
  contrast: 0.5
  saturation: 0.5
  hue: 0.0
  prob: 1.0

state:
  noise:
    noise_std: 0.0002
    prob: 1.0
```

### Contrast fix

Current point-cloud contrast is centered around constant 0.5.

R3D uses the current flattened RGB global mean.

Required contrast transform:

```text
mean = rgb.mean()
rgb = (rgb - mean) * factor + mean
```

Implement it in the shared `PointColorJitter` helper because that class already claims R3D-aligned semantics.

Do not create a ManiFlow-only duplicate augmentor.

### Hue

Keep generic hue support in the augmentor, but set:

```yaml
hue: 0.0
```

for the ManiFlow R3D-style recipe.

Do not remove hue support globally.

### Preserve

Keep:

- additive brightness semantics;
- existing saturation formula when algebraically equivalent to R3D;
- clipping to valid RGB range;
- coordinate-noise clipping behavior;
- state-noise clipping behavior.

---

## 10. Random FPS / point-order behavior

### Important current-data fact

Current DexMani point clouds are already produced as exactly 1024 points upstream.

Therefore current policy input commonly has:

```text
N == num_points == 1024
```

and the existing policy-side random FPS path is not active.

### Required behavior for this task

Do **not** fake R3D random FPS by running FPS with `K=N`.

Do **not** add a separate `N==K` shuffle-only augmentation in this task.

After removing flat point-index PE and using XYZ-dependent PE, pure permutation does not provide a meaningful new geometric subset and only adds overhead.

Keep existing randomized FPS capability for future higher-density inputs where:

```text
N_raw > 1024
```

and document this accurately in `maniflow.yaml`.

Do not change `dexmani_sim` or the Zarr format in this task.

A future full R3D random-FPS experiment should use higher-density stored input and randomly sample to 1024 during training.

---

## 11. Point dropout policy

Do not enable aggressive R3D point dropout in the default ManiFlow configuration.

Reason:

- R3D applies point dropout before a patch/grouping/Uni3D pipeline.
- ManiFlow PointNetDense emits independent pointwise tokens directly into DiT-X K/V memory.
- Copying R3D's high dropout ratio can create a materially different and much harsher observation distribution.

Leave existing generic point-dropout utilities untouched unless tests require a bug fix.

Point dropout can be evaluated later as a separate augmentation ablation.

---

## 12. ManiFlow observation encoder final tensor contract

After the changes, enforce:

```text
input point cloud:
[B*T, N, 6]

PointNet:
[B*T, N, 128]

NeRF XYZ PE:
[B*T, N, 48]

projected XYZ PE:
[B*T, N, 128]

spatial point feature:
[B*T, N, 128]

StateMLP:
[B*T, 64]

broadcast state:
[B*T, N, 64]

concat:
[B*T, N, 192]

reshape:
[B, T*N, 192]
```

For the default configuration:

```text
T = 2
N = 1024
context = [B, 2048, 192]
```

DiT-X then performs:

```text
context_embedder:
192 -> 768

frame-only PE:
[B, 2048, 768]
```

No observation self-attention is introduced.

---

## 13. Keep these ManiFlow behaviors unchanged

Do not change the following unless a direct official-reference mismatch is discovered while implementing the scoped fixes:

- rectified-flow interpolation;
- flow target `x1 - x0`;
- beta flow-time sampling;
- consistency EMA teacher;
- relative target-t mode;
- merged online student forward;
- flow/consistency mini-batch split;
- action token = full action vector at one trajectory timestep;
- action temporal self-attention;
- action-query -> observation-K/V cross-attention;
- 12 DiT-X blocks;
- hidden size 768;
- 8 heads;
- MLP ratio 4;
- attention dropout 0.1;
- pre_norm_modality false;
- final action dimension;
- horizon = 16;
- observation horizon = 2;
- executed action slice = 1:9;
- BF16;
- torch.compile;
- gradient clipping;
- optimizer settings;
- EMA schedule;
- evaluation NFE override behavior.

---

## 14. Required config changes

Update `dexmani_policy/configs/maniflow.yaml`.

The resulting relevant config should express:

```yaml
agent:
  encoder_type: pointnet_dense
  pc_dim: 6
  state_dim: 19
  num_points: 1024
  state_out_dim: 64

  pc_encoder_config:
    out_channels: 128
    num_points: 1024

  xyz_pe_num_frequencies: 8

  fps_random_config:
    use_random: true
    use_random_start: true
    random_noise_scale: 0.0
    use_shuffle_output: true

  n_layers: 12
  hidden_dim: 768
  n_head: 8
  mlp_ratio: 4.0
  p_drop_attn: 0.1
  qkv_bias: false
  qk_norm: false
  pre_norm_modality: false

  timestep_embed_dim: 128
  target_t_embed_dim: 128

  denoise_timesteps: 10
  num_inference_steps: 10

  flow_batch_ratio: 0.75
  t_sample_mode_for_flow: beta
  t_sample_mode_for_consistency: discrete
  dt_sample_mode_for_consistency: uniform
  target_t_sample_mode: relative
```

Augmentation:

```yaml
augmentation_cfg:
  pc:
    coord_noise:
      noise_std: 0.002
      prob: 1.0
    color:
      brightness: 0.125
      contrast: 0.5
      saturation: 0.5
      hue: 0.0
      prob: 1.0
  state:
    noise:
      noise_std: 0.0002
      prob: 1.0
```

Remove the obsolete ManiFlow `hidden_dims` setting if the code no longer consumes it.

Add a concise comment explaining that random FPS becomes active only when the incoming point count exceeds `num_points`; current DexMani stored clouds are already 1024 points.

---

## 15. Tests — required, not optional

Locate the repository's existing test organization and add focused tests in the appropriate place. Do not create an unrelated standalone test framework.

### 15.1 PointNet structure and shape

Verify for XYZRGB:

```text
[B,N,6] -> [B,N,128]
```

Verify the module topology includes:

```text
6->64
64->128
128->256
256->512
512->128
```

Verify:

- no pooling;
- no global token;
- no LN/ReLU inserted between `256->512` and `512->128`.

### 15.2 NeRF XYZ PE

For `K=8`:

```text
[B,N,3] -> [B,N,48]
```

Verify deterministic output and correct device/dtype behavior.

Verify point permutation equivariance:

```text
PE(PX) == P PE(X)
```

within floating-point tolerance.

### 15.3 ManiFlow observation encoder

Verify:

```text
point_cloud [B*T,N,6]
joint_state [B*T,19]
-> cond [B,T*N,192]
```

For the default small test equivalent, preserve the same dimension relationships even if N is reduced for speed.

### 15.4 Frame-only context PE

Verify:

- no `context_pos_embed` parameter exists;
- `context_frame_pos_embed.shape == [1,T,hidden_dim]`;
- all points from one frame receive the same frame PE;
- different frames are allowed to learn different PE;
- token counts are validated as divisible by `n_obs_steps`.

### 15.5 DiT-X self-attention fidelity

Verify:

- self-attention is `nn.MultiheadAttention`;
- self-attention has `in_proj_bias`;
- cross-attention still uses configured `qkv_bias=false`, `qk_norm=false` for the default ManiFlow config;
- forward shape remains `[B,H,action_dim]`.

### 15.6 Policy point-permutation invariance

In eval mode with attention dropout disabled or eval semantics active:

- use the same point set;
- use the same state;
- use the same noisy action input;
- use the same timestep and target_t;
- independently permute points within each observation frame;
- permute XYZRGB channels together.

Verify the DiT-X velocity output is equal within a reasonable floating-point tolerance.

This test is important: after the fix, point order must not affect the policy output.

### 15.7 Consistency time-grid independence

Construct a decoder with:

```text
denoise_timesteps = 10
```

and vary inference steps, e.g.:

```text
1, 2, 4, 10
```

Verify the training discrete sampler remains based on the 10-step grid and is not changed by inference NFE.

### 15.8 Contrast semantics

For a contrast-only test case where clipping is not triggered, verify:

```text
mean(rgb_after) ~= mean(rgb_before)
```

and verify XYZ is unchanged by color augmentation.

### 15.9 End-to-end ManiFlow smoke test

Verify training loss and inference still run with the final contract:

```text
obs point cloud: [B,2,N,6]
obs state:       [B,2,19]
action:          [B,16,19]

cond:            [B,2*N,192]
pred_action:     [B,16,19]
control_action:  [B,8,19]
```

Use a smaller N / smaller hidden configuration in the smoke test if necessary for speed, while separately testing the default config values.

---

## 16. Compile / dtype / performance requirements

The implementation must remain compatible with:

- BF16 training;
- `torch.compile` on the action backbone;
- CUDA SDPA/cross-attention path;
- EMA model cloning;
- current optimizer grouping.

Avoid Python loops over points.

The NeRF PE must be vectorized.

Do not introduce CPU/Numpy operations into the model forward path.

Do not recompute static frequency tensors on every forward call.

Do not add observation self-attention.

Do not increase observation token count.

---

## 17. Safety checks against collateral changes

Before finalizing:

1. Search all `PointNetDense` construction sites.
2. Search all uses of `hidden_dims` for `pointnet_dense`.
3. Search all `ConsistencyDiTX` construction sites.
4. Search all state-dict/no-decay references to `context_pos_embed`.
5. Search all callers of `ConsistencyFlowMatch`.
6. Confirm new `denoise_timesteps` is wired through every relevant Hydra constructor.
7. Confirm changes to shared `PointColorJitter` match its documented R3D semantics and do not accidentally remove generic hue support.
8. Run the relevant existing DP3/R3D/SAT smoke/tests if shared augmentation code is touched.

Do not make unrelated cleanup changes discovered during these searches. Report them separately if necessary.

---

## 18. Acceptance criteria

The task is complete only when all of the following are true:

### Architecture

- [ ] XYZRGB PointNet matches official ManiFlow topology.
- [ ] PointNet remains dense and pooling-free.
- [ ] Point feature remains 128D.
- [ ] Adapt3R-style continuous XYZ PE is active by default for ManiFlow.
- [ ] XYZ PE uses 8 frequencies per axis, 48D before projection.
- [ ] XYZ PE is projected to 128D and added to PointNet features.
- [ ] State is still encoded to 64D and broadcast per point.
- [ ] Final observation token dim remains 192.
- [ ] DiT-X no longer contains flat point-index `context_pos_embed`.
- [ ] DiT-X uses frame-only observation PE.
- [ ] Self-attention uses official `nn.MultiheadAttention` semantics.
- [ ] Cross-attention remains action-query -> observation-K/V.

### Flow training

- [ ] `denoise_timesteps` exists separately from `num_inference_steps`.
- [ ] training `TimeSampler` uses `denoise_timesteps`.
- [ ] consistency target denominator does not depend on inference NFE.
- [ ] EMA consistency path remains intact.
- [ ] evaluation NFE override remains intact.

### Augmentation

- [ ] XYZ noise = 0.002.
- [ ] state noise = 0.0002.
- [ ] brightness = 0.125.
- [ ] contrast range corresponds to 0.5 around factor 1.0.
- [ ] contrast is centered around current RGB global mean.
- [ ] saturation = 0.5.
- [ ] ManiFlow hue = 0.
- [ ] no aggressive point dropout is newly enabled.
- [ ] current random-FPS inactivity for N=1024 is accurately documented.

### Behavior

- [ ] point permutation does not change policy output except floating-point noise.
- [ ] train/eval shapes are unchanged at the policy boundary.
- [ ] `pred_action` remains [B,16,19] in joint mode.
- [ ] `control_action` remains [B,8,19].
- [ ] BF16 and compile paths remain valid.
- [ ] EMA model creation/loading remains valid.

### Repository quality

- [ ] no unrelated model is modified without necessity.
- [ ] no unnecessary new abstraction layer is introduced.
- [ ] no branch is created.
- [ ] tests pass.
- [ ] comments/docstrings describe actual behavior rather than intended-but-inactive behavior.

---

## 19. Expected final report from Codex

After implementation, return a concise report containing:

1. Files changed.
2. Exact architecture changes.
3. Exact config changes.
4. Tests added/updated.
5. Commands/tests executed and results.
6. Any behavior intentionally left unchanged.
7. Any reference mismatch discovered during implementation.
8. Confirmation that no unrelated policy architecture was changed.
9. Confirmation that old ManiFlow checkpoints are not guaranteed to strict-load because the attention parameter names and observation positional embedding architecture changed.

Do not claim success without running the relevant tests/smoke checks.

---

## 20. Final design summary

The intended model after this task is:

```text
ManiFlow core
+ official ManiFlow XYZRGB dense PointNet
+ projected Adapt3R-style continuous XYZ positional encoding
+ frame-only observation temporal positional encoding
+ R3D-compatible XYZ/RGB/proprio augmentation
```

Formally, each point feature is:

```text
z_point =
    PointNetDense(XYZRGB)
    + LayerNorm(Linear(NeRF_XYZ_PE(XYZ)))
```

Then:

```text
z_obs =
    concat(z_point, StateMLP(joint_state))
```

DiT-X receives:

```text
context =
    Linear_192_to_768(z_obs)
    + FramePE(frame_index)
```

There must be no learned point-index positional embedding.

The ManiFlow action-generation core remains the reference architecture; the observation-side changes exist specifically to support XYZRGB dense point tokens and R3D-style randomized 3D observations without introducing arbitrary point-order dependence.
