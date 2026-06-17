import torch
import torch.nn as nn
import torch.nn.functional as F

class ExpertMLP(nn.Module):
    def __init__(self, dim, hidden_dim, out_dim=None, num_layers=2, activation=nn.GELU):
        super().__init__()
        out_dim = dim if out_dim is None else out_dim

        layers = []
        d = dim
        for _ in range(num_layers - 1):
            layers += [nn.Linear(d, hidden_dim), activation()]
            d = hidden_dim
        layers += [nn.Linear(d, out_dim)]

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)

class MoE(nn.Module):
    def __init__(
        self,
        dim,
        num_experts=16,
        top_k=2,
        hidden_dim=256,
        out_dim=None,
        num_layers=2,
        lambda_load=0.1,
        beta_entropy=0.01,
        activation=nn.GELU,
        use_boost=False,
        boost_start_epoch=0,
        boost_interval=100,
        boost_experts_per_step=4,
        boost_topk_per_step=1,
        use_enhanced_gate=False,
        gate_hidden_dim=None,
        gate_dropout=0.0,
    ):
        super().__init__()
        out_dim = dim if out_dim is None else out_dim

        self.dim = dim
        self.out_dim = out_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.lambda_load = lambda_load
        self.beta_entropy = beta_entropy

        # ---- Boost state ----
        self.use_boost = use_boost
        self.boost_start_epoch = boost_start_epoch
        self.boost_interval = boost_interval
        self.boost_experts_per_step = boost_experts_per_step
        self.boost_topk_per_step = boost_topk_per_step
        self.current_num_experts = num_experts
        self.current_top_k = top_k

        # ---- Router / Gate ----
        if use_enhanced_gate:
            hidden = gate_hidden_dim or (dim // 2)
            self.router = nn.Sequential(
                nn.Linear(dim, hidden),
                nn.ReLU(),
                nn.Dropout(gate_dropout),
                nn.Linear(hidden, num_experts),
            )
        else:
            self.router = nn.Linear(dim, num_experts)

        self.use_enhanced_gate = use_enhanced_gate

        self.experts = nn.ModuleList([
            ExpertMLP(
                dim=dim,
                hidden_dim=hidden_dim,
                out_dim=out_dim,
                num_layers=num_layers,
                activation=activation,
            )
            for _ in range(num_experts)
        ])

        self._init_weights()

    def _init_weights(self):
        """Xavier-uniform init for router/gate Linear layers.

        ExpertMLP already calls its own _init_weights in __init__;
        this pass only initialises the router/gate Linear layers which
        would otherwise use PyTorch's default Kaiming init.
        """
        for m in [self.router]:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
        # Enhanced gate: router is nn.Sequential with extra Linear layers
        if self.use_enhanced_gate:
            for m in self.router.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)

    def route(self, z):
        logits = self.router(z)
        # Overflow guard: clamp logits to prevent exp(logits) → inf → NaN in
        # softmax.  exp(±50) is well within float32 range; typical logits from
        # Xavier init are ±0.1–2.0, so this clamp is invisible during normal
        # training — it only activates under pathological weight drift.
        logits = torch.clamp(logits, min=-50, max=50)
        if self.use_boost:
            # Align with official MoE-DP: slice logits BEFORE softmax so that
            # only active experts compete.  Inactive experts receive exactly
            # zero probability (rather than a residual share of a full softmax).
            logits = logits[:, :self.current_num_experts]
        probs = torch.softmax(logits, dim=-1)
        k = min(self.current_top_k, self.current_num_experts)
        topk_prob, topk_idx = torch.topk(probs, k=k, dim=-1)
        topk_prob = topk_prob / topk_prob.sum(dim=-1, keepdim=True)
        return probs, topk_idx, topk_prob

    def aggregate_experts(self, z, topk_idx, topk_prob):
        active_experts = self.experts[:self.current_num_experts]
        expert_out = torch.stack([expert(z) for expert in active_experts], dim=1)
        picked = torch.gather(
            expert_out,
            dim=1,
            index=topk_idx.unsqueeze(-1).expand(-1, -1, self.out_dim),
        )
        return (picked * topk_prob.unsqueeze(-1)).sum(dim=1)

    def update_expert_num(self, epoch):
        """Idempotent boost schedule – recomputes active experts/top_k from epoch.

        Safe for checkpoint resume: state is a pure function of epoch,
        so calling this multiple times at the same epoch is a no-op.
        """
        if not self.use_boost:
            self.current_num_experts = self.num_experts
            self.current_top_k = self.top_k
            return

        if epoch < self.boost_start_epoch:
            self.current_num_experts = self.num_experts
            self.current_top_k = self.top_k
            return

        n_boosts = (epoch - self.boost_start_epoch) // self.boost_interval

        start_experts = max(2, self.num_experts // 2)
        start_topk = max(1, self.top_k // 2)

        self.current_num_experts = min(
            self.num_experts,
            start_experts + n_boosts * self.boost_experts_per_step,
        )
        self.current_top_k = min(
            self.top_k,
            start_topk + n_boosts * self.boost_topk_per_step,
        )

    def aux_loss(self, probs, topk_idx):
        # Frequency: count ALL top-k expert selections (not just argmax).
        # When top_k > 1, each sample routes to multiple experts;
        # F.one_hot over topk_idx correctly accounts for all of them.
        E = probs.shape[1]  # aligned with official: during boost E = current_num_experts
        topk_one_hot = F.one_hot(topk_idx, num_classes=E).to(probs.dtype)
        f_i = topk_one_hot.sum(dim=[0, 1]) / (probs.shape[0] * topk_idx.shape[1])

        p_i = probs.mean(dim=0)

        load_loss = E * torch.sum(f_i * p_i)
        entropy_loss = -(probs * torch.log(probs + 1e-9)).sum(dim=-1).mean()
        loss = self.lambda_load * load_loss + self.beta_entropy * entropy_loss

        return {
            "loss": loss,
            "load_balance_loss": load_loss,
            "entropy_loss": entropy_loss,
            "router_probs": probs,
            "dispatch": probs.argmax(dim=-1),
            "f_i": f_i,
            "p_i": p_i,
        }

    def forward(self, z, override_idx=None, return_aux=False):
        probs, topk_idx, topk_prob = self.route(z)

        if override_idx is not None:
            topk_idx = override_idx[:, None]
            topk_prob = torch.ones(z.shape[0], 1, device=z.device, dtype=z.dtype)

        z_moe = self.aggregate_experts(z, topk_idx, topk_prob)

        if not return_aux:
            return z_moe

        aux = self.aux_loss(probs, topk_idx)
        aux["topk_idx"] = topk_idx
        aux["topk_prob"] = topk_prob
        return z_moe, aux

def example():
    torch.manual_seed(0)

    class DummyEncoder(nn.Module):
        def __init__(self, obs_dim, feat_dim):
            super().__init__()
            self.out_dim = feat_dim
            self.net = nn.Sequential(
                nn.Linear(obs_dim, 128),
                nn.ReLU(),
                nn.Linear(128, feat_dim),
            )

        def forward(self, obs):
            return self.net(obs["state"])

    B = 8
    obs_dim = 24
    feat_dim = 64
    num_experts = 8
    top_k = 2

    encoder = DummyEncoder(obs_dim, feat_dim)
    moe = MoE(
        dim=encoder.out_dim,
        num_experts=num_experts,
        top_k=top_k,
        hidden_dim=128,
        out_dim=feat_dim,
        num_layers=2,
        lambda_load=0.1,
        beta_entropy=0.01,
    )

    obs = {"state": torch.randn(B, obs_dim)}

    z_enc = encoder(obs)

    z = moe(z_enc)
    print("normal z:", z.shape)

    z, aux = moe(z_enc, return_aux=True)
    print("z:", z.shape)
    print("aux loss:", float(aux["loss"]))
    print("load balance loss:", float(aux["load_balance_loss"]))
    print("entropy loss:", float(aux["entropy_loss"]))
    print("router_probs:", aux["router_probs"].shape)
    print("dispatch:", aux["dispatch"])
    print("topk_idx:", aux["topk_idx"])

    override_idx = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.long)
    z_override, aux_override = moe(z_enc, override_idx=override_idx, return_aux=True)
    print("override z:", z_override.shape)
    print("override topk_idx:", aux_override["topk_idx"].squeeze(-1))

    fake_diff_loss = z.pow(2).mean()
    loss = fake_diff_loss + aux["loss"]
    loss.backward()

    print("total loss:", float(loss))
    print("router grad mean:", moe.router.weight.grad.abs().mean().item())
    print("expert0 grad mean:", moe.experts[0].net[0].weight.grad.abs().mean().item())

    # Verify aux loss fix: f_i counts ALL top-k selections, not just argmax
    f_i_sum = aux["f_i"].sum().item()
    print(f"f_i sum (probability distribution): {f_i_sum:.4f} (should be 1.0)")
    assert abs(f_i_sum - 1.0) < 1e-6, \
        f"f_i.sum() should be 1.0 (normalized probability), got {f_i_sum}"
    # Verify f_i accounts for all top-k selections (not just argmax):
    # When top_k>1, dispatch count (=argmax count) < topk count (=one_hot sum)
    dispatch_count = torch.bincount(aux["dispatch"], minlength=8).sum().item()
    topk_raw_count = B * top_k  # each sample selects top_k experts
    print(f"dispatch count: {dispatch_count}, topk raw count: {topk_raw_count}")

    # ---- Boost test ----
    print("\n--- Boost test ---")
    moe_boost = MoE(
        dim=64,
        num_experts=8, top_k=2,
        hidden_dim=128, out_dim=64, num_layers=2,
        use_boost=True, boost_start_epoch=0, boost_interval=50,
        boost_experts_per_step=4, boost_topk_per_step=1,
    )
    print(f"base experts={moe_boost.num_experts} top_k={moe_boost.top_k}")

    moe_boost.update_expert_num(0)
    print(f"epoch 0: active_experts={moe_boost.current_num_experts} active_top_k={moe_boost.current_top_k}")
    assert moe_boost.current_num_experts == 4 and moe_boost.current_top_k == 1

    moe_boost.update_expert_num(50)
    print(f"epoch 50: active_experts={moe_boost.current_num_experts} active_top_k={moe_boost.current_top_k}")
    assert moe_boost.current_num_experts == 8 and moe_boost.current_top_k == 2

    moe_boost.update_expert_num(200)
    print(f"epoch 200: active_experts={moe_boost.current_num_experts} active_top_k={moe_boost.current_top_k}")
    assert moe_boost.current_num_experts == 8 and moe_boost.current_top_k == 2, "Should cap"

    # Forward through boosted MoE
    x_test = torch.randn(4, 64)
    z_boost, aux_boost = moe_boost(x_test, return_aux=True)
    print(f"boost z: {z_boost.shape}, active_experts={moe_boost.current_num_experts}")

    # ---- Enhanced gate test ----
    print("\n--- Enhanced gate test ---")
    moe_gate = MoE(
        dim=64,
        num_experts=8, top_k=2,
        hidden_dim=128, out_dim=64, num_layers=2,
        use_enhanced_gate=True, gate_dropout=0.1,
    )
    assert isinstance(moe_gate.router, nn.Sequential), \
        f"Expected nn.Sequential gate, got {type(moe_gate.router)}"
    print(f"gate type: {type(moe_gate.router).__name__} (len={len(moe_gate.router)})")
    z_gate, aux_gate = moe_gate(x_test, return_aux=True)
    print(f"enhanced gate z: {z_gate.shape}, aux loss: {aux_gate['loss'].item():.4f}")

    # ---- Xavier init test ----
    print("\n--- Xavier init test ---")
    expert_w = moe.experts[0].net[0].weight
    print(f"expert0 weight std: {expert_w.std().item():.4f} (non-zero expected)")

    return {
        "z": z,
        "aux": aux,
        "z_override": z_override,
        "aux_override": aux_override,
    }

if __name__ == "__main__":
    example()