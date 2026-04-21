import torch
import torch.nn as nn


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
    ):
        super().__init__()
        out_dim = dim if out_dim is None else out_dim

        self.dim = dim
        self.out_dim = out_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.lambda_load = lambda_load
        self.beta_entropy = beta_entropy

        self.router = nn.Linear(dim, num_experts)
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

    def route(self, z):
        probs = torch.softmax(self.router(z), dim=-1)
        topk_prob, topk_idx = torch.topk(probs, k=self.top_k, dim=-1)
        topk_prob = topk_prob / topk_prob.sum(dim=-1, keepdim=True)
        return probs, topk_idx, topk_prob

    def mix(self, z, topk_idx, topk_prob):
        expert_out = torch.stack([expert(z) for expert in self.experts], dim=1)
        picked = torch.gather(
            expert_out,
            dim=1,
            index=topk_idx.unsqueeze(-1).expand(-1, -1, self.out_dim),
        )
        return (picked * topk_prob.unsqueeze(-1)).sum(dim=1)

    def aux_loss(self, probs):
        dispatch = probs.argmax(dim=-1)
        f_i = torch.bincount(dispatch, minlength=self.num_experts).to(probs.dtype)
        f_i = f_i / probs.shape[0]

        p_i = probs.mean(dim=0)

        load_loss = self.num_experts * torch.sum(f_i * p_i)
        entropy_loss = -(probs * torch.log(probs + 1e-9)).sum(dim=-1).mean()
        loss = self.lambda_load * load_loss + self.beta_entropy * entropy_loss

        return {
            "loss": loss,
            "load_balance_loss": load_loss,
            "entropy_loss": entropy_loss,
            "router_probs": probs,
            "dispatch": dispatch,
            "f_i": f_i,
            "p_i": p_i,
        }

    def forward(self, z, override_idx=None, return_aux=False):
        probs, topk_idx, topk_prob = self.route(z)

        if override_idx is not None:
            topk_idx = override_idx[:, None]
            topk_prob = torch.ones(z.shape[0], 1, device=z.device, dtype=z.dtype)

        z_moe = self.mix(z, topk_idx, topk_prob)

        if not return_aux:
            return z_moe

        aux = self.aux_loss(probs)
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

    return {
        "z": z,
        "aux": aux,
        "z_override": z_override,
        "aux_override": aux_override,
    }


if __name__ == "__main__":
    example()