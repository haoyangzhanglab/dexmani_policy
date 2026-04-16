import torch
import torch.nn as nn
import torch.nn.functional as F


class StateFusion(nn.Module):
    """State → feature fusion.

    Supports 2D (B, C) or 3D (B, seq_len, C) inputs for both
    feat and state. Output is always (B, seq_len, out_dim).

    Modes:
        concat : state → MLP(hidden), cat(feat, state)        → out = feat_dim + hidden
        film   : state → γ/β MLP, feat * γ + β               → out = feat_dim
        xattn  : feat→Q, state→K,V, cross-attend state       → out = feat_dim
    """

    VALID_MODES = ("concat", "film", "xattn")

    def __init__(
        self,
        mode: str = "concat",
        feat_dim: int = 128,
        state_dim: int = 19,
        hidden_dim: int = 64,
    ):
        super().__init__()
        if mode not in self.VALID_MODES:
            raise ValueError(f"mode must be one of {self.VALID_MODES}, got '{mode}'")

        self.mode = mode
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim

        if mode == "concat":
            self.state_mlp = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

        elif mode == "film":
            self.state_mlp = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 2 * feat_dim),
            )

        elif mode == "xattn":
            head_dim = hidden_dim
            self.feat_proj = nn.Linear(feat_dim, head_dim)
            self.state_k_proj = nn.Linear(state_dim, head_dim)
            self.state_v_proj = nn.Linear(state_dim, head_dim)
            self.out_proj = nn.Linear(head_dim, feat_dim)

    @property
    def out_dim(self) -> int:
        if self.mode == "concat":
            return self.feat_dim + self.hidden_dim
        return self.feat_dim

    @staticmethod
    def as_sequence(x: torch.Tensor) -> torch.Tensor:
        """(B, C) → (B, 1, C); (B, seq, C) → unchanged."""
        if x.ndim == 2:
            return x.unsqueeze(1)
        return x

    def forward(self, feat: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        ft = self.as_sequence(feat)
        st = self.as_sequence(state)

        # Broadcast state to match feat sequence length
        if st.size(1) == 1 and ft.size(1) > 1:
            st = st.expand(-1, ft.size(1), -1)

        if self.mode == "concat":
            state_feat = self.state_mlp(st)
            return torch.cat([ft, state_feat], dim=-1)

        if self.mode == "film":
            params = self.state_mlp(st)
            gamma, beta = params.chunk(2, dim=-1)
            return ft * gamma + beta

        # xattn: feat queries, state provides key/value → output matches feat sequence length
        q = self.feat_proj(ft)              # (B, seq_f, head_dim)
        k = self.state_k_proj(st)           # (B, seq_s, head_dim)
        v = self.state_v_proj(st)           # (B, seq_s, head_dim)
        out = F.scaled_dot_product_attention(q, k, v)  # (B, seq_f, head_dim)
        return self.out_proj(out)


def example():
    B, seq_len = 4, 128
    state_dim = 19

    print("=== StateFusion Example ===\n")

    for mode in ("concat", "film", "xattn"):
        fusion = StateFusion(mode=mode, feat_dim=128, state_dim=state_dim, hidden_dim=64)

        # Case 1: 3D feat + 3D state (state seq_len=1)
        feat_3d = torch.randn(B, seq_len, 128)
        st_3d = torch.randn(B, 1, state_dim)
        out1 = fusion(feat_3d, st_3d)
        print(f"[{mode}]  3D feat + 3D state (s=1)")
        print(f"  feat:{tuple(feat_3d.shape)} state:{tuple(st_3d.shape)} → out:{tuple(out1.shape)} (out_dim={fusion.out_dim})")

        # Case 2: 2D feat + 2D state
        feat_2d = torch.randn(B, 128)
        st_2d = torch.randn(B, state_dim)
        out2 = fusion(feat_2d, st_2d)
        print(f"[{mode}]  2D feat + 2D state")
        print(f"  feat:{tuple(feat_2d.shape)} state:{tuple(st_2d.shape)} → out:{tuple(out2.shape)} (out_dim={fusion.out_dim})")

        # Case 3: 2D state + 3D feat
        out3 = fusion(feat_3d, st_2d)
        print(f"[{mode}]  3D feat + 2D state")
        print(f"  feat:{tuple(feat_3d.shape)} state:{tuple(st_2d.shape)} → out:{tuple(out3.shape)} (out_dim={fusion.out_dim})")

        # Case 4: 3D state matching feat seq_len
        st_match = torch.randn(B, seq_len, state_dim)
        out4 = fusion(feat_3d, st_match)
        print(f"[{mode}]  3D feat + 3D state (s={seq_len})")
        print(f"  feat:{tuple(feat_3d.shape)} state:{tuple(st_match.shape)} → out:{tuple(out4.shape)} (out_dim={fusion.out_dim})")
        print()


if __name__ == "__main__":
    example()
