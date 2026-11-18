"""Byte-level model candidates for the neural tier bake-off (float training, Phase 3a).

Every model maps bytes -> next-byte logits and supports chunked evaluation with carried
state: `forward(x, state) -> (logits, state)` with `state=None` at a lane's start. This is
how the coder runs them: lanes of hundreds of KB, processed in order.

`decode_macs()` estimates multiply-accumulates per decoded byte in a step-by-step decoder
(weights touched once per byte plus recurrent-state / attention-window work). It is the
x-axis of the bake-off: candidates are sized to similar decode cost.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def linear_macs(m: nn.Module) -> int:
    return sum(p.numel() for n, p in m.named_parameters() if p.dim() == 2 and "emb" not in n)


class MLP(nn.Module):
    def __init__(self, d, mult):
        super().__init__()
        self.norm = nn.RMSNorm(d)
        self.up = nn.Linear(d, mult * d, bias=False)
        self.down = nn.Linear(mult * d, d, bias=False)

    def forward(self, x):
        return x + self.down(F.gelu(self.up(self.norm(x))))


class ByteModel(nn.Module):
    """Embedding -> body -> norm -> head; subclasses implement body(x, state)."""

    def __init__(self, d):
        super().__init__()
        self.d = d
        self.emb = nn.Embedding(256, d)
        self.norm = nn.RMSNorm(d)
        self.head = nn.Linear(d, 256, bias=False)

    def forward(self, x, state=None):
        h, state = self.body(self.emb(x), state)
        return self.head(self.norm(h)), state

    def extra_macs(self) -> int:
        return 0

    def state_elems(self) -> int:
        """Recurrent state (or attention window) elements per lane, read+written per byte."""
        return 0

    def decode_macs(self) -> int:
        return linear_macs(self) + self.extra_macs()


# ------------------------------------------------------------------------- LSTM
class LSTMModel(ByteModel):
    def __init__(self, d=352, layers=2):
        super().__init__(d)
        self.rnn = nn.LSTM(d, d, num_layers=layers, batch_first=True)

    def body(self, x, state):
        return self.rnn(x, state)

    def state_elems(self):
        return 2 * self.rnn.num_layers * self.d


# ------------------------------------------------------------------------- minGRU
def _log_g(x):
    return torch.where(x >= 0, (F.relu(x) + 0.5).log(), -F.softplus(-x))


class MinGRU(nn.Module):
    """minGRU (Feng et al. 2024, "Were RNNs all we needed?"): gates depend only on the
    input, so training is a parallel scan (log space), decoding a cheap elementwise step."""

    def __init__(self, d):
        super().__init__()
        self.norm = nn.RMSNorm(d)
        self.proj = nn.Linear(d, 2 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x, h0):
        k, hh = self.proj(self.norm(x)).float().chunk(2, dim=-1)
        log_z = -F.softplus(-k)
        log_coeff = -F.softplus(k)  # log(1 - z)
        log_vals = log_z + _log_g(hh)
        if h0 is not None:
            log_vals = torch.cat([h0.clamp_min(1e-30).log().unsqueeze(1), log_vals], 1)
            log_coeff = torch.cat([torch.zeros_like(log_coeff[:, :1]), log_coeff], 1)
        a = log_coeff.cumsum(1)
        h = (a + (log_vals - a).logcumsumexp(1)).exp()
        if h0 is not None:
            h = h[:, 1:]
        return x + self.out(h.to(x.dtype)), h[:, -1]


class MinGRUModel(ByteModel):
    def state_elems(self):
        return len(self.rnn) * self.d

    def __init__(self, d=224, layers=5, mlp=3):
        super().__init__(d)
        self.rnn = nn.ModuleList(MinGRU(d) for _ in range(layers))
        self.mlp = nn.ModuleList(MLP(d, mlp) for _ in range(layers))

    def body(self, x, state):
        state = state or [None] * len(self.rnn)
        new = []
        for r, m, s in zip(self.rnn, self.mlp, state):
            x, hs = r(x, s)
            x = m(x)
            new.append(hs)
        return x, new


# ------------------------------------------------------------------------- delta rule (fla)
class GDNModel(ByteModel):
    """Gated DeltaNet layers (delta-rule fast weights: in-context learning of the file
    as it goes), from flash-linear-attention."""

    def __init__(self, d=256, layers=3, heads=4, head_dim=64, expand_v=1.0, mlp=2):
        super().__init__(d)
        from fla.layers import GatedDeltaNet
        self.heads, self.head_dim, self.expand_v = heads, head_dim, expand_v
        self.norms = nn.ModuleList(nn.RMSNorm(d) for _ in range(layers))
        self.mix = nn.ModuleList(GatedDeltaNet(hidden_size=d, num_heads=heads, head_dim=head_dim, expand_v=expand_v,
                                               mode="chunk", layer_idx=i) for i in range(layers))
        self.mlp = nn.ModuleList(MLP(d, mlp) for _ in range(layers))

    def body(self, x, state):
        from fla.models.utils import Cache
        cache = state if state is not None else Cache()
        for n, m, f in zip(self.norms, self.mix, self.mlp):
            x = x + m(n(x), past_key_values=cache, use_cache=True)[0]
            x = f(x)
        return x, cache

    def extra_macs(self):
        # Per head per byte: read-out S^T q, prediction S^T k, rank-1 update: ~3 dk*dv.
        dv = int(self.head_dim * self.expand_v)
        return len(self.mix) * self.heads * 3 * self.head_dim * dv

    def state_elems(self):
        return len(self.mix) * self.heads * self.head_dim * int(self.head_dim * self.expand_v)


class RWKV7Model(ByteModel):
    def __init__(self, d=256, layers=3, head_dim=64, mlp=2):
        super().__init__(d)
        from fla.layers import RWKV7Attention
        self.heads, self.head_dim = d // head_dim, head_dim
        self.norms = nn.ModuleList(nn.RMSNorm(d) for _ in range(layers))
        self.mix = nn.ModuleList(RWKV7Attention(mode="chunk", hidden_size=d, head_dim=head_dim, layer_idx=i,
                                                num_hidden_layers=layers) for i in range(layers))
        self.mlp = nn.ModuleList(MLP(d, mlp) for _ in range(layers))

    def body(self, x, state):
        from fla.models.utils import Cache
        cache = state if state is not None else Cache()
        v_first = None
        for n, m, f in zip(self.norms, self.mix, self.mlp):
            out = m(n(x), past_key_values=cache, use_cache=True, v_first=v_first)
            x = x + out[0]
            v_first = out[-1]
            x = f(x)
        return x, cache

    def extra_macs(self):
        return len(self.mix) * self.heads * 3 * self.head_dim * self.head_dim

    def state_elems(self):
        return len(self.mix) * self.heads * self.head_dim * self.head_dim


# ------------------------------------------------------------------------- sliding-window transformer
def rope(x, pos):
    d = x.shape[-1]
    inv = 1.0 / (10000 ** (torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d))
    ang = pos[:, None].float() * inv[None]
    cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2)


class Attn(nn.Module):
    def __init__(self, d, heads, window):
        super().__init__()
        self.h, self.window = heads, window
        self.norm = nn.RMSNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)

    def forward(self, x, mask):
        B, T, D = x.shape
        q, k, v = self.qkv(self.norm(x)).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        pos = torch.arange(T, device=x.device)
        q, k = rope(q, pos), rope(k, pos)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return x + self.o(y.transpose(1, 2).reshape(B, T, D))


class SWAModel(ByteModel):
    """Small transformer with sliding-window causal attention. State = the last `window`
    input bytes (re-encoded as a prefix of the next chunk)."""

    def __init__(self, d=192, layers=3, heads=3, window=512, mlp=4):
        super().__init__(d)
        self.window = window
        self.att = nn.ModuleList(Attn(d, heads, window) for _ in range(layers))
        self.mlp = nn.ModuleList(MLP(d, mlp) for _ in range(layers))

    def forward(self, x, state=None):
        P = 0
        if state is not None:
            x = torch.cat([state, x], 1)
            P = state.shape[1]
        T = x.shape[1]
        i = torch.arange(T, device=x.device)
        mask = (i[None, :] <= i[:, None]) & (i[:, None] - i[None, :] < self.window)
        h = self.emb(x)
        for a, m in zip(self.att, self.mlp):
            h = m(a(h, mask))
        logits = self.head(self.norm(h))[:, P:]
        return logits, x[:, -self.window:]

    def extra_macs(self):
        return len(self.att) * 2 * self.window * self.d  # q.K over the window + attn.V

    def state_elems(self):
        return len(self.att) * 2 * self.window * self.d


ARCHS = {"lstm": LSTMModel, "mingru": MinGRUModel, "gdn": GDNModel, "rwkv7": RWKV7Model, "swa": SWAModel}


def build(arch, **kw):
    return ARCHS[arch](**kw)


def describe(m):
    total = sum(p.numel() for p in m.parameters())
    return f"{type(m).__name__}: {total / 1e6:.2f}M params, ~{m.decode_macs() / 1e6:.2f}M decode MACs/byte"


if __name__ == "__main__":
    dev = "cuda"
    for a in ARCHS:
        m = build(a).to(dev)
        x = torch.randint(0, 256, (2, 1024), device=dev)
        with torch.autocast("cuda", torch.bfloat16):
            full, _ = m(x)
            l1, s = m(x[:, :512])
            l2, _ = m(x[:, 512:], s)
        err = (torch.cat([l1, l2], 1).float() - full.float()).abs().max().item()
        print(f"{describe(m):60s} chunked-vs-full max diff {err:.3g}")
        _ = math
