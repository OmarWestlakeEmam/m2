"""The network: subject adapters -> conv stem -> Conformer -> pooled word
vector, with four heads (word, meaning, sound, neighbours)."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import GENERIC, N_SUBJ
from .phon import FEATS, N_PH


class Spatial(nn.Module):
    """Shared sensor mixing + low-rank per-subject adapter + per-subject bias.
    Index GENERIC carries no adapter: it is the path for unseen subjects."""

    def __init__(self, cin: int, d: int, rank: int):
        super().__init__()
        self.shared = nn.Conv1d(cin, d, 1)
        self.U = nn.Parameter(torch.zeros(N_SUBJ + 1, cin, rank))
        self.V = nn.Parameter(torch.randn(N_SUBJ + 1, rank, d) / math.sqrt(rank))
        self.bias = nn.Parameter(torch.zeros(N_SUBJ + 1, d))

    def forward(self, x, sid):
        y = self.shared(x)
        on = (sid != GENERIC).float()[:, None, None]
        a = torch.einsum("bct,bcr->brt", x, self.U[sid])
        a = torch.einsum("brt,brd->bdt", a, self.V[sid])
        return y + on * (a + self.bias[sid][:, :, None])


class Res(nn.Module):
    def __init__(self, d, dil, drop):
        super().__init__()
        self.c = nn.Conv1d(d, d, 3, padding=dil, dilation=dil)
        self.n = nn.GroupNorm(8, d)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return x + self.drop(F.gelu(self.n(self.c(x))))


class Stem(nn.Module):
    def __init__(self, d, dils, win, frames, drop):
        super().__init__()
        s = win // frames
        k = 2 * s
        self.blocks = nn.Sequential(*[Res(d, dl, drop) for dl in dils])
        self.glu = nn.Conv1d(d, 2 * d, 3, padding=1)
        self.down = nn.Conv1d(d, d, k, stride=s, padding=(k - s + 1) // 2)
        self.frames = frames

    def forward(self, x):
        x = self.blocks(x)
        x = F.glu(self.glu(x), dim=1)
        return self.down(x)[..., : self.frames]


def rope(x, base=10000.0):
    """Rotary position embedding on (B, H, T, Dh)."""
    T, Dh = x.shape[-2], x.shape[-1]
    half = Dh // 2
    f = base ** (-torch.arange(half, device=x.device, dtype=torch.float32) / half)
    ang = torch.arange(T, device=x.device, dtype=torch.float32)[:, None] * f[None]
    cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
    x1, x2 = x[..., :half], x[..., half:2 * half]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos, x[..., 2 * half:]], -1)


class DropPath(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        keep = (torch.rand(x.shape[0], *([1] * (x.ndim - 1)), device=x.device) > self.p).to(x.dtype)
        return x * keep / (1 - self.p)


class FF(nn.Module):
    def __init__(self, d, h, drop):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, h), nn.SiLU(), nn.Dropout(drop),
                                 nn.Linear(h, d), nn.Dropout(drop))

    def forward(self, x):
        return self.net(x)


class MHSA(nn.Module):
    def __init__(self, d, heads, drop):
        super().__init__()
        self.h = heads
        self.n = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)
        self.drop = drop

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.n(x)).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        q, k = rope(q), rope(k)
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.drop if self.training else 0.0)
        return F.dropout(self.o(y.transpose(1, 2).reshape(B, T, D)), self.drop, self.training)


class ConvMod(nn.Module):
    def __init__(self, d, k, drop):
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.pw1 = nn.Linear(d, 2 * d)
        self.dw = nn.Conv1d(d, d, k, padding=k // 2, groups=d)
        self.n2 = nn.LayerNorm(d)
        self.pw2 = nn.Linear(d, d)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        y = F.glu(self.pw1(self.n1(x)), dim=-1)
        y = self.dw(y.transpose(1, 2)).transpose(1, 2)
        return self.drop(self.pw2(F.silu(self.n2(y))))


class Block(nn.Module):
    def __init__(self, d, heads, ff, k, drop, dp):
        super().__init__()
        self.f1, self.att, self.conv, self.f2 = FF(d, ff, drop), MHSA(d, heads, drop), ConvMod(d, k, drop), FF(d, ff, drop)
        self.n = nn.LayerNorm(d)
        self.dp = DropPath(dp)

    def forward(self, x):
        x = x + self.dp(0.5 * self.f1(x))
        x = x + self.dp(self.att(x))
        x = x + self.dp(self.conv(x))
        x = x + self.dp(0.5 * self.f2(x))
        return self.n(x)


class Pool(nn.Module):
    def __init__(self, d, out, heads=4):
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.att = nn.MultiheadAttention(d, heads, batch_first=True)
        self.proj = nn.Sequential(nn.Linear(d, out), nn.LayerNorm(out))

    def forward(self, x):
        y, _ = self.att(self.q.expand(x.shape[0], -1, -1), x, x, need_weights=False)
        return self.proj(y[:, 0])


class Net(nn.Module):
    def __init__(self, m: dict, n_vocab: int, d_emb: int, win: int, frames: int, cin: int = 306):
        super().__init__()
        d, out = m["d"], m["out"]
        self.spatial = Spatial(cin, d, m["rank"])
        self.stem = Stem(d, m["stem_dil"], win, frames, m["drop"])
        dps = [m["droppath"] * i / max(m["depth"] - 1, 1) for i in range(m["depth"])]
        self.enc = nn.ModuleList([Block(d, m["heads"], m["ff"], m["kernel"], m["drop"], p) for p in dps])
        self.pool = Pool(d, out)
        self.head_word = nn.Linear(out, n_vocab)
        self.head_mean = nn.Sequential(nn.Linear(out, 2 * out), nn.GELU(), nn.Linear(2 * out, d_emb))
        self.t = nn.Parameter(torch.tensor(math.log(10.0)))
        self.b = nn.Parameter(torch.tensor(-10.0))
        self.head_ph = nn.Linear(d, N_PH)
        self.head_ft = nn.Linear(d, len(FEATS))
        self.head_nb = nn.Linear(out, n_vocab)
        self.head_nx = nn.Linear(out, n_vocab + 1)

    def encode(self, x, sid):
        """x (B, C, T) normalised -> frames (B, F, d), word vector (B, out)."""
        f = self.stem(self.spatial(x, sid)).transpose(1, 2)
        for blk in self.enc:
            f = blk(f)
        return f, self.pool(f)

    def forward(self, x, sid):
        f, h = self.encode(x, sid)
        return dict(h=h, z=self.head_word(h), m=F.normalize(self.head_mean(h), dim=-1),
                    ph=self.head_ph(f), ft=self.head_ft(f), nb=self.head_nb(h), nx=self.head_nx(h))

    def mean_logits(self, m, E):
        """Meaning-head logits against target vectors E (V, d_emb), already normalised."""
        return self.t.exp() * m @ E.T + self.b
