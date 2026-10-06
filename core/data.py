"""Training data: memory-mapped windows, label tensors, sampling, GPU-side
normalisation and augmentation. The same `prep_batch` is used for scoring."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from .common import load_json

N_SUBJ = 33          # subjects 0..32 have their own adapter
GENERIC = 33         # index of the subject-free path (unseen subjects)


def sid_of(subject: int) -> int:
    return int(subject) if 0 <= int(subject) < N_SUBJ else GENERIC


class Store:
    def __init__(self, cache_dir: str):
        c = Path(cache_dir)
        self.meta = load_json(c / "meta.json")
        z = np.load(c / "index.npz")
        self.idx = {k: z[k] for k in z.files}
        self.xdir = c / "x"
        self.win = int(self.meta["win"])
        self.T = np.array([r["T"] for r in self.meta["runs"]], dtype=np.int64)

    def rows(self, split: int, who: str = "all") -> np.ndarray:
        m = self.idx["split"] == split
        if who == "deep":
            m &= self.idx["subj"] == 0
        elif who == "broad":
            m &= self.idx["subj"] > 0
        elif who != "all":
            m &= np.isin(self.idx["subj"], [int(s) for s in str(who).split(",")])
        return np.flatnonzero(m)

    def lead(self) -> np.ndarray:
        """Samples between each word's onset and the onset of the first word of its
        sentence (scoring sentences start at their first word, so nothing earlier exists)."""
        if getattr(self, "_lead", None) is None:
            I = self.idx
            st, sent, rid = I["start"].astype(np.int64), I["sent"].astype(np.int64), I["rid"].astype(np.int64)
            lead = np.zeros(len(st), np.int64)
            m = sent >= 0
            key = rid[m] * 10_000_000 + sent[m]
            u, inv = np.unique(key, return_inverse=True)
            first = np.full(len(u), np.iinfo(np.int64).max)
            np.minimum.at(first, inv, st[m])
            lead[m] = st[m] - first[inv]
            self._lead = lead
        return self._lead

    def prior(self, who: str) -> np.ndarray:
        p = np.asarray(self.meta["prior_deep" if who == "deep" else "prior_broad"], np.float64)
        if p.sum() == 0:
            p = np.asarray(self.meta["prior_deep"], np.float64)
        return (p + 1) / (p + 1).sum()


class Windows(Dataset):
    """Yields (x float16 (win, 306), sid, vid, wid, nxt, nb (50,), ph (frames,))."""

    def __init__(self, store: Store, rows: np.ndarray, jitter: int = 0, avg_p: float = 0.0,
                 avg_k: tuple[int, int] = (2, 4), pre: int = 0, pre_mask: float = 0.0):
        self.s, self.rows, self.jitter = store, np.asarray(rows), int(jitter)
        self.pre, self.pre_mask = int(pre), float(pre_mask)
        self.L = self.s.win + self.pre
        self.lead = store.lead() if self.pre else None
        self.avg_p, self.avg_k = float(avg_p), avg_k
        self._mm: dict = {}
        self._groups = None
        if avg_p > 0:
            self._build_groups()

    def _build_groups(self):
        I = self.s.idx
        r = self.rows[I["vid"][self.rows] >= 0]
        key = I["subj"][r].astype(np.int64) * 100 + I["vid"][r].astype(np.int64)
        order = np.argsort(key, kind="stable")
        key, r = key[order], r[order]
        cuts = np.flatnonzero(np.diff(key)) + 1
        self._groups = {int(k[0]): g for k, g in zip(np.split(key, cuts), np.split(r, cuts))}

    def __len__(self):
        return len(self.rows)

    def _x(self, r: int, rng) -> np.ndarray:
        I = self.s.idx
        rid = int(I["rid"][r])
        mm = self._mm.get(rid)
        if mm is None:
            mm = self._mm[rid] = np.load(self.s.xdir / f"{rid}.npy", mmap_mode="r")
        st = int(I["start"][r]) - self.pre
        if self.jitter:
            st += int(rng.integers(-self.jitter, self.jitter + 1))
        T, L = int(self.s.T[rid]), self.L
        if st >= 0 and st + L <= T:
            return mm[st:st + L]
        out = np.zeros((L, mm.shape[1]), dtype=np.float16)   # zero = session mean, as in the scoring windows
        a, b = max(st, 0), min(st + L, T)
        if b > a:
            out[a - st:b - st] = mm[a:b]
        return out

    def __getitem__(self, i):
        I = self.s.idx
        r = int(self.rows[i])
        rng = np.random.default_rng()
        x = np.asarray(self._x(r, rng), dtype=np.float16)
        if self._groups is not None and I["vid"][r] >= 0 and rng.random() < self.avg_p:
            g = self._groups.get(int(I["subj"][r]) * 100 + int(I["vid"][r]))
            if g is not None and len(g) > 1:
                k = int(rng.integers(self.avg_k[0], self.avg_k[1] + 1)) - 1
                others = rng.choice(g, size=min(k, len(g)), replace=False)
                acc = x.astype(np.float32)
                for o in others:
                    acc += np.asarray(self._x(int(o), rng), dtype=np.float32)
                x = (acc / (1 + len(others))).astype(np.float16)
        if self.pre:
            cut = self.pre - int(self.lead[r])                 # samples before the sentence's first word
            if self.pre_mask and rng.random() < self.pre_mask:
                cut = self.pre                                  # like the scoring set's isolated 1 s word windows
            if cut > 0:
                x = np.array(x, copy=True)
                x[:cut] = 0
        return (torch.from_numpy(np.ascontiguousarray(x)),
                sid_of(I["subj"][r]), int(I["vid"][r]), int(I["wid"][r]), int(I["nxt"][r]),
                torch.from_numpy(I["nb"][r].astype(np.float32)),
                torch.from_numpy(I["ph"][r].astype(np.int64)))


def sampler(store: Store, rows: np.ndarray, n: int, deep_share: float, vocab_boost: float, seed: int):
    I = store.idx
    deep = I["subj"][rows] == 0
    w = np.where(deep, deep_share / max(deep.sum(), 1), (1 - deep_share) / max((~deep).sum(), 1))
    if not deep.any() or deep.all():
        w = np.ones(len(rows))
    w = w * np.where(I["vid"][rows] >= 0, vocab_boost, 1.0)
    g = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), num_samples=n, replacement=True, generator=g)


# --------------------------------------------------------------------------- GPU side

def prep_batch(x: torch.Tensor, mode: str = "inst", clamp: float = 5.0) -> torch.Tensor:
    """(B, T, C) half -> (B, C, T) float, normalised per window and channel."""
    x = x.float().transpose(1, 2)
    if mode == "inst":
        x = (x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True) + 1e-4)
    elif mode == "robust":
        q = torch.quantile(x, torch.tensor([0.25, 0.5, 0.75], device=x.device), dim=-1, keepdim=True)
        x = (x - q[1]) / ((q[2] - q[0]) / 1.349 + 1e-4)
    elif mode != "sess":
        raise ValueError(mode)
    return x.clamp(-clamp, clamp)


def augment(x: torch.Tensor, a: dict) -> torch.Tensor:
    B, C, T = x.shape
    dev = x.device
    if a.get("chan_drop", 0) > 0:
        x = x * (torch.rand(B, C, 1, device=dev) > a["chan_drop"]).float()
    if a.get("scale", 0) > 0:
        x = x * (1 + (torch.rand(B, C, 1, device=dev) * 2 - 1) * a["scale"])
    if a.get("noise", 0) > 0:
        x = x + torch.randn_like(x) * a["noise"]
    if a.get("tmask_p", 0) > 0:
        L = int(a.get("tmask_max", 20))
        on = torch.rand(B, device=dev) < a["tmask_p"]
        ln = torch.randint(1, L + 1, (B,), device=dev)
        st = (torch.rand(B, device=dev) * (T - ln)).long()
        t = torch.arange(T, device=dev)[None]
        m = on[:, None] & (t >= st[:, None]) & (t < (st + ln)[:, None])
        x = x.masked_fill(m[:, None, :], 0.0)
    return x
