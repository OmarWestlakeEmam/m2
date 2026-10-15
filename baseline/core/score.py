"""Turning a trained network into 50-way scores; combining ensemble members."""
from __future__ import annotations

import contextlib

import numpy as np
import torch
import torch.nn.functional as F

from .data import prep_batch
from .net import Net


def amp(device):
    if str(device).startswith("cuda"):
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def load_member(path: str, device) -> tuple[Net, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = Net(ck["cfg"]["model"], ck["n_vocab"], ck["d_emb"], ck["win"], ck["frames"])
    net.load_state_dict(ck["state"])
    net.to(device).eval()
    return net, ck


@torch.no_grad()
def outputs(net: Net, batches, Ev: torch.Tensor, Ev2: torch.Tensor | None, norm: dict, device):
    """batches yields (x (B, T, C) half, sid (B,)). Returns z, meaning logits (vocab, vocab2), h."""
    zs, bs, b2s, hs = [], [], [], []
    for x, sid in batches:
        x = prep_batch(x.to(device, non_blocking=True), norm["mode"], norm["clamp"])
        sid = sid.to(device)
        with amp(device):
            o = net(x, sid)
        m = o["m"].float()
        zs.append(o["z"].float().cpu())
        bs.append(net.mean_logits(m, Ev).float().cpu())
        if Ev2 is not None:
            b2s.append(net.mean_logits(m, Ev2).float().cpu())
        hs.append(o["h"].float().cpu())
    cat = lambda v: torch.cat(v).numpy() if v else None
    return cat(zs), cat(bs), cat(b2s), cat(hs)


def log_softmax(a: np.ndarray) -> np.ndarray:
    return F.log_softmax(torch.as_tensor(a, dtype=torch.float64), -1).numpy()


def member_lp(z: np.ndarray, b: np.ndarray | None, lam: float) -> np.ndarray:
    lp = log_softmax(z)
    if b is not None and lam:
        lp = lp + lam * log_softmax(b)
    return lp


def combine(lps: list[np.ndarray], tau: float, prior: np.ndarray) -> np.ndarray:
    """Average members' log-scores, subtract tau * log prior. Returns scores (N, V)."""
    return np.mean(lps, 0) - tau * np.log(prior)[None]


def probs(scores: np.ndarray) -> np.ndarray:
    return torch.softmax(torch.as_tensor(scores, dtype=torch.float64), -1).numpy()
