"""Optional sentence-context model (use only if the rules allow it).

A small transformer reads the pooled word vectors of all scored words in one
sentence and corrects each word's logits. Features must be out-of-fold: train a
base network with `train.fold=k/K`, extract features for fold k, fit on them.

    python -m core.train -c cfg/a.yaml -s train.name=f0 train.fold=0/2
    python -m core.ctx extract -c cfg/a.yaml --member d/r/f0/best.pt --fold 0/2 --name c0
    python -m core.ctx fit     -c cfg/a.yaml --name c0
    python -m core.ctx val     -c cfg/a.yaml --name c0 --who deep     # -> npz for calib --extra
    python -m core.ctx predict -c cfg/a.yaml --name c0 --track deep   # -> npz for infer --extra
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from . import ho
from .common import bacc_at_k, load_cfg, log, seed_all
from .data import Store, Windows
from .score import load_member, log_softmax, outputs
from .train import row_fold


class CtxNet(nn.Module):
    def __init__(self, din, n_vocab, d=384, depth=4, heads=6, max_len=64, drop=0.1):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(din + n_vocab, d), nn.LayerNorm(d))
        self.pos = nn.Parameter(torch.zeros(1, max_len, d))
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, drop, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, depth)
        self.out = nn.Linear(d, n_vocab)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, h, z, pad):
        """h (B, L, din), z (B, L, V) base logits, pad (B, L) True = padding."""
        u = self.inp(torch.cat([h, F.log_softmax(z, -1)], -1)) + self.pos[:, : h.shape[1]]
        return z + self.out(self.tf(u, src_key_padding_mask=pad))


def groups_from_rows(store: Store, rows: np.ndarray, vocab_only: bool) -> list[np.ndarray]:
    """Sentences as lists of positions into `rows`, in word order."""
    I = store.idx
    keep = np.arange(len(rows)) if not vocab_only else np.flatnonzero(I["vid"][rows] >= 0)
    s, p = I["sent"][rows][keep], I["pos"][rows][keep]
    single = s < 0
    out = [np.array([i]) for i in keep[single]]
    k2, s2, p2 = keep[~single], s[~single], p[~single]
    o = np.lexsort((p2, s2))
    k2, s2 = k2[o], s2[o]
    cuts = np.flatnonzero(np.diff(s2)) + 1
    out += [g for g in np.split(k2, cuts) if len(g)]
    return out


def chunk(groups, L):
    out = []
    for g in groups:
        for a in range(0, len(g), L):
            out.append(g[a:a + L])
    return out


def pad_batch(groups, H, Z, Y=None):
    L = max(len(g) for g in groups)
    B = len(groups)
    h = np.zeros((B, L, H.shape[1]), np.float32)
    z = np.zeros((B, L, Z.shape[1]), np.float32)
    pad = np.ones((B, L), bool)
    y = np.full((B, L), -100, np.int64)
    for i, g in enumerate(groups):
        h[i, : len(g)], z[i, : len(g)], pad[i, : len(g)] = H[g], Z[g], False
        if Y is not None:
            y[i, : len(g)] = Y[g]
    return map(torch.as_tensor, (h, z, pad, y))


@torch.no_grad()
def run_ctx(net, groups, H, Z, device, bs=256):
    out = np.zeros_like(Z)
    net.eval()
    for a in range(0, len(groups), bs):
        gs = groups[a:a + bs]
        h, z, pad, _ = pad_batch(gs, H, Z)
        o = net(h.to(device), z.to(device), pad.to(device)).float().cpu().numpy()
        for i, g in enumerate(gs):
            out[g] = o[i, : len(g)]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["extract", "fit", "val", "predict"])
    ap.add_argument("-c", "--cfg", default="cfg/a.yaml")
    ap.add_argument("-s", "--set", nargs="*", default=[])
    ap.add_argument("--name", required=True)
    ap.add_argument("--member")
    ap.add_argument("--fold")
    ap.add_argument("--who", choices=["deep", "broad"])
    ap.add_argument("--track", choices=["deep", "broad"])
    args = ap.parse_args()
    cfg = load_cfg(args.cfg, args.set)
    cc = cfg["ctx"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = Path(cfg["paths"]["runs"]) / args.name
    d.mkdir(parents=True, exist_ok=True)
    store = Store(cfg["paths"]["cache"])
    seed_all(cfg["train"]["seed"])

    if args.cmd == "extract":
        k, K = map(int, args.fold.split("/"))
        fold = row_fold(store, K)
        I = store.idx
        rows = np.flatnonzero(((I["split"] == 0) & (fold == k)) | (I["split"] > 0))
        net, ck = load_member(args.member, device)
        dl = DataLoader(Windows(store, rows, pre=int(ck.get("pre", 0))), batch_size=512, num_workers=cfg["train"]["workers"])
        z, _, _, h = outputs(net, ((t[0], t[1]) for t in dl), torch.as_tensor(ck["Ev"], device=device),
                             None, ck["cfg"]["norm"], device)
        np.savez(d / "feat.npz", rows=rows, h=h.astype(np.float16), z=z.astype(np.float32), member=args.member)
        log(f"features for {len(rows)} rows -> {d / 'feat.npz'}")
        return

    if args.cmd == "fit":
        f = np.load(d / "feat.npz")
        rows, H, Z = f["rows"], f["h"].astype(np.float32), f["z"]
        I = store.idx
        Y = I["vid"][rows].astype(np.int64)
        tr = np.flatnonzero(I["split"][rows] == 0)
        va = np.flatnonzero((I["split"][rows] == 1) & (I["subj"][rows] == 0))
        gtr = chunk([tr[g] for g in groups_from_rows(store, rows[tr], cc["vocab_only"])], cc["max_len"])
        gva = chunk([va[g] for g in groups_from_rows(store, rows[va], cc["vocab_only"])], cc["max_len"])
        log(f"{len(gtr)} train sequences, {len(gva)} val sequences")
        net = CtxNet(H.shape[1], Z.shape[1], cc["d"], cc["depth"], cc["heads"], cc["max_len"]).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=cc["lr"], weight_decay=0.05)
        prior = store.prior("deep")
        lp = torch.as_tensor(np.log(prior), dtype=torch.float32, device=device)
        rng = np.random.default_rng(0)
        best = -1
        base = bacc_at_k(Z[va][Y[va] >= 0], Y[va][Y[va] >= 0], 10)
        for ep in range(cc["epochs"]):
            net.train()
            seqs = []
            for g in gtr:  # some sentences become isolated words, like the scoring set's word rows
                seqs += [g[i:i + 1] for i in range(len(g))] if rng.random() < cc["p_single"] else [g]
            order = rng.permutation(len(seqs))
            for a in range(0, len(order), cc["batch"]):
                h, z, pad, y = pad_batch([seqs[i] for i in order[a:a + cc["batch"]]], H, Z, Y)
                h, z, pad, y = h.to(device), z.to(device), pad.to(device), y.to(device)
                h = h + torch.randn_like(h) * cc["noise"]
                o = net(h, z, pad)
                m = y >= 0
                loss = F.cross_entropy(o[m] + lp, y[m], label_smoothing=0.1)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
            O = run_ctx(net, gva, H, Z, device)
            ok = Y[va] >= 0
            v = bacc_at_k(O[va][ok], Y[va][ok], 10)
            log(f"ctx epoch {ep}: val bacc10 {v:.4f} (base {base:.4f})")
            if v > best:
                best = v
                torch.save(dict(state=net.state_dict(), cc=cc, din=H.shape[1], nv=Z.shape[1],
                                member=str(f["member"])), d / "ctx.pt")
        log(f"best ctx val bacc10 {best:.4f} -> {d / 'ctx.pt'}")
        return

    ck = torch.load(d / "ctx.pt", map_location="cpu", weights_only=False)
    net = CtxNet(ck["din"], ck["nv"], ck["cc"]["d"], ck["cc"]["depth"], ck["cc"]["heads"], ck["cc"]["max_len"]).to(device)
    net.load_state_dict(ck["state"])

    if args.cmd == "val":
        f = np.load(d / "feat.npz")
        rows, H, Z = f["rows"], f["h"].astype(np.float32), f["z"]
        I = store.idx
        want = (I["split"][rows] == 1) & ((I["subj"][rows] == 0) if args.who == "deep" else (I["subj"][rows] > 0))
        sel = np.flatnonzero(want)
        g = chunk([sel[x] for x in groups_from_rows(store, rows[sel], cc["vocab_only"])], cc["max_len"])
        O = run_ctx(net, g, H, Z, device)
        keep = sel[I["vid"][rows[sel]] >= 0]
        p = Path(cfg["paths"]["out"]) / f"xv_{args.who}_{args.name}.npz"
        np.savez(p, rows=rows[keep], lp=log_softmax(O[keep]))
        log(f"-> {p}")
        return

    # predict on the scoring set
    base, bck = load_member(ck["member"], device)
    H0 = ho.load(cfg, args.track, int(bck.get("pre", 0)))
    z, _, _, h = outputs(base, ho.batches(H0["x"], H0["sid"]), torch.as_tensor(bck["Ev"], device=device),
                         None, bck["cfg"]["norm"], device)
    meta = H0["meta"]
    by = {}   # rows may be shuffled: gather each sentence's words wherever they are
    for i, m in enumerate(meta):
        k = (m["subject"], m["source"], m["epoch"]) if m["source"] == "sentence" else ("w", i)
        by.setdefault(k, []).append(i)
    groups = list(by.values())
    groups = [np.array(sorted(g, key=lambda i: meta[i]["word"])) for g in groups]
    O = run_ctx(net, chunk(groups, cc["max_len"]), h, z, device)
    p = Path(cfg["paths"]["out"]) / f"x_{args.track}_{args.name}.npz"
    np.savez(p, lp=log_softmax(O))
    log(f"-> {p} ({len(groups)} groups)")


if __name__ == "__main__":
    main()
