"""Sentence model trained end to end: the brain encoder and a sentence-level
transformer learn together on every word of a sentence chunk (not only the 50
scored words). Chunks look like the scoring sentences: they start at a word,
span at most ~3 s of onsets, and hold no signal before their first word.

    python -m core.seq train   -c cfg/a.yaml -s data.pre=250 train.init=d/r/p2/best.pt --name s1
    python -m core.seq val     -c cfg/a.yaml --name s1 --who deep      # -> d/o/xv_deep_s1.npz (for calib --extra)
    python -m core.seq predict -c cfg/a.yaml --name s1 --track deep    # -> d/o/x_deep_s1.npz  (for infer --extra)
"""
from __future__ import annotations

import argparse
import copy
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from . import ext, ho
from . import loss as L
from .common import bacc_at_k, load_cfg, log, save_json, seed_all
from .data import GENERIC, Store, augment, prep_batch, sid_of
from .net import Net
from .score import amp, log_softmax
from .train import build_opt, emb_tables, row_fold

SF = 250
SPAN = 3.0          # seconds of word onsets per chunk (scoring sentences span <= ~3.1 s)
NT = 3              # timing features per word


def timing(on: np.ndarray) -> np.ndarray:
    """Per-word features from onsets (s) of one chunk: time since chunk start, gap to previous word, index."""
    on = np.asarray(on, np.float64)
    t = on - on[0]
    gap = np.diff(on, prepend=on[0])
    return np.stack([t / SPAN, np.clip(gap, 0, 2), np.arange(len(on)) / 32.0], 1).astype(np.float32)


# --------------------------------------------------------------------------- data

class Chunks(Dataset):
    """Training/val chunks of consecutive words from one sentence of one run."""

    def __init__(self, store: Store, rows: np.ndarray, pre: int, max_len: int, train: bool,
                 p_single: float = 0.1, jitter: int = 3):
        I = store.idx
        self.s, self.pre, self.L, self.train = store, int(pre), int(max_len), train
        self.p_single, self.jitter = p_single, jitter
        self.win = store.win + self.pre
        rows = np.asarray(rows)
        rows = rows[I["sent"][rows] >= 0]
        key = I["rid"][rows].astype(np.int64) * 10_000_000 + I["sent"][rows].astype(np.int64)
        o = np.lexsort((I["onset"][rows], key))
        rows, key = rows[o], key[o]
        cuts = np.flatnonzero(np.diff(key)) + 1
        self.sents = [g for g in np.split(rows, cuts) if len(g)]
        if not train:   # fixed, non-overlapping chunks covering every word once
            self.items = []
            for g in self.sents:
                on = I["onset"][g]
                a = 0
                while a < len(g):
                    b = a + 1
                    while b < len(g) and b - a < self.L and on[b] - on[a] <= SPAN:
                        b += 1
                    self.items.append(g[a:b])
                    a = b
        self._mm = {}

    def __len__(self):
        return len(self.items) if not self.train else len(self.sents)

    def chunk(self, i, rng):
        if not self.train:
            return self.items[i]
        g = self.sents[i]
        on = self.s.idx["onset"][g]
        a = int(rng.integers(len(g)))
        if rng.random() < self.p_single:
            return g[a:a + 1]
        b = a + 1
        while b < len(g) and b - a < self.L and on[b] - on[a] <= SPAN:
            b += 1
        return g[a:b]

    def window(self, r, first_start, rng):
        I = self.s.idx
        rid = int(I["rid"][r])
        mm = self._mm.get(rid)
        if mm is None:
            mm = self._mm[rid] = np.load(self.s.xdir / f"{rid}.npy", mmap_mode="r")
        st = int(I["start"][r]) - self.pre
        if self.train and self.jitter:
            st += int(rng.integers(-self.jitter, self.jitter + 1))
        T, W = int(self.s.T[rid]), self.win
        out = np.zeros((W, mm.shape[1]), np.float16)
        a, b = max(st, 0, first_start), min(st + W, T)
        if b > a:
            out[a - st:b - st] = mm[a:b]
        return out

    def __getitem__(self, i):
        rng = np.random.default_rng()
        g = self.chunk(i, rng)
        I = self.s.idx
        first = int(I["start"][g[0]])
        x = np.stack([self.window(int(r), first, rng) for r in g])
        return (x, sid_of(I["subj"][g[0]]), timing(I["onset"][g]), I["vid"][g].astype(np.int64),
                I["wid"][g].astype(np.int64), g.astype(np.int64))


def collate(batch):
    Lm = max(b[0].shape[0] for b in batch)
    B = len(batch)
    W, C = batch[0][0].shape[1:]
    x = np.zeros((B, Lm, W, C), np.float16)
    tf = np.zeros((B, Lm, NT), np.float32)
    vid = np.full((B, Lm), -1, np.int64)
    wid = np.full((B, Lm), -1, np.int64)
    rows = np.full((B, Lm), -1, np.int64)
    pad = np.ones((B, Lm), bool)
    sid = np.zeros(B, np.int64)
    for i, (xi, s, t, v, w, r) in enumerate(batch):
        n = len(xi)
        x[i, :n], tf[i, :n], vid[i, :n], wid[i, :n], rows[i, :n], pad[i, :n], sid[i] = xi, t, v, w, r, False, s
    return tuple(map(torch.as_tensor, (x, sid, tf, vid, wid, rows, pad)))


# --------------------------------------------------------------------------- model

class SeqNet(nn.Module):
    def __init__(self, enc: Net, n_vocab: int, d_emb: int, d: int = 512, depth: int = 4, heads: int = 8,
                 drop: float = 0.1, max_len: int = 64):
        super().__init__()
        self.enc = enc
        dout = enc.pool.proj[0].out_features
        self.inp = nn.Sequential(nn.Linear(dout + NT, d), nn.LayerNorm(d))
        self.pos = nn.Parameter(torch.zeros(1, max_len, d))
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, drop, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, depth)
        self.head_word = nn.Linear(d, n_vocab)
        self.head_mean = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d_emb))
        nn.init.zeros_(self.head_word.weight)
        nn.init.zeros_(self.head_word.bias)
        self.t = nn.Parameter(torch.tensor(math.log(10.0)))
        self.b = nn.Parameter(torch.tensor(-10.0))

    def forward(self, x, sid, tf, pad):
        """x (B, L, C, T) normalised, sid (B,), tf (B, L, NT), pad (B, L) True = padding."""
        B, Lm = pad.shape
        keep = ~pad
        xs = x[keep]
        ss = sid[:, None].expand(B, Lm)[keep]
        o = self.enc(xs, ss)
        h = torch.zeros(B, Lm, o["h"].shape[-1], device=x.device, dtype=o["h"].dtype)
        h[keep] = o["h"]
        z0 = torch.zeros(B, Lm, o["z"].shape[-1], device=x.device, dtype=o["z"].dtype)
        z0[keep] = o["z"]
        u = self.inp(torch.cat([h, tf.to(h.dtype)], -1)) + self.pos[:, :Lm]
        u = self.tf(u, src_key_padding_mask=pad)
        z = z0 + self.head_word(u)
        m = F.normalize(self.head_mean(u), dim=-1)
        return z, m, z0

    def mean_logits(self, m, E):
        return self.t.exp() * m @ E.T + self.b


def build(ck_enc: dict, n_vocab, d_emb, sc: dict) -> SeqNet:
    enc = Net(ck_enc["cfg"]["model"], n_vocab, d_emb, ck_enc["win"], ck_enc["frames"])
    enc.load_state_dict(ck_enc["state"])
    return SeqNet(enc, n_vocab, d_emb, sc["d"], sc["depth"], sc["heads"], sc["drop"], sc["max_len"])


def load_seq(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    enc = Net(ck["enc_cfg"]["model"], ck["n_vocab"], ck["d_emb"], ck["win"], ck["frames"])
    sc = ck["sc"]
    net = SeqNet(enc, ck["n_vocab"], ck["d_emb"], sc["d"], sc["depth"], sc["heads"], sc["drop"], sc["max_len"])
    net.load_state_dict(ck["state"])
    return net.to(device).eval(), ck


@torch.no_grad()
def run(net, batches, norm, device, Ev):
    """Yields (rows, scores (n, V) log-space incl. meaning, z (n,V), b (n,V)) per batch."""
    for x, sid, tf, vid, wid, rows, pad in batches:
        x = x.to(device, non_blocking=True)
        B, Lm, W, C = x.shape
        xn = prep_batch(x.view(B * Lm, W, C), norm["mode"], norm["clamp"]).view(B, Lm, C, W)
        with amp(device):
            z, m, _ = net(xn, sid.to(device), tf.to(device), pad.to(device))
        keep = ~pad
        b = net.mean_logits(m.float(), Ev)
        yield rows[keep].numpy(), z.float()[keep.to(device)].cpu().numpy(), b[keep.to(device)].cpu().numpy()


def evaluate(net, ds, cfg, device, Ev, prior, who_rows):
    dl = DataLoader(ds, batch_size=16, num_workers=min(8, cfg["train"]["workers"]), collate_fn=collate)
    R, Z, Bm = [], [], []
    for r, z, b in run(net, dl, cfg["norm"], device, Ev):
        R.append(r), Z.append(z), Bm.append(b)
    R, Z, Bm = np.concatenate(R), np.concatenate(Z), np.concatenate(Bm)
    I = ds.s.idx
    out = {}
    for k, rows in who_rows.items():
        m = np.isin(R, rows) & (I["vid"][R] >= 0)
        if not m.any():
            continue
        y = I["vid"][R[m]]
        best = (-1, None)
        for lam in (0.0, 0.5, 1.0):
            s = log_softmax(Z[m]) + lam * log_softmax(Bm[m]) - 0 * np.log(prior[k])[None]
            v = bacc_at_k(s, y, 10)
            if v > best[0]:
                best = (v, lam)
        out[k] = dict(bacc10=bacc_at_k(Z[m], y, 10), best=best[0], lam=best[1], n=int(m.sum()))
    return out, (R, Z, Bm)


# --------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["train", "val", "predict"])
    ap.add_argument("-c", "--cfg", default="cfg/a.yaml")
    ap.add_argument("-s", "--set", nargs="*", default=[])
    ap.add_argument("--name", required=True)
    ap.add_argument("--who", choices=["deep", "broad"])
    ap.add_argument("--track", choices=["deep", "broad"])
    ap.add_argument("--lam", type=float, default=None, help="meaning weight in the saved scores (default: tuned)")
    args = ap.parse_args()
    cfg = load_cfg(args.cfg, args.set)
    sc = dict(d=512, depth=4, heads=8, drop=0.1, max_len=64, batch=16, epochs=8, steps=1500,
              lr_enc=1e-4, lr=3e-4, chunk=32, p_single=0.1, w_mean=0.5, w_base=0.3)
    sc.update(cfg.get("seq") or {})
    tc = cfg["train"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = Path(cfg["paths"]["runs"]) / args.name
    d.mkdir(parents=True, exist_ok=True)
    store = Store(cfg["paths"]["cache"])
    E, ev, ev2 = emb_tables(cfg, store)
    Ev = torch.as_tensor(ev, device=device)
    pri = {k: store.prior(k) for k in ("deep", "broad")}
    seed_all(tc["seed"])

    if args.cmd == "train":
        ck_enc = torch.load(tc["init"], map_location="cpu", weights_only=False)
        pre = int(ck_enc.get("pre", 0))
        assert pre == int(cfg["data"].get("pre", 0)), "data.pre must match the encoder checkpoint"
        net = build(ck_enc, len(store.meta["vocab"]), E.shape[1], sc).to(device)
        E_t = torch.as_tensor(E, device=device)
        tr = store.rows(0, tc.get("who", "all"))
        if tc.get("fold"):
            k, K = map(int, str(tc["fold"]).split("/"))
            tr = tr[row_fold(store, K)[tr] != k]
        dtr = Chunks(store, tr, pre, sc["chunk"], True, sc["p_single"], int(tc["jitter"]))
        val_rows = {"deep": store.rows(1, "deep"), "broad": store.rows(1, "broad")}
        dva = Chunks(store, np.concatenate(list(val_rows.values())), pre, sc["chunk"], False)
        I = store.idx
        subj0 = np.array([I["subj"][g[0]] == 0 for g in dtr.sents])
        w = np.where(subj0, tc["deep_share"] / max(subj0.sum(), 1), (1 - tc["deep_share"]) / max((~subj0).sum(), 1))
        log(f"{len(dtr.sents)} train sentences, {len(dva.items)} val chunks, pre {pre}")
        groups = {"decay": [], "plain": [], "enc": []}
        for n, p in net.named_parameters():
            if n.startswith("enc."):
                groups["enc"].append(p)
            elif p.ndim < 2 or n in ("t", "b", "pos"):
                groups["plain"].append(p)
            else:
                groups["decay"].append(p)
        opt = torch.optim.AdamW([dict(params=groups["enc"], lr=sc["lr_enc"], weight_decay=tc["wd"]),
                                 dict(params=groups["decay"], lr=sc["lr"], weight_decay=tc["wd"]),
                                 dict(params=groups["plain"], lr=sc["lr"], weight_decay=0.0)], betas=(0.9, 0.98))
        for g in opt.param_groups:
            g["base_lr"] = g["lr"]
        ema = copy.deepcopy(net).eval()
        for p in ema.parameters():
            p.requires_grad_(False)
        lp_train = torch.as_tensor(np.log(pri["deep"] * 0.5 + pri["broad"] * 0.5), device=device, dtype=torch.float32)
        total, warm, step, best = sc["epochs"] * sc["steps"], 300, 0, -1.0
        best_k = {}
        for ep in range(sc["epochs"]):
            smp = torch.utils.data.WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), sc["steps"] * sc["batch"],
                                                         replacement=True, generator=torch.Generator().manual_seed(ep))
            dl = DataLoader(dtr, batch_size=sc["batch"], sampler=smp, num_workers=tc["workers"], collate_fn=collate,
                            drop_last=True, pin_memory=True, prefetch_factor=4 if tc["workers"] else None)
            net.train()
            t0, run_l, n_l = time.time(), np.zeros(3), 0
            for x, sid, tf, vid, wid, rows, pad in dl:
                f = min(1.0, (step + 1) / warm) * (0.5 * (1 + math.cos(math.pi * min(step / total, 1.0))) * 0.97 + 0.03)
                for g in opt.param_groups:
                    g["lr"] = g["base_lr"] * f
                x = x.to(device, non_blocking=True)
                B, Lm, W, C = x.shape
                pad = pad.to(device)
                xn = prep_batch(x.view(B * Lm, W, C), cfg["norm"]["mode"], cfg["norm"]["clamp"])
                xn = augment(xn, cfg["aug"]).view(B, Lm, C, W)
                sid = sid.to(device)
                sid = torch.where(torch.rand_like(sid, dtype=torch.float) < tc["subj_drop"], torch.full_like(sid, GENERIC), sid)
                tf, vid, wid = tf.to(device), vid.to(device), wid.to(device)
                with amp(device):
                    z, m, z0 = net(xn, sid, tf, pad)
                keep = ~pad
                lw = L.word(z.float()[keep], vid[keep], lp_train, cfg["loss"]["tau_train"], cfg["loss"]["smooth"])
                lb = L.word(z0.float()[keep], vid[keep], lp_train, cfg["loss"]["tau_train"], cfg["loss"]["smooth"])
                ok = keep & (wid >= 0)
                lm = L.meaning(net, m.float()[ok], E_t, wid[ok]) if sc["w_mean"] else lw * 0
                tot = lw + sc["w_base"] * lb + sc["w_mean"] * lm
                opt.zero_grad(set_to_none=True)
                tot.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), tc["clip"])
                opt.step()
                with torch.no_grad():
                    dec = tc["ema"] if step > warm else 0.0
                    for pe, p in zip(ema.parameters(), net.parameters()):
                        pe.lerp_(p.detach(), 1 - dec)
                    for be, b_ in zip(ema.buffers(), net.buffers()):
                        be.copy_(b_)
                run_l += np.array([lw.item(), lb.item(), lm.item()])
                n_l += 1
                step += 1
                if step % 100 == 0:
                    r = run_l / n_l
                    log(f"ep {ep} step {step} word {r[0]:.3f} base {r[1]:.3f} mean {r[2]:.3f} lr x{f:.3f} "
                        f"{(time.time() - t0) / n_l:.2f}s/step")
                    run_l[:], n_l, t0 = 0, 0, time.time()
            res, _ = evaluate(ema, dva, cfg, device, Ev, pri, val_rows)
            log("== epoch " + str(ep) + " " + " | ".join(f"{k}: bacc10 {r['bacc10']:.4f} best {r['best']:.4f}@lam{r['lam']}"
                                                         for k, r in res.items()))
            score = np.mean([r["best"] for r in res.values()])
            ck = dict(state=ema.state_dict(), enc_cfg=ck_enc["cfg"], cfg=cfg, sc=sc, n_vocab=len(store.meta["vocab"]),
                      d_emb=E.shape[1], win=ck_enc["win"], frames=ck_enc["frames"], pre=pre, Ev=ev,
                      prior_deep=pri["deep"], prior_broad=pri["broad"], epoch=ep, res=res)
            torch.save(ck, d / "last.pt")
            if score > best:
                best = score
                torch.save(ck, d / "best.pt")
                log(f"new best {score:.4f}")
            for k, r in res.items():
                if r["best"] > best_k.get(k, -1.0):
                    best_k[k] = r["best"]
                    torch.save(ck, d / f"best_{k}.pt")
                    log(f"new best {k} {r['best']:.4f}")
            save_json(dict(epoch=ep, res=res), d / f"ep{ep}.json")
        log(f"done. best {best:.4f}")
        return

    k_ = args.who or args.track
    p_ = d / f"best_{k_}.pt"
    p_ = p_ if p_.exists() else d / "best.pt"
    log(f"loading {p_}")
    net, ck = load_seq(p_, device)
    pre = int(ck["pre"])

    if args.cmd == "val":
        rows = store.rows(1, args.who)
        ds = Chunks(store, rows, pre, ck["sc"]["chunk"], False)
        res, (R, Z, Bm) = evaluate(net, ds, cfg, device, Ev, pri, {args.who: rows})
        lam = res[args.who]["lam"] if args.lam is None else args.lam
        I = store.idx
        m = I["vid"][R] >= 0
        lp = log_softmax(log_softmax(Z[m]) + lam * log_softmax(Bm[m]))
        p = Path(cfg["paths"]["out"]) / f"xv_{args.who}_{args.name}.npz"
        np.savez(p, rows=R[m], lp=lp, lam=lam)
        log(f"{args.who} val {res[args.who]} -> {p}")
        return

    # predict on the scoring set: every word of each scoring sentence goes in, scored words come out
    track = args.track
    lam = ck["res"].get(track, {}).get("lam", 0.5) if args.lam is None else args.lam
    base = ho.load(cfg, track, 0)
    meta = base["meta"]
    S = ext.Scoring(track, cfg["paths"]["ho"], download=True)
    stats = {s: ho._subject_stats(S, s) for s in S.subjects}
    W = SF + pre
    out = np.zeros((len(meta), ck["n_vocab"]), np.float32)
    by = {}
    for i, mm in enumerate(meta):
        k = (mm["subject"], "s", mm["epoch"]) if mm["source"] == "sentence" else (mm["subject"], "w", mm["epoch"])
        by.setdefault(k, []).append(i)
    items = list(by.items())
    cache = {}

    def make(key, idx):
        s, kind, ep = key
        src = "sentence" if kind == "s" else "word"
        if (s, src) not in cache:
            cache.clear()
            dd = np.load(S.file(s, src))
            cache[(s, src)] = {k: dd[k] for k in dd.files}
        dd = cache[(s, src)]
        mu, sd = stats[s]
        if kind == "w":
            x = np.zeros((1, W, 306), np.float32)
            x[0, pre:] = ((dd["meg"][ep] - mu[:, None]) / sd[:, None]).T
            return np.clip(x, -30, 30).astype(np.float16), np.zeros(1), {idx[0]: 0}
        seg = dd["meg"][ep][:, : int(dd["sentence_n_times"][ep])]
        seg = np.clip((seg - mu[:, None]) / sd[:, None], -30, 30)
        ons = dd["word_onsets_s"][ep]
        valid = np.flatnonzero(np.isfinite(ons))
        on = ons[valid].astype(np.float64)
        order = np.argsort(on, kind="stable")
        valid, on = valid[order], on[order]
        col2pos = {int(c): j for j, c in enumerate(valid)}
        x = np.zeros((len(valid), W, 306), np.float32)
        for j, o in enumerate(on):
            a = int(round(o * SF)) - pre
            lo, hi = max(a, 0), min(a + W, seg.shape[1])
            if hi > lo:
                x[j, lo - a:hi - a] = seg[:, lo:hi].T
        return x.astype(np.float16), on, {i: col2pos[int(meta[i]["word"])] for i in idx}

    bs = 16
    for a in range(0, len(items), bs):
        chunk = items[a:a + bs]
        built = [make(k, idx) for k, idx in chunk]
        batch = []
        for (key, idx), (x, on, pos) in zip(chunk, built):
            Lmax = ck["sc"]["max_len"]
            if len(x) > Lmax:   # keep the window of words around the scored ones
                x, on = x[:Lmax], on[:Lmax]
            batch.append((x, sid_of(key[0]), timing(on) if len(on) else np.zeros((1, NT), np.float32),
                          np.zeros(len(x), np.int64), np.zeros(len(x), np.int64), np.arange(len(x))))
        x, sid, tf, _, _, _, pad = collate(batch)
        for rr, z, b in run(net, [(x, sid, tf, None, None, torch.full(pad.shape, -1), pad)], cfg["norm"], device, Ev):
            pass
        # map back: run() returns kept positions in row-major order of the batch
        lp = log_softmax(log_softmax(z) + lam * log_softmax(b))
        keep = (~pad).numpy()
        flat_index = -np.ones(keep.shape, np.int64)
        flat_index[keep] = np.arange(keep.sum())
        for bi, ((key, idx), (_, _, pos)) in enumerate(zip(chunk, built)):
            for i in idx:
                j = pos[i]
                if j < keep.shape[1] and flat_index[bi, j] >= 0:
                    out[i] = lp[flat_index[bi, j]]
                else:
                    out[i] = np.log(1.0 / out.shape[1])
    p = Path(cfg["paths"]["out"]) / f"x_{track}_{args.name}.npz"
    np.savez(p, lp=out, lam=lam)
    log(f"-> {p} ({len(items)} sentences/words, lam {lam})")


if __name__ == "__main__":
    main()
