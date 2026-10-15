"""Train (or fine-tune) one network.

    python -m core.train -c cfg/a.yaml -s train.name=j1
    python -m core.train -c cfg/ft_deep.yaml -s train.init=d/r/j1/best.pt train.name=j1d
"""
from __future__ import annotations

import copy
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import loss as L
from .common import bacc_at_k, cli, log, norm_word, save_json, seed_all
from .data import GENERIC, Store, Windows, augment, prep_batch, sampler
from .net import Net
from .score import amp, combine, member_lp, outputs

SPATIAL_ADAPTER = ("spatial.U", "spatial.V", "spatial.bias")


def emb_tables(cfg, store):
    from .common import load_json
    E = np.load(Path(cfg["paths"]["emb"]) / "E.npy").astype(np.float32)
    words = load_json(Path(cfg["paths"]["emb"]) / "words.json")["words"]
    pos = {w: i for i, w in enumerate(words)}
    from . import ext
    ev = np.stack([E[pos[norm_word(w)]] for w in ext.vocab()])
    ev2 = np.stack([E[pos[norm_word(w)]] for w in ext.vocab2()])
    assert words[: len(store.meta["words"])] == store.meta["words"], "embeddings out of date: rerun core.emb"
    return E, ev, ev2


def run_fold(store, K: int) -> np.ndarray:
    """Fold of each run, by stimulus (task + session), so all subjects who heard
    the same audio land in the same fold."""
    import zlib
    return np.array([zlib.crc32(f"{r['task']}|{r['session']}".encode()) % K for r in store.meta["runs"]])


def row_fold(store, K: int) -> np.ndarray:
    """Fold of every word row. Runs whose audio only one subject heard keep their
    run's fold; audio heard by several subjects (where all the broad subjects' data
    is) is split by blocks of 10 sentences, the same way for every subject, so each
    fold sees every subject but never the same sentences."""
    import zlib
    I, runs = store.idx, store.meta["runs"]
    fold = run_fold(store, K)[I["rid"]]
    heard = {}
    for r in runs:
        heard.setdefault((r["task"], str(r["session"])), set()).add(int(r["subject"]))
    shared = np.array([len(heard[(r["task"], str(r["session"]))]) > 1 for r in runs])
    m = shared[I["rid"]] & (I["sent"] >= 0)
    if m.any():
        keys = [f"{runs[int(q)]['task']}|{runs[int(q)]['session']}|{int(s) // 10}"
                for q, s in zip(I["rid"][m], I["sent"][m])]
        fold[m] = np.array([zlib.crc32(k.encode()) % K for k in keys])
    return fold


def build_opt(net, tc):
    groups = {"decay": [], "plain": [], "adapter": []}
    for n, p in net.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith(SPATIAL_ADAPTER):
            groups["adapter"].append(p)
        elif p.ndim < 2 or n in ("t", "b"):
            groups["plain"].append(p)
        else:
            groups["decay"].append(p)
    pg = [dict(params=groups["decay"], weight_decay=tc["wd"], lr=tc["lr"]),
          dict(params=groups["plain"], weight_decay=0.0, lr=tc["lr"]),
          dict(params=groups["adapter"], weight_decay=0.0, lr=tc.get("lr_adapter", tc["lr"]))]
    pg = [g for g in pg if g["params"]]
    for g in pg:
        g["base_lr"] = g["lr"]
    return torch.optim.AdamW(pg, betas=(0.9, 0.98), fused=torch.cuda.is_available())


def evaluate(net, store, rows, Ev, cfg, device, prior):
    """BAcc@10 / @1 on in-vocabulary rows, at the configured lam/tau and over a small grid."""
    rows = rows[store.idx["vid"][rows] >= 0]
    if len(rows) == 0:
        return {}
    ds = Windows(store, rows, pre=int(cfg["data"].get("pre", 0)))
    dl = DataLoader(ds, batch_size=512, num_workers=min(4, cfg["train"]["workers"]), shuffle=False)
    batches = ((b[0], b[1]) for b in dl)
    z, b, _, _ = outputs(net, batches, Ev, None, cfg["norm"], device)
    y = store.idx["vid"][rows].astype(np.int64)
    sc = cfg["score"]
    s = combine([member_lp(z, b, sc["lam"])], sc["tau"], prior)
    res = dict(n=len(rows), bacc10=bacc_at_k(s, y, 10), bacc1=bacc_at_k(s, y, 1))
    best = (-1, None)
    for lam in (0.0, 0.5, 1.0):
        for tau in (0.0, 0.5, 1.0):
            v = bacc_at_k(combine([member_lp(z, b, lam)], tau, prior), y, 10)
            if v > best[0]:
                best = (v, (lam, tau))
    res["grid_best"], res["grid_at"] = best
    return res


def ema_update(ema, net, decay):
    with torch.no_grad():
        for pe, p in zip(ema.parameters(), net.parameters()):
            pe.lerp_(p.detach(), 1 - decay)
        for be, b in zip(ema.buffers(), net.buffers()):
            be.copy_(b)


def main():
    args, cfg = cli(__doc__)
    tc, lc = cfg["train"], cfg["loss"]
    seed_all(tc["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(cfg["paths"]["runs"]) / tc["name"]
    out.mkdir(parents=True, exist_ok=True)
    save_json(cfg, out / "cfg.json")

    store = Store(cfg["paths"]["cache"])
    E, ev, ev2 = emb_tables(cfg, store)
    E_t = torch.as_tensor(E, device=device)
    Ev = torch.as_tensor(ev, device=device)
    n_vocab = len(store.meta["vocab"])
    win, nfr = int(store.meta["win"]), int(store.meta["frames"])
    pre = int(cfg["data"].get("pre", 0))
    hop = win // nfr
    assert pre % hop == 0, f"data.pre must be a multiple of {hop}"
    f0 = pre // hop                     # first frame of the labelled 1 s after onset
    win, frames = win + pre, (win + pre) // hop

    net = Net(cfg["model"], n_vocab, E.shape[1], win, frames).to(device)
    if tc.get("init"):
        ck = torch.load(tc["init"], map_location="cpu", weights_only=False)
        missing, unexpected = net.load_state_dict(ck["state"], strict=False)
        log(f"init from {tc['init']} (missing {len(missing)}, unexpected {len(unexpected)})")
    only, frz = tuple(tc.get("train_only") or ()), tuple(tc.get("freeze") or ())
    for n, p in net.named_parameters():
        if (only and not n.startswith(only)) or (frz and n.startswith(frz)):
            p.requires_grad_(False)
    n_train = sum(p.numel() for p in net.parameters() if p.requires_grad)
    log(f"params {sum(p.numel() for p in net.parameters()) / 1e6:.1f}M, trainable {n_train / 1e6:.1f}M, device {device}")
    ema = copy.deepcopy(net).eval()
    for p in ema.parameters():
        p.requires_grad_(False)

    tr_rows = store.rows(0, tc.get("who", "all"))
    if tc.get("fold"):  # "k/K": leave out stimulus fold k (for out-of-fold context features)
        k, K = map(int, str(tc["fold"]).split("/"))
        tr_rows = tr_rows[row_fold(store, K)[tr_rows] != k]
    val = {"deep": store.rows(1, "deep"), "broad": store.rows(1, "broad")}
    pri = {k: store.prior(k) for k in ("deep", "broad")}
    lp_train = torch.as_tensor(np.log(store.prior("deep") * 0.5 + store.prior("broad") * 0.5), device=device, dtype=torch.float32)
    log(f"train rows {len(tr_rows)}, val deep {len(val['deep'])}, val broad {len(val['broad'])}")

    opt = build_opt(net, tc)
    steps_ep, epochs = int(tc["steps_per_epoch"]), int(tc["epochs"])
    total, warm = steps_ep * epochs, int(tc["warmup"])
    step, best_score = 0, -1.0
    best_k = {}
    hist = []
    for ep in range(epochs):
        avg_p = float(tc.get("avg_p", 0)) * max(0.0, 1 - ep / max(1, epochs // 2))
        ds = Windows(store, tr_rows, jitter=int(tc["jitter"]), avg_p=avg_p, pre=pre, pre_mask=float(tc.get("pre_mask", 0)))
        smp = sampler(store, tr_rows, steps_ep * tc["batch"], tc["deep_share"], tc["vocab_boost"], tc["seed"] * 1000 + ep)
        dl = DataLoader(ds, batch_size=tc["batch"], sampler=smp, num_workers=tc["workers"], drop_last=True,
                        pin_memory=device.type == "cuda", prefetch_factor=4 if tc["workers"] else None)
        net.train()
        t0, run, nrun = time.time(), np.zeros(5), 0
        for x, sid, vid, wid, nxt, nb, ph in dl:
            lr_f = min(1.0, (step + 1) / max(warm, 1)) * (0.5 * (1 + math.cos(math.pi * min(step / total, 1.0))) * 0.97 + 0.03)
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * lr_f
            x = prep_batch(x.to(device, non_blocking=True), cfg["norm"]["mode"], cfg["norm"]["clamp"])
            x = augment(x, cfg["aug"])
            sid = sid.to(device)
            sid = torch.where(torch.rand_like(sid, dtype=torch.float) < tc["subj_drop"], torch.full_like(sid, GENERIC), sid)
            vid, wid, nxt, nb, ph = (t.to(device) for t in (vid, wid, nxt, nb, ph))
            with amp(device):
                o = net(x, sid)
            lw = L.word(o["z"].float(), vid, lp_train, lc["tau_train"], lc["smooth"])
            lm = L.meaning(net, o["m"].float(), E_t, wid) if lc["w_mean"] else lw * 0
            ls = L.sound(o["ph"][:, f0:f0 + nfr].float(), o["ft"][:, f0:f0 + nfr].float(), ph) if lc["w_sound"] else lw * 0
            ln = L.neighbours(o["nb"].float(), o["nx"].float(), nb, nxt) if lc["w_nb"] else lw * 0
            tot = lc["w_word"] * lw + lc["w_mean"] * lm + lc["w_sound"] * ls + lc["w_nb"] * ln
            opt.zero_grad(set_to_none=True)
            tot.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), tc["clip"])
            opt.step()
            ema_update(ema, net, tc["ema"] if step > warm else 0.0)
            run += np.array([tot.item(), lw.item(), lm.item(), ls.item(), ln.item()])
            nrun += 1
            step += 1
            if step % 100 == 0:
                r = run / nrun
                log(f"ep {ep} step {step} loss {r[0]:.3f} word {r[1]:.3f} mean {r[2]:.3f} sound {r[3]:.3f} "
                    f"nb {r[4]:.3f} lr x{lr_f:.3f} {(time.time() - t0) / nrun:.2f}s/step")
                run[:], nrun, t0 = 0, 0, time.time()

        res = {k: evaluate(ema, store, v, Ev, cfg, device, pri[k]) for k, v in val.items()}
        sel = tc["select"]
        vals = [res[k].get("bacc10", np.nan) for k in (["deep", "broad"] if sel == "mean" else [sel])]
        score = float(np.nanmean(vals)) if not all(np.isnan(vals)) else -1.0
        hist.append(dict(epoch=ep, step=step, score=score, **{f"{k}_{m}": v for k, r in res.items() for m, v in r.items()}))
        save_json(hist, out / "hist.json")
        msg = " | ".join(f"{k}: bacc10 {r['bacc10']:.4f} bacc1 {r['bacc1']:.4f} grid {r['grid_best']:.4f}@{r['grid_at']}"
                         for k, r in res.items() if r)
        log(f"== epoch {ep} {msg}")
        ck = dict(state=ema.state_dict(), cfg=cfg, n_vocab=n_vocab, d_emb=E.shape[1], win=win, frames=frames, pre=pre,
                  vocab=store.meta["vocab"], Ev=ev, Ev2=ev2, prior_deep=pri["deep"], prior_broad=pri["broad"],
                  epoch=ep, score=score)
        torch.save(ck, out / "last.pt")
        if score > best_score:
            best_score = score
            torch.save(ck, out / "best.pt")
            log(f"new best {score:.4f} -> {out / 'best.pt'}")
        for k, r in res.items():
            v = r.get("bacc10", -1.0) if r else -1.0
            if v == v and v > best_k.get(k, -1.0):
                best_k[k] = v
                torch.save(ck, out / f"best_{k}.pt")
    log(f"done. best {sel} bacc10 {best_score:.4f}")


if __name__ == "__main__":
    main()
