"""Pick the ensemble's lam (meaning-head weight) and tau (prior correction) on
validation rows, and report every member and the ensemble.

    python -m core.calib -c cfg/a.yaml --who deep  --members d/r/j1d/best.pt d/r/j2d/best.pt
    python -m core.calib -c cfg/a.yaml --who broad --members d/r/j1b/best.pt --split 2   # check on the locked rows
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .common import bacc_at_k, cli, load_json, log, save_json
from .data import Store, Windows
from .score import combine, load_member, member_lp, outputs

LAMS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
TAUS = [-0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25]


def main():
    def extra(ap):
        ap.add_argument("--who", choices=["deep", "broad"], required=True)
        ap.add_argument("--members", nargs="+", required=True)
        ap.add_argument("--extra", nargs="*", default=[], help="npz files with per-row log-scores (rows, lp)")
        ap.add_argument("--split", type=int, default=1, help="1 = val, 2 = locked rows")
        ap.add_argument("--out", default=None)
    args, cfg = cli(__doc__, extra)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    store = Store(cfg["paths"]["cache"])
    rows = store.rows(args.split, args.who)
    rows = rows[store.idx["vid"][rows] >= 0]
    y = store.idx["vid"][rows].astype(np.int64)
    log(f"{len(rows)} labelled rows ({args.who}, split {args.split})")

    outs, prior = [], None
    for p in args.members:
        net, ck = load_member(p, device)
        Ev = torch.as_tensor(ck["Ev"], device=device)
        dl = DataLoader(Windows(store, rows, pre=int(ck.get("pre", 0))), batch_size=512, num_workers=4)
        z, b, _, _ = outputs(net, ((t[0], t[1]) for t in dl), Ev, None, ck["cfg"]["norm"], device)
        outs.append((p, z, b))
        prior = np.asarray(ck[f"prior_{args.who}"])
        one = bacc_at_k(combine([member_lp(z, b, 0.5)], 0.0, prior), y, 10)
        log(f"member {p}: bacc10 {one:.4f} (lam .5, tau 0)")
        del net
    extras = []
    for p in args.extra:
        d = np.load(p)
        pos = {int(r): i for i, r in enumerate(d["rows"])}
        extras.append(np.stack([d["lp"][pos[int(r)]] for r in rows]))
        log(f"extra {p}: bacc10 {bacc_at_k(extras[-1], y, 10):.4f}")

    best = (-1.0, None)
    for lam in LAMS:
        lps = [member_lp(z, b, lam) for _, z, b in outs] + extras
        for tau in TAUS:
            v = bacc_at_k(combine(lps, tau, prior), y, 10)
            if v > best[0]:
                best = (v, (lam, tau))
    lam, tau = best[1]
    s = combine([member_lp(z, b, lam) for _, z, b in outs] + extras, tau, prior)
    res = dict(lam=lam, tau=tau, bacc10=best[0], bacc1=bacc_at_k(s, y, 1), n=len(rows),
               members=args.members, extra=args.extra, split=args.split)
    log(f"ensemble of {len(outs) + len(extras)}: bacc10 {best[0]:.4f} bacc1 {res['bacc1']:.4f} at lam {lam} tau {tau}")
    out = Path(args.out or Path(cfg["paths"]["out"]) / f"calib_{args.who}.json")
    old = load_json(out) if out.exists() else {}
    old[args.who] = res
    save_json(old, out)
    log(f"-> {out}")


if __name__ == "__main__":
    main()
