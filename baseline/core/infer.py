"""Score a track with an ensemble and write the submission CSV.

    python -m core.infer -c cfg/a.yaml --track deep \
        --members d/r/j1d/best.pt d/r/j2d/best.pt --calib d/o/calib_deep.json
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from . import ext, ho
from .common import cli, load_json, log
from .score import combine, load_member, member_lp, outputs, probs


def main():
    def extra(ap):
        ap.add_argument("--track", choices=["deep", "broad"], required=True)
        ap.add_argument("--members", nargs="+", required=True)
        ap.add_argument("--calib", default=None, help="json from core.calib (else cfg.score)")
        ap.add_argument("--extra", nargs="*", default=[], help="npz with lp (N, 50) in row order")
        ap.add_argument("--out", default=None)
    args, cfg = cli(__doc__, extra)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lam, tau = cfg["score"]["lam"], cfg["score"]["tau"]
    if args.calib:
        c = load_json(args.calib)[args.track]
        lam, tau = c["lam"], c["tau"]
    log(f"lam {lam} tau {tau}")

    H = ho.load(cfg, args.track)
    lps, sec, prior = [], [], None
    for p in args.members:
        net, ck = load_member(p, device)
        Hm = ho.load(cfg, args.track, int(ck.get("pre", 0)))
        x, sid = Hm["x"], Hm["sid"]
        Ev = torch.as_tensor(ck["Ev"], device=device)
        Ev2 = torch.as_tensor(ck["Ev2"], device=device)
        z, b, b2, _ = outputs(net, ho.batches(x, sid), Ev, Ev2, ck["cfg"]["norm"], device)
        lps.append(member_lp(z, b, lam))
        sec.append(probs(b2))
        prior = np.asarray(ck[f"prior_{args.track}"])
        log(f"member {p} done")
        del net
    for p in args.extra:
        lps.append(np.load(p)["lp"])
    s = combine(lps, tau, prior)
    p1 = probs(s)
    p2 = np.mean(sec, 0)
    p2 = p2 / p2.sum(1, keepdims=True)

    out = args.out or str(Path(cfg["paths"]["out"]) / f"s_{args.track}_{time.strftime('%m%d_%H%M')}.csv")
    path = ext.write(out, H["indices"], p1, p2)
    top = np.bincount(p1.argmax(1), minlength=p1.shape[1])
    log(f"wrote {path}: {len(p1)} rows; top-1 picks cover {np.count_nonzero(top)} of {p1.shape[1]} words")


if __name__ == "__main__":
    main()
