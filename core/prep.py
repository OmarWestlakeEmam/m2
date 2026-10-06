"""Build the training cache from downloaded runs.

For every run: per-channel session z-score of the continuous recording, stored
as float16 (T, 306) so a 1 s window is one contiguous read. For every word: its
window start, labels for all four heads, sentence id, and split.

    python -m core.prep -c cfg/a.yaml
"""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from . import ext
from .common import cli, log, norm_word, save_json
from .fetch import select
from .phon import N_PH, SIL, phone_id

CORPORA = ["sherlock", "timit", "mocha", "podcasts"]


def _events(path) -> tuple[pd.DataFrame, pd.DataFrame]:
    ev = pd.read_csv(path, sep="\t")
    ev["timemeg"] = pd.to_numeric(ev["timemeg"], errors="coerce")
    w = ev[ev["kind"] == "word"].dropna(subset=["segment", "timemeg"]).copy()
    w["w"] = w["segment"].map(norm_word)
    w = w[w["w"].str.len() > 0].sort_values("timemeg", kind="stable").reset_index(drop=True)
    p = ev[ev["kind"] == "phoneme"].dropna(subset=["segment", "timemeg"]).copy()
    p = p.sort_values("timemeg", kind="stable").reset_index(drop=True)
    return w, p


def _sent_col(w: pd.DataFrame, name: str) -> np.ndarray:
    if name in w:
        return pd.to_numeric(w[name], errors="coerce").fillna(-1).astype(np.int64).to_numpy()
    return np.full(len(w), -1, dtype=np.int64)


def run_one(job: dict) -> dict:
    rec, rid, dc, vocab, cut, xdir = job["rec"], job["rid"], job["dc"], job["vocab"], job["cut"], job["xdir"]
    h5p, evp = ext.local_paths(rec, job["raw"])
    sf_out, win, nfr = float(dc["sfreq"]), int(dc["win"]), int(dc["frames"])
    win_s = win / sf_out
    vmap = {norm_word(v): i for i, v in enumerate(vocab)}

    # ---- MEG: session z-score, float16, time-major
    with h5py.File(h5p, "r") as f:
        x = f["data"][:].astype(np.float32)               # (306, T)
        sf = float(f.attrs["sample_frequency"])
    if abs(sf - sf_out) > 1e-6:
        from fractions import Fraction
        from scipy.signal import resample_poly
        fr = Fraction(sf_out / sf).limit_denominator(100)
        x = resample_poly(x, fr.numerator, fr.denominator, axis=1).astype(np.float32)
    sub = x[:, ::7].astype(np.float64)
    mu = sub.mean(1, keepdims=True)
    sd = sub.std(1, keepdims=True)
    sd[sd == 0] = 1.0
    x = np.clip((x - mu) / sd, -30, 30).astype(np.float16).T  # (T, 306)
    T = x.shape[0]
    np.save(xdir / f"{rid}.npy", np.ascontiguousarray(x))
    del x

    # ---- words
    w, p = _events(evp)
    onset = w["timemeg"].to_numpy(np.float64)
    start = np.round(onset * sf_out).astype(np.int64)
    ok = (start >= 0) & (start + win <= T)
    words = w["w"].to_numpy()
    sent = _sent_col(w, "sentenceidx")
    pos = _sent_col(w, "wordidx")
    n = len(w)

    # neighbours (vocab words starting later inside the window) and next word
    vid = np.array([vmap.get(s, -1) for s in words], dtype=np.int64)
    nb = np.zeros((n, len(vocab)), dtype=np.uint8)
    nxt = np.full(n, "", dtype=object)
    j = 0
    for i in range(n):
        j = max(j, i + 1)
        while j < n and onset[j] < onset[i] + win_s:
            j += 1
        for k in range(i + 1, j):
            if vid[k] >= 0:
                nb[i, vid[k]] = 1
        if i + 1 < n and onset[i + 1] < onset[i] + win_s:
            nxt[i] = words[i + 1]

    # phoneme label per frame
    ph = np.full((n, nfr), SIL, dtype=np.int8)
    if len(p):
        p_on = p["timemeg"].to_numpy(np.float64)
        dur = pd.to_numeric(p.get("duration", pd.Series(np.zeros(len(p)))), errors="coerce").fillna(0).to_numpy()
        pid = np.array([phone_id(s) for s in p["segment"]], dtype=np.int64)
        centers = onset[:, None] + (np.arange(nfr)[None, :] + 0.5) * (win_s / nfr)
        idx = np.searchsorted(p_on, centers, side="right") - 1
        valid = idx >= 0
        idc = np.clip(idx, 0, None)
        inside = valid & (centers < (p_on[idc] + dur[idc]))
        lab = np.where(inside, pid[idc], SIL)
        ph = np.where(lab < 0, -100, lab).astype(np.int8)  # -100 stays ignored

    # split: 0 train, 1 val, 2 lock
    key = (rec["task"], str(rec["session"]))
    split = np.zeros(n, dtype=np.int8)
    if key in cut:
        split[sent >= cut[key]] = 1
    elif key in job["val"]:
        split[:] = 1
    elif key in job["lock"]:
        split[:] = 2

    m = ok
    return dict(rid=rid, T=T, start=start[m], onset=onset[m], words=list(words[m]), nxt=list(nxt[m]),
                sent=sent[m], pos=pos[m], nb=nb[m], ph=ph[m], split=split[m], vid=vid[m])


def main():
    args, cfg = cli(__doc__, lambda ap: ap.add_argument("-j", type=int, default=4))
    dc, P = cfg["data"], cfg["paths"]
    cdir = Path(P["cache"])
    xdir = cdir / "x"
    xdir.mkdir(parents=True, exist_ok=True)
    vocab = ext.vocab()

    recs = []
    for r in select(ext.records(), dc):
        h5p, evp = ext.local_paths(r, P["raw"])
        if h5p.exists() and evp.exists():
            recs.append(r)
    if not recs:
        raise SystemExit(f"no runs found under {P['raw']}; run core.fetch first")
    log(f"{len(recs)} runs on disk")

    shared = {(t, str(s)) for t, s in dc.get("shared", [])}
    val = {(t, str(s)) for t, s in dc.get("val", [])}
    lock = {(t, str(s)) for t, s in dc.get("lock", [])}
    # Shared-stimulus sessions: cut on sentence index so every subject's val
    # part covers the same, never-trained-on stretch of the story.
    maxs: dict = {}
    for r in recs:
        key = (r["task"], str(r["session"]))
        if key in shared:
            w, _ = _events(ext.local_paths(r, P["raw"])[1])
            s = _sent_col(w, "sentenceidx")
            maxs[key] = max(maxs.get(key, 0), int(s.max()) if len(s) else 0)
    cut = {k: int(round((1 - float(dc["tail_frac"])) * v)) for k, v in maxs.items()}

    jobs = [dict(rec=r, rid=i, dc=dc, vocab=vocab, cut=cut, val=val, lock=lock, xdir=xdir, raw=P["raw"])
            for i, r in enumerate(recs)]
    outs = []
    with ProcessPoolExecutor(args.j) as ex:
        for i, o in enumerate(ex.map(run_one, jobs), 1):
            outs.append(o)
            if i % 5 == 0 or i == len(jobs):
                log(f"prep {i}/{len(jobs)}")

    # ---- global index
    cnt = Counter(w for o in outs for w in o["words"])
    words = [w for w, _ in cnt.most_common()]
    wmap = {w: i for i, w in enumerate(words)}
    vmap = {norm_word(v): i for i, v in enumerate(vocab)}
    cols = {k: [] for k in ["rid", "start", "subj", "corpus", "split", "vid", "wid", "nxt",
                            "sent", "pos", "onset", "nb", "ph"]}
    sent_base = 0
    for o in outs:
        r = recs[o["rid"]]
        n = len(o["start"])
        cols["rid"].append(np.full(n, o["rid"], np.int32))
        cols["start"].append(o["start"].astype(np.int64))
        cols["subj"].append(np.full(n, r["subject"], np.int16))
        cols["corpus"].append(np.full(n, CORPORA.index(r["corpus"]), np.int8))
        cols["split"].append(o["split"])
        cols["vid"].append(o["vid"].astype(np.int16))
        cols["wid"].append(np.array([wmap[w] for w in o["words"]], np.int32))
        nx = [(-100 if s == "" else vmap.get(s, len(vocab))) for s in o["nxt"]]
        cols["nxt"].append(np.array(nx, np.int16))
        s = o["sent"].astype(np.int64)
        cols["sent"].append(np.where(s >= 0, s + sent_base, -1))
        sent_base += int(s.max()) + 2 if len(s) else 1
        cols["pos"].append(o["pos"].astype(np.int32))
        cols["onset"].append(o["onset"].astype(np.float64))
        cols["nb"].append(o["nb"])
        cols["ph"].append(o["ph"])
    idx = {k: np.concatenate(v) for k, v in cols.items()}
    np.savez(cdir / "index.npz", **idx)

    runs = [dict(r, rid=i, T=o["T"]) for i, (r, o) in enumerate(zip(recs, outs))]
    for r in runs:
        r.pop("repos", None)
    tr = idx["split"] == 0
    pri_deep = np.bincount(idx["vid"][tr & (idx["subj"] == 0) & (idx["vid"] >= 0)], minlength=len(vocab))
    pri_broad = np.bincount(idx["vid"][tr & (idx["subj"] > 0) & (idx["vid"] >= 0)], minlength=len(vocab))
    save_json(dict(runs=runs, words=words, vocab=vocab, cut={f"{a}|{b}": v for (a, b), v in cut.items()},
                   prior_deep=pri_deep.tolist(), prior_broad=pri_broad.tolist(),
                   sfreq=dc["sfreq"], win=dc["win"], frames=dc["frames"]), cdir / "meta.json")

    n = len(idx["start"])
    for name, code in [("train", 0), ("val", 1), ("lock", 2)]:
        m = idx["split"] == code
        log(f"{name}: {m.sum()} words ({(m & (idx['vid'] >= 0)).sum()} in vocab), "
            f"subject 0: {(m & (idx['subj'] == 0)).sum()}, others: {(m & (idx['subj'] > 0)).sum()}")
    missing = [v for i, v in enumerate(vocab) if pri_deep[i] == 0]
    if missing:
        log("vocab words with no subject-0 training examples:", missing)
    log(f"{n} words, {len(words)} word types, {len(runs)} runs -> {cdir}")


if __name__ == "__main__":
    main()
