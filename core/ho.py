"""Scoring-set windows, prepared exactly like training windows.

Per subject: z-score each channel with that subject's own statistics (over all of
its unlabelled scoring MEG), clip, cast to float16, store time-major. The result
is cached at paths.out/w_<track>.npy so every member reads the same tensors.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import ext
from .common import load_json, log, save_json
from .data import sid_of


def _subject_stats(S: ext.Scoring, subject: int) -> tuple[np.ndarray, np.ndarray]:
    s1 = np.zeros(306)
    s2 = np.zeros(306)
    n = 0
    with np.load(S.file(subject, "sentence")) as d:
        meg, nt = d["meg"], d["sentence_n_times"].astype(int)
        for i in range(meg.shape[0]):
            seg = meg[i, :, : nt[i]].astype(np.float64)
            s1 += seg.sum(1)
            s2 += (seg ** 2).sum(1)
            n += seg.shape[1]
    with np.load(S.file(subject, "word")) as d:
        meg = d["meg"].astype(np.float64)
        s1 += meg.sum((0, 2))
        s2 += (meg ** 2).sum((0, 2))
        n += meg.shape[0] * meg.shape[2]
    mu = s1 / max(n, 1)
    sd = np.sqrt(np.maximum(s2 / max(n, 1) - mu ** 2, 0))
    sd[sd == 0] = 1.0
    return mu.astype(np.float32), sd.astype(np.float32)


def load(cfg: dict, track: str, pre: int = 0) -> dict:
    out = Path(cfg["paths"]["out"])
    out.mkdir(parents=True, exist_ok=True)
    tag = f"w_{track}" if not pre else f"w_{track}_p{pre}"
    xp, mp = out / f"{tag}.npy", out / f"{tag}.json"
    if xp.exists() and mp.exists():
        m = load_json(mp)
        return dict(x=np.load(xp, mmap_mode="r"), **{k: np.asarray(v) for k, v in m.items() if k != "meta"},
                    meta=m["meta"])
    if pre:
        return _load_pre(cfg, track, pre, xp, mp)
    S = ext.Scoring(track, cfg["paths"]["ho"], download=True)
    N = len(S)
    log(f"preparing {N} scoring windows for {track}")
    stats = {s: _subject_stats(S, s) for s in S.subjects}
    X = np.lib.format.open_memmap(xp, mode="w+", dtype=np.float16, shape=(N, 250, 306))
    i = 0
    for xb, metas in S.windows(batch_size=512):
        for j, m in enumerate(metas):
            mu, sd = stats[m["subject"]]
            z = (xb[j] - mu[:, None]) / sd[:, None]
            X[i] = np.clip(z, -30, 30).T.astype(np.float16)
            i += 1
    X.flush()
    assert i == N
    meta = [dict(subject=int(m["subject"]), source=m["source"], epoch=int(m["epoch"]), word=int(m["word"]))
            for m in S.meta]
    m = dict(indices=[int(v) for v in S.indices], sid=[sid_of(d["subject"]) for d in meta],
             subject=[d["subject"] for d in meta], meta=meta)
    save_json(m, mp)
    return load(cfg, track)


def _load_pre(cfg, track, pre, xp, mp) -> dict:
    """Windows that start `pre` samples before the word onset, cut from the sentence
    recordings (zeros before the sentence's first sample, as in training). Isolated
    word epochs (0-1 s only) get zeros in the pre-onset part. The 0-1 s part is checked
    against the official windows."""
    base = load(cfg, track, 0)                       # official 1 s windows + row metadata
    S = ext.Scoring(track, cfg["paths"]["ho"], download=True)
    meta, N, W = base["meta"], len(base["meta"]), 250
    L = pre + W
    log(f"preparing {N} scoring windows for {track} with {pre} pre-onset samples")
    stats = {s: _subject_stats(S, s) for s in S.subjects}
    X = np.lib.format.open_memmap(xp, mode="w+", dtype=np.float16, shape=(N, L, 306))
    cache, worst = {}, 0.0
    for i, m in enumerate(meta):
        s, src = m["subject"], m["source"]
        if (s, src) not in cache:
            cache.clear()
            d = np.load(S.file(s, src))
            cache[(s, src)] = {k: d[k] for k in d.files}
            assert float(cache[(s, src)]["sfreq"]) == 250.0
        d = cache[(s, src)]
        mu, sd = stats[s]
        w = np.zeros((306, L), np.float32)
        if src == "sentence":
            seg = d["meg"][m["epoch"]][:, : int(d["sentence_n_times"][m["epoch"]])]
            on = int(round(float(d["word_onsets_s"][m["epoch"], m["word"]]) * 250))
            a = on - pre
            lo, hi = max(a, 0), min(a + L, seg.shape[1])
            w[:, lo - a:hi - a] = (seg[:, lo:hi] - mu[:, None]) / sd[:, None]
        else:
            w[:, pre:] = (d["meg"][m["epoch"]] - mu[:, None]) / sd[:, None]
        w = np.clip(w, -30, 30)
        if i < 2000:   # parity of the 0-1 s part with the official windows
            worst = max(worst, float(np.abs(w[:, pre:].T - base["x"][i].astype(np.float32)).max()))
        X[i] = w.T.astype(np.float16)
    X.flush()
    log(f"parity vs official windows (first 2000 rows): max abs diff {worst:.4f}")
    if worst > 0.05:
        raise SystemExit(f"window parity check failed ({worst:.3f}): stop and report")
    m = dict(indices=[int(v) for v in base["indices"]], sid=[int(v) for v in base["sid"]],
             subject=[int(v) for v in base["subject"]], meta=meta)
    save_json(m, mp)
    return load(cfg, track, pre)


def batches(x: np.ndarray, sid: np.ndarray, bs: int = 512):
    import torch
    for a in range(0, len(x), bs):
        yield torch.from_numpy(np.ascontiguousarray(x[a:a + bs])), torch.as_tensor(sid[a:a + bs], dtype=torch.long)
