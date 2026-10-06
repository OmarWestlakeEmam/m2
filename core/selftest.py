"""End-to-end check on synthetic data in the real file formats (CPU, ~5 min).

Plants a word-specific pattern in fake MEG, runs every stage, and checks that the
written submission scores well above chance against the planted labels.

    python -m core.selftest [--keep]
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from . import ext
from .common import bacc_at_k, log, norm_word

SF = 250
RUNS = [(0, "Sherlock2", "1", "1"), (0, "Sherlock2", "2", "1"), (0, "Sherlock2", "3", "1"),
        (0, "Sherlock3", "1", "1"), (0, "Sherlock1", "11", "2"), (0, "Sherlock1", "12", "2"),
        (0, "TheMoth", "29", "1"), (0, "TheMoth", "30", "1"), (0, "MOCHATIMIT", "2", "1"),
        (1, "Sherlock1", "11", "1"), (2, "Sherlock1", "11", "1"), (3, "Sherlock1", "11", "1")]


class World:
    """Word templates + a sensor mixing per subject."""

    def __init__(self, seed=0):
        r = np.random.default_rng(seed)
        self.vocab = ext.vocab()
        self.others = [f"zz{i}" for i in range(150)]
        from scipy.ndimage import gaussian_filter1d   # smooth, like real evoked responses (< 40 Hz)
        def t():
            a = gaussian_filter1d(r.standard_normal((16, 100)), 4, axis=1)[:, 20:80]
            return (a / a.std()).astype(np.float32)
        self.tmpl = {w: t() for w in self.vocab + self.others}
        self.mix = {s: r.standard_normal((306, 16)).astype(np.float32) / 4 for s in range(40)}
        self.r = r

    def word(self):
        if self.r.random() < 0.6:
            p = 1.0 / np.arange(1, len(self.vocab) + 1) ** 0.6
            return self.vocab[self.r.choice(len(self.vocab), p=p / p.sum())]
        return self.others[self.r.integers(len(self.others))]

    def stamp(self, x, subject, w, at):
        pat = self.mix[subject] @ self.tmpl[w]           # (306, 60)
        e = min(x.shape[1], at + 60)
        x[:, at:e] += 0.9 * pat[:, : e - at]


def make_run(world, raw, subject, task, ses, run, dur_s=150.0):
    rec = dict(subject=subject, session=ses, task=task, run=run)
    h5p, evp = ext.local_paths(rec, raw)
    h5p.parent.mkdir(parents=True, exist_ok=True)
    evp.parent.mkdir(parents=True, exist_ok=True)
    T = int(dur_s * SF)
    x = world.r.standard_normal((306, T)).astype(np.float32)
    rows, t, si, wi = [], 1.0, 0, 0
    while t < dur_s - 2:
        w = world.word()
        at = int(round(t * SF))
        world.stamp(x, subject, w, at)
        d = 0.18 + 0.1 * world.r.random()
        rows.append(dict(kind="word", segment=w, timemeg=t, duration=d, sentenceidx=si, wordidx=wi))
        for k, ph in enumerate(["dh_B", "ah_I", "t_E"]):
            rows.append(dict(kind="phoneme", segment=ph, timemeg=t + k * d / 3, duration=d / 3,
                             sentenceidx=si, wordidx=wi))
        wi += 1
        if world.r.random() < 0.15:
            si, wi = si + 1, 0
            rows.append(dict(kind="phoneme", segment="sil", timemeg=t + d, duration=0.2, sentenceidx=si, wordidx=-1))
            t += 0.3
        t += d + 0.12
    with h5py.File(h5p, "w") as f:
        f.create_dataset("data", data=x * 1e-12)
        f.attrs["sample_frequency"] = float(SF)
    pd.DataFrame(rows).to_csv(evp, sep="\t", index=False)


def make_scoring(world, ho_dir):
    d = Path(ho_dir)
    truth = {}
    for s in range(40):
        lab = []
        n_sent, W = 4, 6
        Tm = int(SF * (W * 0.45 + 1.5))
        meg = np.zeros((n_sent, 306, Tm), np.float32)
        on = np.full((n_sent, W), np.nan)
        mask = np.zeros((n_sent, W), bool)
        nt = np.zeros(n_sent, np.int64)
        for i in range(n_sent):
            x = world.r.standard_normal((306, Tm)).astype(np.float32)
            n_w = world.r.integers(3, W + 1)
            for j in range(n_w):
                w = world.vocab[world.r.integers(len(world.vocab))]
                t = 0.2 + 0.45 * j
                world.stamp(x, s, w, int(round(t * SF)))
                on[i, j], mask[i, j] = t, True
                lab.append(world.vocab.index(w))
            nt[i] = int(round((0.2 + 0.45 * (n_w - 1) + 1.05) * SF))
            meg[i, :, : nt[i]] = x[:, : nt[i]] * 1e-12
        smask = np.arange(Tm)[None] < nt[:, None]
        f1 = d / ext.scoring_relpath(s, "sentence")
        f1.parent.mkdir(parents=True, exist_ok=True)
        np.savez(f1, meg=meg, sentence_n_times=nt, sentence_sample_mask=smask,
                 word_onsets_s=on, word_mask=mask, sfreq=float(SF), sentence_extra_end_s=1.05)
        ep = []
        for _ in range(5):
            x = world.r.standard_normal((306, 250)).astype(np.float32)
            w = world.vocab[world.r.integers(len(world.vocab))]
            world.stamp(x, s, w, 0)
            ep.append(x * 1e-12)
            lab.append(world.vocab.index(w))
        np.savez(d / ext.scoring_relpath(s, "word"), meg=np.stack(ep), sfreq=float(SF), tmin=0.0, tmax=1.0)
        truth[s] = lab
    return truth


def sh(*cmd, cwd):
    log("$", " ".join(cmd))
    r = subprocess.run([sys.executable, "-m", *cmd], cwd=cwd)
    if r.returncode:
        raise SystemExit(f"FAILED: {' '.join(cmd)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--pre", type=int, default=0, help="test pre-onset windows (e.g. 125)")
    a = ap.parse_args()
    repo = Path(__file__).resolve().parents[1]
    tmp = Path(tempfile.mkdtemp(prefix="t_"))
    root = tmp / "d"
    world = World()
    log(f"synthetic data in {root}")
    for s, task, ses, run in RUNS:
        make_run(world, str(root / "raw"), s, task, ses, run)
    truth = make_scoring(world, str(root / "ho"))

    cfg = tmp / "t.yaml"
    cfg.write_text(f"""base: {repo / 'cfg' / 'a.yaml'}
paths: {{root: {root}}}
data: {{pre: {a.pre}}}
model: {{d: 64, depth: 2, heads: 4, ff: 128, out: 128, rank: 8}}
train: {{epochs: 3, steps_per_epoch: 60, batch: 64, warmup: 20, workers: 0, ema: 0.95, lr: 1.0e-3}}
ctx: {{epochs: 3, d: 64, heads: 4, depth: 2, batch: 16}}
seq: {{d: 64, depth: 1, heads: 4, batch: 4, epochs: 2, steps: 40, chunk: 8}}
""")
    c = ["-c", str(cfg)]
    sh("core.prep", *c, "-j", "2", cwd=repo)
    sh("core.emb", *c, "--fake", cwd=repo)
    sh("core.train", *c, "-s", "train.name=j", cwd=repo)
    sh("core.train", *c, "-s", "train.name=jd", "train.who=deep", f"train.init={root}/r/j/best.pt",
       "train.epochs=1", "train.lr=5.0e-4", "train.select=deep", "train.deep_share=1.0", cwd=repo)
    sh("core.train", *c, "-s", "train.name=jb", "train.who=broad", f"train.init={root}/r/j/best.pt",
       "train.epochs=1", "train.select=broad", "train.deep_share=0.0",
       "train.freeze=[spatial.shared, stem, enc.0.]", cwd=repo)
    sh("core.calib", *c, "--who", "deep", "--members", f"{root}/r/j/best.pt", f"{root}/r/jd/best.pt", cwd=repo)
    sh("core.calib", *c, "--who", "broad", "--members", f"{root}/r/jb/best.pt", cwd=repo)
    sh("core.infer", *c, "--track", "deep", "--members", f"{root}/r/j/best.pt", f"{root}/r/jd/best.pt",
       "--calib", f"{root}/o/calib_deep.json", "--out", f"{root}/o/deep.csv", cwd=repo)
    sh("core.infer", *c, "--track", "broad", "--members", f"{root}/r/jb/best.pt",
       "--calib", f"{root}/o/calib_broad.json", "--out", f"{root}/o/broad.csv", cwd=repo)
    # context model path
    sh("core.train", *c, "-s", "train.name=f0", "train.fold=0/2", "train.epochs=3", cwd=repo)
    sh("core.ctx", "extract", *c, "--name", "c0", "--member", f"{root}/r/f0/best.pt", "--fold", "0/2", cwd=repo)
    sh("core.ctx", "fit", *c, "--name", "c0", cwd=repo)
    sh("core.ctx", "val", *c, "--name", "c0", "--who", "deep", cwd=repo)
    sh("core.ctx", "predict", *c, "--name", "c0", "--track", "deep", cwd=repo)
    sh("core.calib", *c, "--who", "deep", "--members", f"{root}/r/jd/best.pt",
       "--extra", f"{root}/o/xv_deep_c0.npz", "--out", f"{root}/o/calib_ctx.json", cwd=repo)
    sh("core.infer", *c, "--track", "deep", "--members", f"{root}/r/jd/best.pt", "--extra", f"{root}/o/x_deep_c0.npz",
       "--calib", f"{root}/o/calib_ctx.json", "--out", f"{root}/o/deep_ctx.csv", cwd=repo)

    # end-to-end sentence model
    sh("core.seq", "train", *c, "-s", f"train.init={root}/r/j/best.pt", "--name", "s0", cwd=repo)
    sh("core.seq", "val", *c, "--name", "s0", "--who", "deep", cwd=repo)
    sh("core.seq", "predict", *c, "--name", "s0", "--track", "deep", cwd=repo)
    sh("core.calib", *c, "--who", "deep", "--members", f"{root}/r/jd/best.pt",
       "--extra", f"{root}/o/xv_deep_s0.npz", "--out", f"{root}/o/calib_seq.json", cwd=repo)
    sh("core.infer", *c, "--track", "deep", "--members", f"{root}/r/jd/best.pt", "--extra", f"{root}/o/x_deep_s0.npz",
       "--calib", f"{root}/o/calib_seq.json", "--out", f"{root}/o/deep_seq.csv", cwd=repo)

    ok = True
    for name, subs, f in [("deep", [0], "deep.csv"), ("broad", list(range(1, 40)), "broad.csv"),
                          ("deep+ctx", [0], "deep_ctx.csv"), ("deep+seq", [0], "deep_seq.csv")]:
        df = pd.read_csv(root / "o" / f)
        vocab = ext.vocab()
        assert list(df.columns[1:51]) == vocab and len(df.columns) == 101, "bad columns"
        y = np.concatenate([truth[s] for s in subs])
        p = df.iloc[:, 1:51].to_numpy()
        assert len(p) == len(y), (len(p), len(y))
        assert np.allclose(p.sum(1), 1, atol=1e-3)
        v = bacc_at_k(p, y, 10)
        log(f"{name}: {len(y)} rows, BAcc@10 vs planted labels {v:.3f} (chance 0.20)")
        if name == "broad":  # unseen synthetic subjects have random sensor mixing: only 1-3 are checked
            n_seen = sum(len(truth[s]) for s in (1, 2, 3))
            v = bacc_at_k(p[:n_seen], y[:n_seen], 10)
            log(f"   subjects 1-3 (have adapters): {v:.3f}; "
                f"4-39 (generic path): {bacc_at_k(p[n_seen:], y[n_seen:], 10):.3f}")
        ok &= v > 0.28
    log("SELFTEST PASSED" if ok else "SELFTEST: pipeline ran but scores are low")
    if not a.keep:
        shutil.rmtree(tmp, ignore_errors=True)
    else:
        log(f"kept {tmp}")


if __name__ == "__main__":
    main()
