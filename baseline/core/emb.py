"""Target embeddings for the meaning head.

One vector per word type: a frozen text model's middle-layer state for the
word, averaged over its sub-tokens and over up to K sentence contexts from the
training transcripts. Vocabulary words never seen in context are embedded alone.

    python -m core.emb -c cfg/a.yaml
    python -m core.emb -c cfg/a.yaml --fake     # random vectors, for offline tests
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from . import ext
from .common import cli, load_json, log, norm_word, save_json


def main():
    args, cfg = cli(__doc__, lambda ap: ap.add_argument("--fake", action="store_true"))
    ec, P = cfg["emb"], cfg["paths"]
    meta = load_json(Path(P["cache"]) / "meta.json")
    idx = np.load(Path(P["cache"]) / "index.npz")
    words = list(meta["words"])
    extra = [norm_word(w) for w in ext.vocab() + ext.vocab2()]
    for w in extra:
        if w not in words:
            words.append(w)
    out = Path(P["emb"])
    out.mkdir(parents=True, exist_ok=True)

    if args.fake:
        rng = np.random.default_rng(0)
        E = rng.standard_normal((len(words), 64)).astype(np.float32)
        E /= np.linalg.norm(E, axis=1, keepdims=True)
        np.save(out / "E.npy", E.astype(np.float16))
        save_json(dict(words=words, model="fake", dim=64), out / "words.json")
        log(f"fake embeddings {E.shape} -> {out}")
        return

    import torch
    from transformers import AutoTokenizer, T5EncoderModel

    # rebuild sentences (subject 0 only: the same text the others heard)
    m = (idx["subj"] == 0) & (idx["sent"] >= 0)
    sent, pos, wid = idx["sent"][m], idx["pos"][m], idx["wid"][m]
    order = np.lexsort((pos, sent))
    sent, wid = sent[order], wid[order]
    bounds = np.flatnonzero(np.diff(sent)) + 1
    sents = [s for s in np.split(wid, bounds) if 1 < len(s) <= 80]
    log(f"{len(sents)} sentences for context")

    K = int(ec.get("contexts", 8))
    ctx = defaultdict(list)                     # wid -> [(sentence #, position)]
    rng = np.random.default_rng(0)
    for si in rng.permutation(len(sents)):
        for p, w in enumerate(sents[si]):
            if len(ctx[w]) < K:
                ctx[w].append((si, p))

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(ec["model"])
    net = T5EncoderModel.from_pretrained(ec["model"], torch_dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev).eval()
    layer = int(ec.get("layer", net.config.num_layers // 2))
    D = net.config.d_model
    acc = np.zeros((len(words), D), np.float64)
    num = np.zeros(len(words), np.int64)

    jobs = [(w, si, p) for w, lst in ctx.items() for si, p in lst]
    jobs.sort(key=lambda j: j[1])
    bs = int(ec.get("batch", 64))
    with torch.no_grad():
        for b in range(0, len(jobs), bs):
            chunk = jobs[b:b + bs]
            toks = [[words[i] for i in sents[si]] for _, si, _ in chunk]
            enc = tok(toks, is_split_into_words=True, return_tensors="pt", padding=True, truncation=True, max_length=160)
            hs = net(**{k: v.to(dev) for k, v in enc.items()}, output_hidden_states=True).hidden_states[layer]
            hs = hs.float().cpu().numpy()
            for r, (w, _, p) in enumerate(chunk):
                ids = enc.word_ids(r)
                sel = [t for t, wi in enumerate(ids) if wi == p]
                if sel:
                    acc[w] += hs[r, sel].mean(0)
                    num[w] += 1
            if (b // bs) % 200 == 0:
                log(f"contexts {b}/{len(jobs)}")
        # words with no context: embed alone
        alone = [i for i in range(len(words)) if num[i] == 0]
        for b in range(0, len(alone), bs):
            chunk = alone[b:b + bs]
            enc = tok([[words[i]] for i in chunk], is_split_into_words=True, return_tensors="pt", padding=True)
            hs = net(**{k: v.to(dev) for k, v in enc.items()}, output_hidden_states=True).hidden_states[layer]
            hs = hs.float().cpu().numpy()
            for r, i in enumerate(chunk):
                sel = [t for t, wi in enumerate(enc.word_ids(r)) if wi == 0]
                acc[i] = hs[r, sel].mean(0)
                num[i] = 1
    E = acc / num[:, None]
    E -= E.mean(0, keepdims=True)                    # remove the shared direction
    E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-8
    np.save(out / "E.npy", E.astype(np.float16))
    save_json(dict(words=words, model=ec["model"], layer=layer, dim=D), out / "words.json")
    log(f"embeddings {E.shape} -> {out}")


if __name__ == "__main__":
    main()
