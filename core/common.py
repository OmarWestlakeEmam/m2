"""Shared helpers: config loading, paths, seeding, logging, metrics."""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml


# --------------------------------------------------------------------------- config

def _set_dotted(d: dict, key: str, value: Any) -> None:
    parts = key.split(".")
    cur = d
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = value


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    if "base" in cfg:
        cfg = _merge(_load_yaml(path.parent / cfg.pop("base")), cfg)
    return cfg


def load_cfg(path: str, overrides: list[str] | None = None) -> dict:
    """Load a YAML config. A config may name a parent with `base: other.yaml`.

    Overrides are `a.b.c=value` strings; values are parsed as YAML.
    """
    cfg = _load_yaml(Path(path))
    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"override must look like key=value, got {ov!r}")
        k, v = ov.split("=", 1)
        _set_dotted(cfg, k, yaml.safe_load(v))
    root = Path(cfg.get("paths", {}).get("root", "./d")).expanduser()
    p = cfg.setdefault("paths", {})
    for name, sub in [("raw", "raw"), ("ho", "ho"), ("cache", "c"), ("emb", "e"),
                      ("runs", "r"), ("out", "o")]:
        p.setdefault(name, str(root / sub))
    return cfg


def cli(desc: str, extra=None) -> tuple[argparse.Namespace, dict]:
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("-c", "--cfg", default="cfg/a.yaml")
    ap.add_argument("-s", "--set", nargs="*", default=[], help="overrides key=value")
    if extra:
        extra(ap)
    args = ap.parse_args()
    cfg = load_cfg(args.cfg, args.set)
    return args, cfg


# --------------------------------------------------------------------------- misc

def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def log(*a) -> None:
    print(time.strftime("%H:%M:%S"), *a, flush=True)
    sys.stdout.flush()


def save_json(obj, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def norm_word(w) -> str:
    """Lower-case, unify apostrophes, strip surrounding punctuation."""
    w = str(w).strip().lower().replace("’", "'").replace("‘", "'").replace("`", "'")
    return w.strip(" .,;:!?\"()[]{}-“”")


# --------------------------------------------------------------------------- metric

def bacc_at_k(scores: np.ndarray, labels: np.ndarray, k: int = 10) -> float:
    """Balanced top-k accuracy over the classes present in `labels`."""
    if len(labels) == 0:
        return float("nan")
    k = min(k, scores.shape[1])
    topk = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    hit = (topk == labels[:, None]).any(1)
    rec = [hit[labels == c].mean() for c in np.unique(labels)]
    return float(np.mean(rec))
