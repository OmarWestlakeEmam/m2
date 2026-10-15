"""Thin wrapper around the dataset's official package.

This is the only module that imports it. Everything else talks to these
functions, so the rest of the code never depends on its API directly.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

PROC = "bads+headpos+sss+notch+bp+ds"


def vocab() -> list[str]:
    from pnpl.competition import load_vocabulary
    return load_vocabulary("primary")


def vocab2() -> list[str]:
    from pnpl.competition import load_vocabulary
    return load_vocabulary("moses")


def records() -> list[dict]:
    """Every run the dataset advertises, as plain dicts."""
    from pnpl.datasets.libribrain100.manifest import RUN_RECORDS
    from pnpl.datasets.libribrain100.constants import REPO_KEY_TO_ID
    out = []
    for r in RUN_RECORDS:
        repos = [REPO_KEY_TO_ID[r.repo]] + [v for k, v in REPO_KEY_TO_ID.items() if k != r.repo]
        out.append(dict(subject=int(r.subject), session=r.session, task=r.task, run=r.run,
                        corpus=r.corpus, partition=r.partition, repos=repos))
    return out


def _stem(rec: dict) -> str:
    return f"sub-{rec['subject']}_ses-{rec['session']}_task-{rec['task']}_run-{rec['run']}"


def h5_candidates(rec: dict) -> list[str]:
    stem, t = _stem(rec), rec["task"]
    return [f"{t}/derivatives/serialised/{stem}_proc-{PROC}_meg.h5",
            f"{t}/derivatives/serialised_competition/{stem}_proc-{PROC}_desc-firsthalf_meg.h5",
            f"{t}/derivatives/serialised_competition/{stem}_proc-{PROC}_desc-firstquarter_meg.h5"]


def rel_paths(rec: dict) -> tuple[str, str]:
    return h5_candidates(rec)[0], f"{rec['task']}/derivatives/events/{_stem(rec)}_events.tsv"


def local_paths(rec: dict, raw_dir: str) -> tuple[Path, Path]:
    ev = Path(raw_dir) / rel_paths(rec)[1]
    for c in h5_candidates(rec):
        if (Path(raw_dir) / c).exists():
            return Path(raw_dir) / c, ev
    return Path(raw_dir) / h5_candidates(rec)[0], ev


def _fetch_one(repos: list[str], rel: str, raw_dir: str) -> bool:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError
    for repo in repos:
        try:
            hf_hub_download(repo_id=repo, repo_type="dataset", filename=rel, local_dir=raw_dir)
            return True
        except (EntryNotFoundError, RepositoryNotFoundError):
            continue
    return False


def download(rec: dict, raw_dir: str) -> bool:
    h5, ev = local_paths(rec, raw_dir)
    ok_ev = ev.exists() or _fetch_one(rec["repos"], rel_paths(rec)[1], raw_dir)
    ok_h5 = h5.exists() or any(_fetch_one(rec["repos"], c, raw_dir) for c in h5_candidates(rec))
    return ok_ev and ok_h5


# --------------------------------------------------------------------------- scoring set

class Scoring:
    """Unlabelled scoring windows for one track, in the official row order."""

    def __init__(self, track: str, ho_dir: str, download: bool = True):
        from pnpl.competition import LibriBrainCompetitionHoldout
        self._h = LibriBrainCompetitionHoldout(track=track, data_path=ho_dir, download=download)
        self.indices = self._h.indices
        self.meta = self._h.metadata
        self.subjects = list(self._h.subjects)

    def __len__(self):
        return len(self.meta)

    def file(self, subject: int, source: str) -> str:
        return self._h._ensure_file(subject, source)

    def windows(self, batch_size: int = 256):
        """Yield (windows (B,306,250) float32, metas) in canonical order."""
        yield from self._h.iter_windows(batch_size=batch_size)


def scoring_relpath(subject: int, source: str) -> str:
    """Relative path of one subject's scoring file ('sentence' or 'word')."""
    from pnpl.competition.holdout import _subject_filename
    return _subject_filename(subject, source)


def write(path: str, indices, p1: np.ndarray, p2: np.ndarray | None) -> str:
    from pnpl.competition import write_submission
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    return str(write_submission(path, indices=indices, primary_probs=p1, secondary_probs=p2))
