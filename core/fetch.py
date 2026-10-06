"""Download the training runs and the two scoring sets into paths.raw / paths.ho.

    python -m core.fetch -c cfg/a.yaml            # everything selected in cfg.data
    python -m core.fetch -c cfg/a.yaml --only-ho  # just the scoring sets
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

from . import ext
from .common import cli, log


def select(recs: list[dict], dcfg: dict) -> list[dict]:
    subs = dcfg.get("subjects", "all")
    if subs == "all":
        keep_s = None
    elif subs == "deep":
        keep_s = {0}
    elif subs == "broad":
        keep_s = set(range(1, 33))
    else:
        keep_s = {int(s) for s in subs}
    corp = set(dcfg.get("corpora", ["sherlock", "timit", "mocha", "podcasts"]))
    return [r for r in recs if (keep_s is None or r["subject"] in keep_s) and r["corpus"] in corp]


def main():
    args, cfg = cli(__doc__, lambda ap: (ap.add_argument("--only-ho", action="store_true"),
                                         ap.add_argument("-j", type=int, default=8)))
    raw, ho = cfg["paths"]["raw"], cfg["paths"]["ho"]
    if not args.only_ho:
        recs = select(ext.records(), cfg["data"])
        log(f"{len(recs)} runs selected -> {raw}")
        missing = []
        with ThreadPoolExecutor(args.j) as ex:
            futs = {ex.submit(ext.download, r, raw): r for r in recs}
            for i, f in enumerate(as_completed(futs), 1):
                r = futs[f]
                try:
                    ok = f.result()
                except Exception as e:  # network errors: report and continue
                    ok = False
                    log("error", r["subject"], r["task"], r["session"], repr(e)[:200])
                if not ok:
                    missing.append(r)
                if i % 10 == 0 or i == len(recs):
                    log(f"{i}/{len(recs)} done, {len(missing)} unavailable")
        for r in missing:
            log("unavailable:", r["subject"], r["task"], r["session"], r["run"])
    for track in ("deep", "broad"):
        s = ext.Scoring(track, ho, download=True)
        log(f"scoring set {track}: {len(s)} rows")


if __name__ == "__main__":
    main()
