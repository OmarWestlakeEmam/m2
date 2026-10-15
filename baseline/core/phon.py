"""ARPAbet phoneme set (39 + silence) and 14 binary articulatory features."""
from __future__ import annotations

import numpy as np

PHONES = ['aa', 'ae', 'ah', 'ao', 'aw', 'ay', 'b', 'ch', 'd', 'dh', 'eh', 'er', 'ey', 'f', 'g',
          'hh', 'ih', 'iy', 'jh', 'k', 'l', 'm', 'n', 'ng', 'ow', 'oy', 'p', 'r', 's', 'sh', 't',
          'th', 'uh', 'uw', 'v', 'w', 'y', 'z', 'zh']
SIL = len(PHONES)            # class 39 = silence / between words
N_PH = len(PHONES) + 1       # 40 frame classes
SIL_LABELS = {"sil", "sp", "spn", "h#", "pau", "epi"}

FEATS = ["cons", "syll", "son", "voice", "nasal", "cont", "lab", "cor", "dors",
         "high", "low", "back", "round", "diph"]

_V = {"syll", "son", "voice", "cont"}           # every vowel
_TABLE = {
    # vowels
    "aa": _V | {"low", "back"},
    "ae": _V | {"low"},
    "ah": _V | {"back"},
    "ao": _V | {"low", "back", "round"},
    "aw": _V | {"low", "back", "diph"},
    "ay": _V | {"low", "diph"},
    "eh": _V,
    "er": _V | {"cor"},
    "ey": _V | {"diph"},
    "ih": _V | {"high"},
    "iy": _V | {"high"},
    "ow": _V | {"back", "round", "diph"},
    "oy": _V | {"back", "round", "diph"},
    "uh": _V | {"high", "back", "round"},
    "uw": _V | {"high", "back", "round"},
    # consonants
    "b": {"cons", "voice", "lab"},
    "ch": {"cons", "cor", "high"},
    "d": {"cons", "voice", "cor"},
    "dh": {"cons", "voice", "cont", "cor"},
    "f": {"cons", "cont", "lab"},
    "g": {"cons", "voice", "dors", "high", "back"},
    "hh": {"cont"},
    "jh": {"cons", "voice", "cor", "high"},
    "k": {"cons", "dors", "high", "back"},
    "l": {"cons", "son", "voice", "cont", "cor"},
    "m": {"cons", "son", "voice", "nasal", "lab"},
    "n": {"cons", "son", "voice", "nasal", "cor"},
    "ng": {"cons", "son", "voice", "nasal", "dors", "high", "back"},
    "p": {"cons", "lab"},
    "r": {"cons", "son", "voice", "cont", "cor"},
    "s": {"cons", "cont", "cor"},
    "sh": {"cons", "cont", "cor", "high"},
    "t": {"cons", "cor"},
    "th": {"cons", "cont", "cor"},
    "v": {"cons", "voice", "cont", "lab"},
    "w": {"son", "voice", "cont", "lab", "dors", "high", "back", "round"},
    "y": {"son", "voice", "cont", "dors", "high"},
    "z": {"cons", "voice", "cont", "cor"},
    "zh": {"cons", "voice", "cont", "cor", "high"},
}

PH_INDEX = {p: i for i, p in enumerate(PHONES)}

# (40, 14) feature matrix; silence row is all zeros.
FEAT_MAT = np.zeros((N_PH, len(FEATS)), dtype=np.float32)
for _p, _fs in _TABLE.items():
    for _f in _fs:
        FEAT_MAT[PH_INDEX[_p], FEATS.index(_f)] = 1.0


def phone_id(label) -> int:
    """'dh_B' -> index of 'dh'; silence -> SIL; unknown -> -100 (ignored)."""
    s = str(label).strip().lower()
    base = s.split("_")[0].rstrip("012")
    if base in SIL_LABELS or s in SIL_LABELS:
        return SIL
    return PH_INDEX.get(base, -100)
