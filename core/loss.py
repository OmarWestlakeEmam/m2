"""Losses for the four heads."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .phon import FEAT_MAT


def word(z, vid, log_prior, tau, smooth):
    """Logit-adjusted cross-entropy on in-vocabulary rows (targets balanced accuracy)."""
    m = vid >= 0
    if not m.any():
        return z.sum() * 0
    return F.cross_entropy(z[m] + tau * log_prior, vid[m], label_smoothing=smooth)


def meaning(net, m, E, wid):
    """Sigmoid contrastive loss over the batch; repeats of a word are positives."""
    tgt = E[wid]
    logits = net.mean_logits(m, tgt)
    pos = (wid[:, None] == wid[None, :]).float()
    return -F.logsigmoid((2 * pos - 1) * logits).sum() / m.shape[0]


def sound(ph_logits, ft_logits, ph):
    ok = ph >= 0
    if not ok.any():
        return ph_logits.sum() * 0
    l1 = F.cross_entropy(ph_logits[ok], ph[ok])
    fm = torch.as_tensor(FEAT_MAT, device=ph.device)[ph[ok]]
    l2 = F.binary_cross_entropy_with_logits(ft_logits[ok], fm)
    return l1 + l2


def neighbours(nb_logits, nx_logits, nb, nxt):
    l1 = F.binary_cross_entropy_with_logits(nb_logits, nb)
    if (nxt >= 0).any():
        l1 = l1 + F.cross_entropy(nx_logits, nxt, ignore_index=-100)
    return l1
