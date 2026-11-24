"""Turning raw cosine similarity into a score in [0, 1] that means something.

Raw CLIP cosines on this dataset live in a narrow band (with CLIP B/32: random noise ~0.56,
median nearest neighbour ~0.86), so "0.62" says nothing to a user. We use the empirical CDF of
the dataset's own top-k leave-one-out neighbour similarities:

    score(c) = fraction of "real neighbour" similarities in the dataset that are <= c

So 0.9 reads as "more similar than 90% of the neighbour pairs in this catalogue". It is
monotonic (ranking is unchanged), bounded, and calibrated to *this* index and model.
"""

from __future__ import annotations

import numpy as np

N_QUANTILES = 1001


def loo_neighbour_sims(emb: np.ndarray, k: int = 10) -> np.ndarray:
    """(n, k) top-k cosine similarities of each item to the *other* items, descending."""
    sims = emb @ emb.T
    np.fill_diagonal(sims, -np.inf)
    k = min(k, len(emb) - 1)
    top = -np.sort(-sims, axis=1)[:, :k]
    return top.astype(np.float32)


def reference_quantiles(emb: np.ndarray, k: int = 10) -> np.ndarray:
    """Compact reference distribution: N_QUANTILES quantiles of the top-k LOO sims."""
    return np.quantile(loo_neighbour_sims(emb, k).ravel(), np.linspace(0, 1, N_QUANTILES))


def calibrate(cos: np.ndarray | float, ref_q: np.ndarray) -> np.ndarray:
    """Map cosine(s) to [0, 1] with the empirical CDF, anchored at 0 -> 0 and 1 -> 1."""
    xp = np.concatenate([[min(0.0, ref_q[0] - 1e-6)], ref_q, [max(1.0, ref_q[-1] + 1e-6)]])
    xp = np.maximum.accumulate(xp + np.arange(len(xp)) * 1e-9)  # strictly increasing
    fp = np.concatenate([[0.0], np.linspace(0, 1, len(ref_q)), [1.0]])
    return np.clip(np.interp(cos, xp, fp), 0.0, 1.0)


def nn1_threshold(emb: np.ndarray, percentile: float = 2.0) -> float:
    """Low percentile of each item's nearest-neighbour cosine.

    Used as the out-of-distribution cut: a query whose best match is below what ~98% of
    dataset images achieve with their own nearest neighbour does not belong here. The
    minimum would be too permissive (a solid-colour square scored 0.657
    vs a minimum of 0.656 in our evaluation).
    """
    return float(np.percentile(loo_neighbour_sims(emb, 1)[:, 0], percentile))
