"""Distances under each source measure, as Chroma defines them (smaller = closer)."""

from __future__ import annotations

import numpy as np

from .model import Metric


def distances(metric: Metric, q: np.ndarray, m: np.ndarray) -> np.ndarray:
    if metric == Metric.L2:
        return ((m - q) ** 2).sum(axis=1)  # Chroma's l2 is squared
    if metric == Metric.IP:
        return 1.0 - m @ q
    norms = np.linalg.norm(m, axis=1) * (np.linalg.norm(q) or 1.0)
    return 1.0 - (m @ q) / np.where(norms == 0, 1.0, norms)
