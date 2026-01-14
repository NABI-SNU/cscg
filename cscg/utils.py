"""Utility functions for CSCG module."""

from __future__ import annotations

import numba as nb
import numpy as np


def validate_seq(x: np.ndarray, a: np.ndarray, n_clones: np.ndarray | None = None) -> None:
    """Validate an input sequence of observations x and actions a"""
    assert len(x) == len(a) > 0
    assert len(x.shape) == len(a.shape) == 1, "Flatten your array first"
    assert x.dtype == a.dtype == np.int64
    assert 0 <= x.min(), "Number of emissions inconsistent with training sequence"
    if n_clones is not None:
        assert len(n_clones.shape) == 1, "Flatten your array first"
        assert n_clones.dtype == np.int64
        assert np.all(n_clones > 0), "You can't provide zero clones for any emission"
        assert (
            x.max() < n_clones.shape[0]
        ), "Number of emissions inconsistent with training sequence"
    return None


@nb.njit
def rargmax(x):
    """Random argmax - returns a random index from all indices that have the maximum value.

    This is used to avoid systematic bias when there are ties in max-product decoding.
    """
    # return x.argmax()  # <- favors clustering towards smaller state numbers
    m = x[0]
    for i in range(1, x.size):
        if x[i] > m:
            m = x[i]
    chosen = 0
    count = 0
    for i in range(x.size):
        if x[i] == m:
            count += 1
            if np.random.random() < 1.0 / count:
                chosen = i
    return chosen
