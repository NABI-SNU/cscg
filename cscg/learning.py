"""EM update functions for learning transition and emission parameters."""

from __future__ import annotations

import numba as nb
import numpy as np


@nb.njit
def updateC(
    C: np.ndarray,
    T: np.ndarray,
    n_clones: np.ndarray,
    mess_fwd: np.ndarray,
    mess_bwd: np.ndarray,
    x: np.ndarray,
    a: np.ndarray,
):
    """Update transition count matrix C using expected counts from forward-backward.

    This is the E-step update for EM learning of transition parameters.
    Accumulates expected transition counts from posterior marginals.

    Parameters
    ----------
    C : np.ndarray
        Count tensor to update in-place, shape (n_actions, n_states, n_states)
    T : np.ndarray
        Current transition tensor, shape (n_actions, n_states, n_states)
    n_clones : np.ndarray
        Number of clones per observation, shape (n_emissions,)
    mess_fwd : np.ndarray
        Forward messages, shape (sum(n_clones[x]),)
    mess_bwd : np.ndarray
        Backward messages, shape (sum(n_clones[x]),)
    x : np.ndarray
        Observation sequence, shape (T,)
    a : np.ndarray
        Action sequence, shape (T,)
    """
    state_loc: np.ndarray = np.concatenate((np.array([0], dtype=n_clones.dtype), n_clones)).cumsum()
    mess_loc: np.ndarray = np.concatenate(
        (np.array([0], dtype=n_clones.dtype), n_clones[x])
    ).cumsum()
    timesteps: int = len(x)
    C[:] = 0
    for t in range(1, timesteps):
        aij, i, j = (
            a[t - 1],
            x[t - 1],
            x[t],
        )  # at time t-1 -> t we go from observation i to observation j
        (tm1_start, tm1_stop), (t_start, t_stop) = (
            mess_loc[t - 1 : t + 1],
            mess_loc[t : t + 2],
        )
        (i_start, i_stop), (j_start, j_stop) = (
            state_loc[i : i + 2],
            state_loc[j : j + 2],
        )
        q = (
            mess_fwd[tm1_start:tm1_stop].reshape(-1, 1)
            * T[aij, i_start:i_stop, j_start:j_stop]
            * mess_bwd[t_start:t_stop].reshape(1, -1)
        )
        q /= q.sum()
        C[aij, i_start:i_stop, j_start:j_stop] += q


def updateCE(
    CE: np.ndarray,
    E: np.ndarray,
    n_clones: np.ndarray,
    mess_fwd: np.ndarray,
    mess_bwd: np.ndarray,
    x: np.ndarray,
    a: np.ndarray,
):
    """Update emission count matrix CE using expected counts from forward-backward.

    This is the E-step update for EM learning of emission parameters.
    Accumulates expected emission counts from posterior marginals.

    Parameters
    ----------
    CE : np.ndarray
        Emission count matrix to update in-place, shape (n_states, n_emissions)
    E : np.ndarray
        Current emission matrix, shape (n_states, n_emissions)
    n_clones : np.ndarray
        Number of clones per observation, shape (n_emissions,)
    mess_fwd : np.ndarray
        Forward messages, shape (T, n_states)
    mess_bwd : np.ndarray
        Backward messages, shape (T, n_states)
    x : np.ndarray
        Observation sequence, shape (T,)
    a : np.ndarray
        Action sequence, shape (T,)
    """
    timesteps = len(x)
    gamma = mess_fwd * mess_bwd
    norm = gamma.sum(1, keepdims=True)
    norm[norm == 0] = 1
    gamma /= norm
    CE[:] = 0
    for t in range(timesteps):
        CE[:, x[t]] += gamma[t]
        CE[:, x[t]] += gamma[t]
