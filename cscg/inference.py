"""Inference algorithms for CSCG: forward-backward, Viterbi, and variants."""

from __future__ import annotations

import numba as nb
import numpy as np

from .utils import rargmax


@nb.njit
def forward(T_tr, Pi, n_clones, x, a, store_messages=False):
    """Sum-product forward pass (filtering) for clone-structured HMM.

    Computes log-probability of a sequence (in base 2) and optionally stores
    forward messages for use in backward pass or EM updates.

    Parameters
    ----------
    T_tr : np.ndarray
        Transposed transition tensor, shape (n_actions, n_states, n_states)
        where T_tr[a, j, i] = T[a, i, j] for efficient matrix-vector products
    Pi : np.ndarray
        Initial probability distribution over clones, shape (n_states,)
    n_clones : np.ndarray
        Number of clones per observation, shape (n_emissions,)
    x : np.ndarray
        Observation sequence, shape (T,) of integers
    a : np.ndarray
        Action sequence, shape (T,) of integers
    store_messages : bool, default=False
        If True, return forward messages for use in backward pass

    Returns
    -------
    log2_lik : np.ndarray
        Log-likelihood (base 2) per time step, shape (T,)
    mess_fwd : np.ndarray or None
        Forward messages (shape (sum(n_clones[x]),)) if store_messages=True, else None
    """
    state_loc = np.concatenate((np.array([0], dtype=n_clones.dtype), n_clones)).cumsum()
    dtype = T_tr.dtype.type

    # forward pass
    t = 0
    log2_lik = np.zeros(len(x), dtype)
    j = x[t]
    j_start, j_stop = state_loc[j], state_loc[j + 1]
    message = Pi[j_start:j_stop].copy().astype(dtype)

    # normalize message
    p_obs = message.sum()
    assert p_obs > 0
    message /= p_obs

    log2_lik[0] = np.log2(p_obs)  # log-likelihood of the first time step

    if store_messages:
        mess_loc = np.concatenate((np.array([0], dtype=n_clones.dtype), n_clones[x])).cumsum()
        mess_fwd_arr = np.empty(mess_loc[-1], dtype=dtype)
        t_start, t_stop = mess_loc[t], mess_loc[t + 1]
        mess_fwd_arr[t_start:t_stop] = message

    for t in range(1, x.shape[0]):
        aij, i, j = (
            a[t - 1],
            x[t - 1],
            x[t],
        )  # at time t-1 -> t we go from observation i to observation j
        (i_start, i_stop), (j_start, j_stop) = (
            state_loc[i : i + 2],
            state_loc[j : j + 2],
        )
        message = np.ascontiguousarray(T_tr[aij, j_start:j_stop, i_start:i_stop]).dot(message)
        p_obs = message.sum()
        assert p_obs > 0
        message /= p_obs
        log2_lik[t] = np.log2(p_obs)

        if store_messages:
            t_start, t_stop = mess_loc[t : t + 2]
            mess_fwd_arr[t_start:t_stop] = message

    return log2_lik, mess_fwd_arr if store_messages else None


@nb.njit
def backward(T, n_clones, x, a):
    """Sum-product backward pass (smoothing) for clone-structured HMM.

    Computes backward messages for use in EM updates or posterior marginals.

    Parameters
    ----------
    T : np.ndarray
        Transition tensor, shape (n_actions, n_states, n_states)
    n_clones : np.ndarray
        Number of clones per observation, shape (n_emissions,)
    x : np.ndarray
        Observation sequence, shape (T,)
    a : np.ndarray
        Action sequence, shape (T,)

    Returns
    -------
    mess_bwd : np.ndarray
        Backward messages, shape (sum(n_clones[x]),)
    """
    state_loc = np.concatenate((np.array([0], dtype=n_clones.dtype), n_clones)).cumsum()
    dtype = T.dtype.type

    # backward pass
    t = x.shape[0] - 1
    i = x[t]
    message = np.ones(n_clones[i], dtype) / n_clones[i]
    message /= message.sum()
    mess_loc = np.concatenate((np.array([0], dtype=n_clones.dtype), n_clones[x])).cumsum()
    mess_bwd = np.empty(mess_loc[-1], dtype)
    t_start, t_stop = mess_loc[t], mess_loc[t + 1]
    mess_bwd[t_start:t_stop] = message
    for t in range(x.shape[0] - 2, -1, -1):
        aij, i, j = (
            a[t],
            x[t],
            x[t + 1],
        )  # at time t -> t+1 we go from observation i to observation j
        (i_start, i_stop), (j_start, j_stop) = (
            state_loc[i : i + 2],
            state_loc[j : j + 2],
        )
        message = np.ascontiguousarray(T[aij, i_start:i_stop, j_start:j_stop]).dot(message)
        p_obs = message.sum()
        assert p_obs > 0
        message /= p_obs
        t_start, t_stop = mess_loc[t], mess_loc[t + 1]
        mess_bwd[t_start:t_stop] = message
    return mess_bwd


@nb.njit
def forward_mp(T_tr, Pi, n_clones, x, a, store_messages=False):
    """Max-product forward pass (Viterbi-style) for clone-structured HMM.

    Computes MAP log-probability using max-product message passing.

    Parameters
    ----------
    T_tr : np.ndarray
        Transposed transition tensor, shape (n_actions, n_states, n_states)
        where T_tr[a, j, i] = T[a, i, j] for efficient matrix-vector products
    Pi : np.ndarray
        Initial distribution over clones, shape (n_states,)
    n_clones : np.ndarray
        Number of clones per observation, shape (n_emissions,)
    x : np.ndarray
        Observation sequence, shape (T,) of integers
    a : np.ndarray
        Action sequence, shape (T,) of integers
    store_messages : bool, default=False
        If True, return forward messages for use in backtrace

    Returns
    -------
    log2_lik : np.ndarray
        Log-likelihood (base 2) per time step, shape (T,)
    mess_fwd : np.ndarray or None
        Forward messages (shape (sum(n_clones[x]),)) if store_messages=True, else None
    """
    state_loc = np.concatenate((np.array([0], dtype=n_clones.dtype), n_clones)).cumsum()
    dtype = T_tr.dtype.type

    # forward pass
    t = 0
    log2_lik = np.zeros(len(x), dtype)
    j = x[t]
    j_start, j_stop = state_loc[j], state_loc[j + 1]
    message = Pi[j_start:j_stop].copy().astype(dtype)
    p_obs = message.max()
    assert p_obs > 0
    message /= p_obs
    log2_lik[0] = np.log2(p_obs)  # log-likelihood of the first time step
    if store_messages:
        mess_loc = np.concatenate((np.array([0], dtype=n_clones.dtype), n_clones[x])).cumsum()
        mess_fwd_arr = np.empty(mess_loc[-1], dtype=dtype)
        t_start, t_stop = mess_loc[t : t + 2]
        mess_fwd_arr[t_start:t_stop] = message

    for t in range(1, x.shape[0]):
        aij, i, j = (
            a[t - 1],
            x[t - 1],
            x[t],
        )  # at time t-1 -> t we go from observation i to observation j
        (i_start, i_stop), (j_start, j_stop) = (
            state_loc[i : i + 2],
            state_loc[j : j + 2],
        )
        new_message = np.zeros(j_stop - j_start, dtype=dtype)
        for d in range(len(new_message)):
            new_message[d] = (T_tr[aij, j_start + d, i_start:i_stop] * message).max()
        message = new_message
        p_obs = message.max()
        assert p_obs > 0
        message /= p_obs
        log2_lik[t] = np.log2(p_obs)
        if store_messages:
            t_start, t_stop = mess_loc[t : t + 2]
            mess_fwd_arr[t_start:t_stop] = message

    return log2_lik, mess_fwd_arr if store_messages else None


@nb.njit
def backtrace(T, n_clones, x, a, mess_fwd):
    """Backtrace for max-product decoding (Viterbi path).

    Computes the MAP assignment of latent states given forward messages.

    Parameters
    ----------
    T : np.ndarray
        Transition tensor, shape (n_actions, n_states, n_states)
    n_clones : np.ndarray
        Number of clones per observation, shape (n_emissions,)
    x : np.ndarray
        Observation sequence, shape (T,) of integers
    a : np.ndarray
        Action sequence, shape (T,)
    mess_fwd : np.ndarray
        Forward messages from forward_mp, shape (sum(n_clones[x]),)

    Returns
    -------
    states : np.ndarray
        MAP state sequence (global clone indices), shape (T,)
    """
    state_loc = np.hstack((np.array([0], dtype=n_clones.dtype), n_clones)).cumsum()
    mess_loc = np.hstack((np.array([0], dtype=n_clones.dtype), n_clones[x])).cumsum()
    code = np.zeros(x.shape[0], dtype=np.int64)

    # backward pass
    t = x.shape[0] - 1
    i = x[t]
    t_start, t_stop = mess_loc[t : t + 2]
    belief = mess_fwd[t_start:t_stop]
    code[t] = rargmax(belief)
    for t in range(x.shape[0] - 2, -1, -1):
        aij, i, j = (
            a[t],
            x[t],
            x[t + 1],
        )  # at time t -> t+1 we go from observation i to observation j
        (i_start, i_stop), j_start = state_loc[i : i + 2], state_loc[j]
        t_start, t_stop = mess_loc[t : t + 2]
        belief = mess_fwd[t_start:t_stop] * T[aij, i_start:i_stop, j_start + code[t + 1]]
        code[t] = rargmax(belief)
    states = state_loc[x] + code
    return states


def forwardE(T_tr, E, Pi, n_clones, x, a, store_messages=False):
    """Sum-product forward pass with learned emission matrix E.

    Variant of forward() that uses a learned emission matrix instead of
    deterministic clone-structured emissions.

    Parameters
    ----------
    T_tr : np.ndarray
        Transposed transition tensor, shape (n_actions, n_states, n_states)
    E : np.ndarray
        Emission matrix, shape (n_states, n_emissions)
    Pi : np.ndarray
        Initial distribution over clones, shape (n_states,)
    n_clones : np.ndarray
        Number of clones per observation, shape (n_emissions,)
    x : np.ndarray
        Observation sequence, shape (T,)
    a : np.ndarray
        Action sequence, shape (T,)
    store_messages : bool, default=False
        If True, return forward messages

    Returns
    -------
    log2_lik : np.ndarray or tuple
        If store_messages=False: log-likelihood array, shape (T,)
        If store_messages=True: (log2_lik, mess_fwd) tuple
    """
    assert (n_clones.sum(), len(n_clones)) == E.shape
    dtype = T_tr.dtype.type

    # forward pass
    t, log2_lik = 0, np.zeros(len(x), dtype)
    j = x[t]
    message = Pi * E[:, j]
    p_obs = message.sum()
    assert p_obs > 0
    message /= p_obs
    log2_lik[0] = np.log2(p_obs)
    if store_messages:
        mess_fwd = np.empty((len(x), E.shape[0]), dtype=dtype)
        mess_fwd[t] = message
    for t in range(1, x.shape[0]):
        aij, j = (
            a[t - 1],
            x[t],
        )  # at time t-1 -> t we go from observation i to observation j
        message = T_tr[aij].dot(message)
        message *= E[:, j]
        p_obs = message.sum()
        assert p_obs > 0
        message /= p_obs
        log2_lik[t] = np.log2(p_obs)
        if store_messages:
            mess_fwd[t] = message
    if store_messages:
        return log2_lik, mess_fwd
    else:
        return log2_lik


def backwardE(T, E, n_clones, x, a):
    """Sum-product backward pass with learned emission matrix E.

    Parameters
    ----------
    T : np.ndarray
        Transition tensor, shape (n_actions, n_states, n_states)
    E : np.ndarray
        Emission matrix, shape (n_states, n_emissions)
    n_clones : np.ndarray
        Number of clones per observation, shape (n_emissions,)
    x : np.ndarray
        Observation sequence, shape (T,)
    a : np.ndarray
        Action sequence, shape (T,)

    Returns
    -------
    mess_bwd : np.ndarray
        Backward messages, shape (T, n_states)
    """
    assert (n_clones.sum(), len(n_clones)) == E.shape
    dtype = T.dtype.type

    # backward pass
    t = x.shape[0] - 1
    message = np.ones(E.shape[0], dtype)
    message /= message.sum()
    mess_bwd = np.empty((len(x), E.shape[0]), dtype=dtype)
    mess_bwd[t] = message
    for t in range(x.shape[0] - 2, -1, -1):
        aij, j = (
            a[t],
            x[t + 1],
        )  # at time t -> t+1 we go from observation i to observation j
        message = T[aij].dot(message * E[:, j])
        p_obs = message.sum()
        assert p_obs > 0
        message /= p_obs
        mess_bwd[t] = message
    return mess_bwd


def forwardE_mp(T_tr, E, Pi, n_clones, x, a, store_messages=False):
    """Max-product forward pass with learned emission matrix E.

    Parameters
    ----------
    T_tr : np.ndarray
        Transposed transition tensor, shape (n_actions, n_states, n_states)
    E : np.ndarray
        Emission matrix, shape (n_states, n_emissions)
    Pi : np.ndarray
        Initial distribution over clones, shape (n_states,)
    n_clones : np.ndarray
        Number of clones per observation, shape (n_emissions,)
    x : np.ndarray
        Observation sequence, shape (T,)
    a : np.ndarray
        Action sequence, shape (T,)
    store_messages : bool, default=False
        If True, return forward messages

    Returns
    -------
    log2_lik : np.ndarray or tuple
        If store_messages=False: log-likelihood array, shape (T,)
        If store_messages=True: (log2_lik, mess_fwd) tuple
    """
    assert (n_clones.sum(), len(n_clones)) == E.shape
    dtype = T_tr.dtype.type

    # forward pass
    t, log2_lik = 0, np.zeros(len(x), dtype)
    j = x[t]
    message = Pi * E[:, j]
    p_obs = message.max()
    assert p_obs > 0
    message /= p_obs
    log2_lik[0] = np.log2(p_obs)
    if store_messages:
        mess_fwd = np.empty((len(x), E.shape[0]), dtype=dtype)
        mess_fwd[t] = message
    for t in range(1, x.shape[0]):
        aij, j = (
            a[t - 1],
            x[t],
        )  # at time t-1 -> t we go from observation i to observation j
        message = (T_tr[aij] * message.reshape(1, -1)).max(1)
        message *= E[:, j]
        p_obs = message.max()
        assert p_obs > 0
        message /= p_obs
        log2_lik[t] = np.log2(p_obs)
        if store_messages:
            mess_fwd[t] = message
    if store_messages:
        return log2_lik, mess_fwd
    else:
        return log2_lik


def backtraceE(T, E, n_clones, x, a, mess_fwd):
    """Backtrace for max-product decoding with learned emission matrix E.

    Parameters
    ----------
    T : np.ndarray
        Transition tensor, shape (n_actions, n_states, n_states)
    E : np.ndarray
        Emission matrix, shape (n_states, n_emissions)
    n_clones : np.ndarray
        Number of clones per observation, shape (n_emissions,)
    x : np.ndarray
        Observation sequence, shape (T,)
    a : np.ndarray
        Action sequence, shape (T,)
    mess_fwd : np.ndarray
        Forward messages from forwardE_mp, shape (T, n_states)

    Returns
    -------
    states : np.ndarray
        MAP state sequence (global clone indices), shape (T,)
    """
    assert (n_clones.sum(), len(n_clones)) == E.shape
    states = np.zeros(x.shape[0], dtype=np.int64)

    # backward pass
    t = x.shape[0] - 1
    belief = mess_fwd[t]
    states[t] = rargmax(belief)
    for t in range(x.shape[0] - 2, -1, -1):
        aij = a[t]  # at time t -> t+1 we go from observation i to observation j
        belief = mess_fwd[t] * T[aij, :, states[t + 1]]
        states[t] = rargmax(belief)
    return states


def forward_mp_all(T_tr, Pi_x, Pi_a, n_clones, target_state, max_steps):
    """Max-product forward pass for bridging between states.

    Used to find a path from an initial state distribution to a target state.

    Parameters
    ----------
    T_tr : np.ndarray
        Transposed transition tensor, shape (n_actions, n_states, n_states)
    Pi_x : np.ndarray
        Initial distribution over states, shape (n_states,)
    Pi_a : np.ndarray
        Action prior distribution, shape (n_actions,)
    n_clones : np.ndarray
        Number of clones per observation, shape (n_emissions,)
    target_state : int
        Target state index to reach
    max_steps : int
        Maximum number of steps to search

    Returns
    -------
    log2_lik : np.ndarray
        Log-likelihood per step, shape (n_steps,)
    mess_fwd : np.ndarray
        Forward messages, shape (n_steps, n_states)
    """
    # forward pass
    log2_lik = []
    message = Pi_x
    p_obs = message.max()
    assert p_obs > 0
    message /= p_obs
    log2_lik.append(np.log2(p_obs))
    mess_fwd = []
    mess_fwd.append(message)
    T_tr_maxa = (T_tr * Pi_a.reshape(-1, 1, 1)).max(0)
    for _ in range(1, max_steps):
        message = (T_tr_maxa * message.reshape(1, -1)).max(1)
        p_obs = message.max()
        assert p_obs > 0
        message /= p_obs
        log2_lik.append(np.log2(p_obs))
        mess_fwd.append(message)
        if message[target_state] > 0:
            break
    else:
        raise ValueError("Unable to find a bridging path")
    return np.array(log2_lik), np.array(mess_fwd)


def backtrace_all(T, Pi_a, n_clones, mess_fwd, target_state):
    """Backtrace for bridging path finding.

    Parameters
    ----------
    T : np.ndarray
        Transition tensor, shape (n_actions, n_states, n_states)
    Pi_a : np.ndarray
        Action prior distribution, shape (n_actions,)
    n_clones : np.ndarray
        Number of clones per observation, shape (n_emissions,)
    mess_fwd : np.ndarray
        Forward messages from forward_mp_all, shape (n_steps, n_states)
    target_state : int
        Target state index

    Returns
    -------
    actions : np.ndarray
        Action sequence, shape (n_steps,)
    states : np.ndarray
        State sequence, shape (n_steps,)
    """
    states = np.zeros(mess_fwd.shape[0], dtype=np.int64)
    actions = np.zeros(mess_fwd.shape[0], dtype=np.int64)
    n_states = T.shape[1]
    # backward pass
    t = mess_fwd.shape[0] - 1
    actions[t], states[t] = (
        -1,
        target_state,
    )  # last actions is irrelevant, use an invalid value
    for t in range(mess_fwd.shape[0] - 2, -1, -1):
        belief = mess_fwd[t].reshape(1, -1) * T[:, :, states[t + 1]] * Pi_a.reshape(-1, 1)
        a_s = rargmax(belief.flatten())
        actions[t], states[t] = a_s // n_states, a_s % n_states
    return actions, states
