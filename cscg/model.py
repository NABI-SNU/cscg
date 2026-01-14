"""CHMM (Clone-Structured HMM) model class."""

from __future__ import annotations

import sys

import numpy as np
from tqdm import trange

from .inference import (
    backtrace,
    backtrace_all,
    backtraceE,
    backward,
    backwardE,
    forward,
    forward_mp,
    forward_mp_all,
    forwardE,
    forwardE_mp,
)
from .learning import updateC, updateCE
from .utils import validate_seq


class CHMM:
    """Clone-Structured Hidden Markov Model (CHMM) with action-conditioned transitions.

    This is an action-augmented HMM where:
    - Each observation symbol has multiple "clones" (hidden states)
    - Each clone deterministically emits one observation symbol
    - Transitions are conditioned on actions

    Parameters
    ----------
    n_clones : np.ndarray
        Array where n_clones[i] is the number of clones assigned to observation i
    x : np.ndarray
        Observation sequence for initialization/validation, shape (T,)
    a : np.ndarray
        Action sequence for initialization/validation, shape (T,)
    pseudocount : float, default=0.0
        Pseudocount for transition matrix smoothing
    dtype : numpy dtype, default=np.float32
        Data type for parameters
    seed : int, default=42
        Random seed for initialization
    """

    def __init__(self, n_clones, x, a, pseudocount=0.0, dtype=np.float32, seed=42):
        """Construct a CHMM object."""
        np.random.seed(seed)
        self.n_clones = n_clones
        validate_seq(x, a, self.n_clones)
        assert pseudocount >= 0.0, "The pseudocount should be positive"
        print("Average number of clones:", n_clones.mean())
        self.pseudocount = pseudocount
        self.dtype = dtype
        n_states = self.n_clones.sum()
        n_actions = a.max() + 1
        self.C = np.random.rand(n_actions, n_states, n_states).astype(dtype)
        self.Pi_x = np.ones(n_states) / n_states
        self.Pi_a = np.ones(n_actions) / n_actions
        self.update_T()

    def update_T(self):
        """Update the transition matrix given the accumulated counts matrix."""
        self.T = self.C + self.pseudocount
        norm = self.T.sum(2, keepdims=True)
        norm[norm == 0] = 1
        self.T /= norm

    def update_E(self, CE):
        """Update the emission matrix from count matrix CE.

        Parameters
        ----------
        CE : np.ndarray
            Emission count matrix, shape (n_states, n_emissions)

        Returns
        -------
        E : np.ndarray
            Normalized emission matrix, shape (n_states, n_emissions)
        """
        E = CE + self.pseudocount
        norm = E.sum(1, keepdims=True)
        norm[norm == 0] = 1
        E /= norm
        return E

    def bps(self, x, a):
        """Compute bits per symbol (negative log-likelihood in base 2).

        Parameters
        ----------
        x : np.ndarray
            Observation sequence, shape (T,)
        a : np.ndarray
            Action sequence, shape (T,)

        Returns
        -------
        bps : np.ndarray
            Bits per symbol per time step, shape (T,)
        """
        validate_seq(x, a, self.n_clones)
        log2_lik = forward(self.T.transpose(0, 2, 1), self.Pi_x, self.n_clones, x, a)[0]
        return -log2_lik

    def bpsE(self, E, x, a):
        """Compute bits per symbol using an alternate emission matrix.

        Parameters
        ----------
        E : np.ndarray
            Emission matrix, shape (n_states, n_emissions)
        x : np.ndarray
            Observation sequence, shape (T,)
        a : np.ndarray
            Action sequence, shape (T,)

        Returns
        -------
        bps : np.ndarray
            Bits per symbol per time step, shape (T,)
        """
        validate_seq(x, a, self.n_clones)
        log2_lik = forwardE(self.T.transpose(0, 2, 1), E, self.Pi_x, self.n_clones, x, a)
        return -log2_lik

    def bpsV(self, x, a):
        """Compute bits per symbol using Viterbi (max-product) decoding.

        Parameters
        ----------
        x : np.ndarray
            Observation sequence, shape (T,)
        a : np.ndarray
            Action sequence, shape (T,)

        Returns
        -------
        bps : np.ndarray
            Bits per symbol per time step, shape (T,)
        """
        validate_seq(x, a, self.n_clones)
        log2_lik = forward_mp(self.T.transpose(0, 2, 1), self.Pi_x, self.n_clones, x, a)[0]
        return -log2_lik

    def decode(self, x, a):
        """Compute the MAP assignment of latent variables using max-product message passing.

        Parameters
        ----------
        x : np.ndarray
            Observation sequence, shape (T,)
        a : np.ndarray
            Action sequence, shape (T,)

        Returns
        -------
        log2_lik : np.ndarray
            Log-likelihood (base 2) per time step, shape (T,)
        states : np.ndarray
            MAP state sequence (global clone indices), shape (T,)
        """
        log2_lik, mess_fwd = forward_mp(
            self.T.transpose(0, 2, 1),
            self.Pi_x,
            self.n_clones,
            x,
            a,
            store_messages=True,
        )
        states = backtrace(self.T, self.n_clones, x, a, mess_fwd)
        return -log2_lik, states

    def decodeE(self, E, x, a):
        """Compute the MAP assignment using max-product with an alternative emission matrix.

        Parameters
        ----------
        E : np.ndarray
            Emission matrix, shape (n_states, n_emissions)
        x : np.ndarray
            Observation sequence, shape (T,)
        a : np.ndarray
            Action sequence, shape (T,)

        Returns
        -------
        log2_lik : np.ndarray
            Log-likelihood (base 2) per time step, shape (T,)
        states : np.ndarray
            MAP state sequence (global clone indices), shape (T,)
        """
        log2_lik, mess_fwd = forwardE_mp(
            self.T.transpose(0, 2, 1),
            E,
            self.Pi_x,
            self.n_clones,
            x,
            a,
            store_messages=True,
        )
        states = backtraceE(self.T, E, self.n_clones, x, a, mess_fwd)
        return -log2_lik, states

    def learn_em_T(self, x, a, n_iter=100, term_early=True):
        """Run EM training, keeping E deterministic and fixed, learning T.

        This is the core CSCG/CHMM training loop.

        Parameters
        ----------
        x : np.ndarray
            Observation sequence, shape (T,)
        a : np.ndarray
            Action sequence, shape (T,)
        n_iter : int, default=100
            Maximum number of EM iterations
        term_early : bool, default=True
            If True, terminate early when likelihood stops improving

        Returns
        -------
        convergence : list
            List of bits per symbol values per iteration
        """
        sys.stdout.flush()
        convergence = []
        pbar = trange(n_iter, position=0)
        log2_lik_old = -np.inf
        for _ in pbar:
            # E-step
            log2_lik, mess_fwd = forward(
                self.T.transpose(0, 2, 1),
                self.Pi_x,
                self.n_clones,
                x,
                a,
                store_messages=True,
            )
            mess_bwd = backward(self.T, self.n_clones, x, a)
            updateC(self.C, self.T, self.n_clones, mess_fwd, mess_bwd, x, a)
            # M-step
            self.update_T()
            convergence.append(-log2_lik.mean())
            pbar.set_postfix(train_bps=convergence[-1])
            if log2_lik.mean() <= log2_lik_old:
                if term_early:
                    break
            log2_lik_old = log2_lik.mean()
        return convergence

    def learn_viterbi_T(self, x, a, n_iter=100):
        """Run Viterbi training, keeping E deterministic and fixed, learning T.

        Uses hard assignments instead of soft posterior marginals.

        Parameters
        ----------
        x : np.ndarray
            Observation sequence, shape (T,)
        a : np.ndarray
            Action sequence, shape (T,)
        n_iter : int, default=100
            Maximum number of iterations

        Returns
        -------
        convergence : list
            List of bits per symbol values per iteration
        """
        sys.stdout.flush()
        convergence = []
        pbar = trange(n_iter, position=0)
        log2_lik_old = -np.inf
        for _ in pbar:
            # E-step (hard assignment)
            log2_lik, mess_fwd = forward_mp(
                self.T.transpose(0, 2, 1),
                self.Pi_x,
                self.n_clones,
                x,
                a,
                store_messages=True,
            )
            states = backtrace(self.T, self.n_clones, x, a, mess_fwd)
            self.C[:] = 0
            for t in range(1, len(x)):
                aij, i, j = (
                    a[t - 1],
                    states[t - 1],
                    states[t],
                )  # at time t-1 -> t we go from observation i to observation j
                self.C[aij, i, j] += 1.0
            # M-step
            self.update_T()

            convergence.append(-log2_lik.mean())
            pbar.set_postfix(train_bps=convergence[-1])
            if log2_lik.mean() <= log2_lik_old:
                break
            log2_lik_old = log2_lik.mean()
        return convergence

    def learn_em_E(self, x, a, n_iter=100, pseudocount_extra=1e-20):
        """Run EM training, keeping T fixed, learning E.

        Parameters
        ----------
        x : np.ndarray
            Observation sequence, shape (T,)
        a : np.ndarray
            Action sequence, shape (T,)
        n_iter : int, default=100
            Maximum number of EM iterations
        pseudocount_extra : float, default=1e-20
            Additional pseudocount for emission matrix

        Returns
        -------
        convergence : list
            List of bits per symbol values per iteration
        E : np.ndarray
            Learned emission matrix, shape (n_states, n_emissions)
        """
        sys.stdout.flush()
        n_emissions, n_states = len(self.n_clones), self.n_clones.sum()
        CE = np.ones((n_states, n_emissions), self.dtype)
        E = self.update_E(CE + pseudocount_extra)
        convergence = []
        pbar = trange(n_iter, position=0)
        log2_lik_old = -np.inf
        for _ in pbar:
            # E-step
            log2_lik, mess_fwd = forwardE(
                self.T.transpose(0, 2, 1),
                E,
                self.Pi_x,
                self.n_clones,
                x,
                a,
                store_messages=True,
            )
            mess_bwd = backwardE(self.T, E, self.n_clones, x, a)
            updateCE(CE, E, self.n_clones, mess_fwd, mess_bwd, x, a)
            # M-step
            E = self.update_E(CE + pseudocount_extra)
            convergence.append(-log2_lik.mean())
            pbar.set_postfix(train_bps=convergence[-1])
            if log2_lik.mean() <= log2_lik_old:
                break
            log2_lik_old = log2_lik.mean()
        return convergence, E

    def sample(self, length):
        """Sample from the CHMM.

        Parameters
        ----------
        length : int
            Length of sequence to sample

        Returns
        -------
        sample_x : np.ndarray
            Sampled observation sequence, shape (length,)
        sample_a : np.ndarray
            Sampled action sequence, shape (length,)
        """
        assert length > 0
        state_loc = np.hstack(([0], self.n_clones)).cumsum(0)
        sample_x = np.zeros(length, dtype=np.int64)
        sample_a = np.random.choice(len(self.Pi_a), size=length, p=self.Pi_a)

        # Sample
        p_h = self.Pi_x
        for t in range(length):
            h = np.random.choice(len(p_h), p=p_h)
            sample_x[t] = np.digitize(h, state_loc) - 1
            p_h = self.T[sample_a[t], h]
        return sample_x, sample_a

    def sample_sym(self, sym, length):
        """Sample from the CHMM conditioning on an initial observation.

        Parameters
        ----------
        sym : int
            Initial observation symbol
        length : int
            Length of sequence to sample (including initial symbol)

        Returns
        -------
        seq : list
            Sequence of observation symbols
        """
        # Prepare structures
        assert length > 0
        state_loc = np.hstack(([0], self.n_clones)).cumsum(0)

        seq = [sym]

        alpha = np.ones(self.n_clones[sym])
        alpha /= alpha.sum()

        for _ in range(length):
            obs_tm1 = seq[-1]
            T_weighted = self.T.sum(0)

            long_alpha = np.dot(alpha, T_weighted[state_loc[obs_tm1] : state_loc[obs_tm1 + 1], :])
            long_alpha /= long_alpha.sum()
            idx = np.random.choice(np.arange(self.n_clones.sum()), p=long_alpha)

            sym = np.digitize(idx, state_loc) - 1
            seq.append(sym)

            temp_alpha = long_alpha[state_loc[sym] : state_loc[sym + 1]]
            temp_alpha /= temp_alpha.sum()
            alpha = temp_alpha

        return seq

    def bridge(self, state1, state2, max_steps=100):
        """Find a bridging path from state1 to state2.

        Parameters
        ----------
        state1 : int
            Starting state (global clone index)
        state2 : int
            Target state (global clone index)
        max_steps : int, default=100
            Maximum number of steps to search

        Returns
        -------
        actions : np.ndarray
            Action sequence for bridging path
        states : np.ndarray
            State sequence for bridging path
        """
        Pi_x = np.zeros(self.n_clones.sum(), dtype=self.dtype)
        Pi_x[state1] = 1
        log2_lik, mess_fwd = forward_mp_all(
            self.T.transpose(0, 2, 1), Pi_x, self.Pi_a, self.n_clones, state2, max_steps
        )
        s_a = backtrace_all(self.T, self.Pi_a, self.n_clones, mess_fwd, state2)
        return s_a
