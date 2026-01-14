# Clone-Structured Cognitive Graphs

[![DOI](https://zenodo.org/badge/344697858.svg)](https://zenodo.org/badge/latestdoi/344697858)

Code for ["Learning cognitive maps as structured graphs for vicarious evaluation"](https://www.biorxiv.org/content/10.1101/864421v4.full)

## Implementation-Oriented Documentation

This document describes the **clone-structured cognitive graph (CSCG)** as implemented by the provided code. Conceptually, this implementation is an **action-augmented cloned HMM** (often called a **CHMM** in code): an HMM whose emission model is *structurally constrained* so that **each hidden state (“clone”) deterministically emits exactly one observation symbol**, and transitions are **conditioned on actions**.

> In the CSCG formulation, the joint density over an observation–action trajectory is
>
> $$ P(x_{1:N}, a_{1:N-1}) = \sum_{z_1 \in C(x_1)} \cdots \sum_{z_N \in C(x_N)} P(z_1)\prod_{n=1}^{N-1} P(z_{n+1}, a_n \mid z_n). $$
>
> where $C(j)$ denotes the set of clones that emit observation $j$.

---

### 1. Notation and data model

#### Observations and actions

- Observations: $x_j \in \{0,1,\dots,E-1\}$ (integers)
- Actions: $a_t \in \{0,1,\dots,A-1\}$ (integers)

The code expects **paired sequences**:

- `x`: shape `(T,)`, dtype `np.int64`
- `a`: shape `(T,)`, dtype `np.int64` (note: in the code, `a[t-1]` is used for the transition $t-1 \to t$; the last action can be unused but still present)

#### Clones (hidden states)

Let there be $E$ observation symbols. Each symbol $j$ has `n_clones[j] = K_j` hidden states (“clones”) assigned to it. Thus, the total number of hidden states is:
> $$ H = \sum_{j=0}^{E-1} K_j. $$

Define the clone index ranges via a prefix-sum:

- `state_loc = np.cumsum([0] + n_clones.tolist())` so that clones for observation $j$ are:

> $$ C(j) = \{\, \mathrm{state\_loc}[j] , \mathrm{state\_loc}[j] + 1, \cdots, \mathrm{state\_loc}[j+1]-1 \,\}. $$

---

### 2. Parameters (what the implementation actually learns)

#### 2.1 Transition tensor (action-conditioned)

The implementation maintains:

- Count tensor: `C[a, i, j]` (float)
- Transition tensor: `T[a, i, j]` (float), normalized row-wise over `j`

Interpretation in this implementation: $T[a, i, j] \approx P(z_{t}=j \mid z_{t-1}=i,\ a_{t-1}=a)$.

This corresponds to an **action-conditioned transition model**. (The paper also discusses a variant that models $P(z_{t+1}, a_t \mid z_t)$; the code contains a commented-out alternative normalization, but the active implementation is the action-conditioned version.)

#### 2.2 Initial distribution

- `Pi_x`: shape `(H,)`, initial distribution over clones $P(z_1)$

In the provided class, it is initialized uniform over all clones.

#### 2.3 Emission model (deterministic-by-structure)

There are two modes in the code:

1. **Deterministic emission via clone structure** (default training for `learn_em_T`):
   The forward/backward routines in `forward` / `backward` assume that at time $t$, the hidden state must lie in $C(x_t)$. Therefore, no dense emission matrix is needed, and inference only tracks probabilities over the clones consistent with the current observation.

2. **Learned emission matrix `E`** (optional; used by `learn_em_E`, `forwardE`, etc.):
   - `E`: shape `(H, E)`
   - row-normalized so each hidden state has a distribution over observation symbols:
$$ E[i, j] = P(x_t=j \mid z_t=i), \quad \sum_{j} E[i,j]=1. $$
   This is used for transfer / relabeling experiments and more general observation noise models.

---

### 3. Inference algorithms

The implementation uses **message passing** (forward–backward) with **per-step normalization** to avoid underflow. It stores **log-likelihood in base 2** using the per-step normalization constants.

#### 3.1 Sum-product forward pass (filtering)

For the deterministic-emission (clone-structured) model, define the forward message at time $t$ over clones in $C(x_t)$:

$$ \alpha_t(j) \propto P(z_t=j \mid x_{1:t}, a_{1:t-1}), \quad j \in C(x_t). $$

> **Initialization** (restrict to clones of $x_1$):
> $$ \alpha_1(j) = \frac{\Pi(j)}{\sum_{k\in C(x_1)} \Pi(k)},\quad j\in C(x_1). $$
>

Next,

> **Recursion** for $t\ge 2$, with $a_{t-1}$:
> $$ \tilde{\alpha}_t(j) = \sum_{i\in C(x_{t-1})} T[a_{t-1}, i, j]\ \alpha_{t-1}(i), \quad j\in C(x_t), $$
> then normalize:
> $$ \alpha_t(j) = \frac{\tilde{\alpha}_t(j)}{\sum_{k\in C(x_t)} \tilde{\alpha}_t(k)}. $$

The code computes the normalization constant $p_t = \sum \tilde{\alpha}_t$ and stores $\log_2(p_t)$ per step.

**Implementation detail:** `forward()` receives `T_tr = T.transpose(0,2,1)` so it can compute a matrix–vector product with contiguous blocks efficiently (Numba-friendly).

#### 3.2 Sum-product backward pass (smoothing support)

Backward message at time $t$ over clones in $C(x_t)$:

$$
\beta_t(i) \propto P(x_{t+1:T} \mid z_t=i, a_{t:T-1}).
$$

Recursion (restricted to clone blocks):

$$
\tilde{\beta}_t(i) = \sum_{j\in C(x_{t+1})} T[a_t, i, j]\, \beta_{t+1}(j),
\quad i \in C(x_t),
$$
then normalize by $\sum_i \tilde{\beta}_t(i)$ (the code normalizes each step).

#### 3.3 Posterior marginals

Given stored forward/backward messages (already normalized), the code forms:

$$
\gamma_t(i) \propto \alpha_t(i)\beta_t(i),
$$
followed by normalization across the clone block for that $t$.

This is used to update counts.

#### 3.4 Max-product (Viterbi-style) decoding

The functions `forward_mp` and `backtrace` compute the MAP path:

- Replace sums with maxima:
  $$
  \tilde{\delta}_t(j) = \max_{i\in C(x_{t-1})} T[a_{t-1}, i, j]\ \delta_{t-1}(i)
  $$
- Normalize by dividing by `max()` per step (for stability)
- `backtrace()` chooses argmax *with random tie-breaking* via `rargmax()` to reduce systematic bias.

Returned `states[t]` are **global clone indices** in `0..H-1`.

---

### 4. Learning (EM for transitions and/or emissions)

#### 4.1 EM for transitions `T` with deterministic emissions (`learn_em_T`)

This is the core CSCG/CHMM training loop in the provided code.

**E-step:** compute expected transition counts:

For each time step $t = 2..T$, action $a_{t-1}$, previous observation $x_{t-1}=i$, current observation $x_t=j$:

- Let $I=C(i)$, $J=C(j)$.
- Compute local pairwise posterior over clone-to-clone transitions:

$$
q_{t-1}(u,v)
=
\frac{
\alpha_{t-1}(u)\ T[a_{t-1},u,v]\ \beta_t(v)
}{
\sum_{u'\in I}\sum_{v'\in J}
\alpha_{t-1}(u')\ T[a_{t-1},u',v']\ \beta_t(v')
},
\quad (u\in I, v\in J)
$$

Then accumulate expected counts:

$$
C[a_{t-1}, u, v] \leftarrow \sum_{t} q_{t-1}(u,v).
$$

This is implemented in `updateC()` using contiguous block slices of `mess_fwd` and `mess_bwd`.

**M-step:** normalize with pseudocount
