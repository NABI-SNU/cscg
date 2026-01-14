# CSCG Project - Agent Documentation

This document provides context for AI coding assistants working on the Clone-Structured Cognitive Graphs (CSCG) project.

## Project Overview

CSCG is a Python package implementing **Clone-Structured Cognitive Graphs** - an action-augmented cloned Hidden Markov Model (CHMM) that models hippocampal place cell representations. The project's **primary goal** is to create an **interactive map that compares biological hippocampus data to an artificial CSCG hippocampus model**.

**Paper**: ["Learning cognitive maps as structured graphs for vicarious evaluation"](https://www.biorxiv.org/content/10.1101/864421v4.full)

### Core Concept

The CSCG models cognitive maps where:

- Each observation symbol has multiple "clones" (hidden states) - analogous to place cells
- Each clone deterministically emits exactly one observation symbol
- Transitions are conditioned on actions (spatial navigation)
- The learned transition graph represents a cognitive map
- Place fields can be computed from forward messages, similar to hippocampal place cell firing patterns

### End Goal: Interactive Map

**The ultimate deliverable is an interactive visualization that:**

- Displays place fields from the CSCG model (artificial hippocampus)
- Compares them side-by-side with biological hippocampus data
- Allows interactive exploration of the learned cognitive maps
- Visualizes the transition graphs learned by the model
- Shows how CSCG place fields correspond to biological place cell data

This interactive map should be the primary output that demonstrates how the artificial CSCG hippocampus compares to biological hippocampal representations.

## Project Structure

```text
cscg/
├── cscg/              # Main package source code
│   ├── __init__.py    # Package exports and version
│   ├── model.py       # CHMM main model class
│   ├── inference.py   # Forward/backward/Viterbi algorithms
│   ├── learning.py    # EM update functions
│   ├── datagen.py     # Synthetic data generation utilities
│   └── utils.py       # Validation and helper functions
├── notebooks/         # Jupyter notebooks for analysis/figures
│   ├── intro.ipynb    # Main tutorial with place field examples
│   ├── ext_data_fig_8_*.ipynb  # Biological data comparison notebooks
│   └── figures/       # Generated visualization outputs
├── conda/             # Conda environment configuration
├── .github/workflows/ # CI/CD workflows
├── pyproject.toml     # Package configuration and dependencies
├── requirements.txt   # Basic dependencies
├── .pre-commit-config.yaml  # Pre-commit hooks configuration
└── README.md          # Detailed mathematical documentation
```

## ⚠️ CRITICAL: Pre-commit Requirements

**ALWAYS run pre-commit before committing code changes!**

### Setup Pre-commit (if not already done)

```bash
pip install pre-commit
pre-commit install
```

### Run Pre-commit

```bash
# Run on all files (recommended before committing)
pre-commit run --all-files

# Or let it run automatically on git commit (if installed)
git commit -m "your message"
```

### Pre-commit Hooks Configured

- **isort**: Import sorting (with black profile)
- **ruff**: Linting and auto-fixing code issues
- **mypy**: Type checking (relaxed for numba compatibility)
- **nbstripout**: Strips notebook outputs
- **Standard hooks**: AST checking, YAML/TOML validation, trailing whitespace removal

**The GitHub Actions workflow (`.github/workflows/pre-commit.yml`) automatically runs pre-commit on all pushes and pull requests. All code must pass pre-commit checks.**

## Key Components

### Main Model Class: `CHMM`

Located in `cscg/model.py`. The primary interface for the clone-structured HMM.

**Key attributes:**

- `n_clones`: Array specifying number of clones per observation symbol
- `C`: Count tensor `C[a, i, j]` for transitions
- `T`: Transition tensor `T[a, i, j]` (normalized, action-conditioned)
- `Pi_x`: Initial distribution over clones
- `E`: Optional emission matrix (for learned emissions mode)

**Key methods:**

- `learn_em_T()`: EM training with deterministic emissions (main training method)
- `learn_em_E()`: EM training with learned emissions
- `learn_viterbi_T()`: Viterbi-based training (refinement)
- `forward()` / `backward()`: Inference with deterministic emissions
- `forwardE()` / `backwardE()`: Inference with learned emissions
- `decode()`: Viterbi-style MAP decoding (returns most likely clone sequence)

### Inference Functions (`cscg/inference.py`)

**Deterministic emission mode:**

- `forward()`: Forward pass (filtering) with per-step normalization
- `backward()`: Backward pass for smoothing
- `forward_mp()`: Max-product forward pass (Viterbi)
- `backtrace()`: Backtrace for Viterbi decoding

**Learned emission mode:**

- `forwardE()`: Forward pass with emission matrix
- `backwardE()`: Backward pass with emission matrix
- `forwardE_mp()`: Max-product with emissions
- `backtraceE()`: Backtrace with emissions

**Batch variants:**

- `forward_mp_all()`: Batch max-product
- `backtrace_all()`: Batch backtrace

### Learning Functions (`cscg/learning.py`)

- `updateC()`: Update transition counts from forward/backward messages
- `updateCE()`: Update counts with learned emissions

### Data Generation (`cscg/datagen.py`)

- `datagen_structured_obs_room()`: Generate synthetic room navigation data
  - Returns: `(actions, observations, row_col_positions)`
  - Useful for creating training data that mimics spatial navigation

### Utilities (`cscg/utils.py`)

- `validate_seq()`: Validate observation/action sequences
- `rargmax()`: Random argmax for tie-breaking in Viterbi

## Data Format

**Observations and Actions:**

- `x`: shape `(T,)`, dtype `np.int64` - observation sequence
- `a`: shape `(T,)`, dtype `np.int64` - action sequence
- Note: `a[t-1]` is used for transition `t-1 → t`

**Clone Structure:**

- `n_clones`: shape `(E,)`, dtype `np.int64` - number of clones per observation
- Total hidden states: `H = sum(n_clones)`
- Clone ranges: `state_loc = np.cumsum([0] + n_clones.tolist())`
- Clones for observation `j`: `C(j) = {state_loc[j], ..., state_loc[j+1]-1}`

**Spatial Data:**

- `rc`: shape `(T, 2)` - row/column positions for place field computation
- Generated by `datagen_structured_obs_room()` or from biological tracking data

## Place Fields and Visualization

### Computing Place Fields

Place fields (analogous to hippocampal place cell firing patterns) are computed from forward messages:

```python
from cscg import CHMM, forwardE, get_mess_fwd  # get_mess_fwd from notebooks

# After training
mess_fwd = get_mess_fwd(chmm, x, pseudocount_E=0.1)

# Compute place field for a specific clone
def place_field(mess_fwd, rc, clone):
    """Compute place field for a clone given forward messages and positions."""
    field = np.zeros(rc.max(0) + 1)
    count = np.zeros(rc.max(0) + 1, int)
    for t in range(mess_fwd.shape[0]):
        r, c = rc[t]
        field[r, c] += mess_fwd[t, clone]
        count[r, c] += 1
    count[count == 0] = 1
    return field / count
```

### Graph Visualization

The learned transition graph can be visualized using igraph (see `notebooks/intro.ipynb`):

```python
def plot_graph(chmm, x, a, output_file, cmap, vertex_size=30):
    """Plot the learned cognitive map as a graph."""
    states = chmm.decode(x, a)[1]
    # ... graph construction and plotting
```

## Coding Conventions

### Code Style

- **Line length**: 100 characters
- **Python version**: 3.8+ (supports 3.8-3.12)
- **Type hints**: Used but not strictly enforced (`disallow_untyped_defs = false` in mypy)
- **Formatting**: Black with `--profile=black` for isort
- **Linting**: Ruff (E, W, F, I, B, C4, UP rules)
- **Type checking**: MyPy (with relaxed settings for numba compatibility)

### Pre-commit Workflow

**MANDATORY STEPS:**

1. Make code changes
2. Run `pre-commit run --all-files` to check/fix issues
3. Fix any remaining issues manually
4. Commit only after all pre-commit checks pass

**Pre-commit will:**

- Auto-fix import sorting (isort)
- Auto-fix many linting issues (ruff)
- Check types (mypy)
- Strip notebook outputs (nbstripout)
- Validate YAML/TOML files

### Dependencies

**Core dependencies:**

- `numpy>=1.20.0`
- `numba>=0.56.0` (for JIT compilation)
- `tqdm>=4.60.0`

**Development dependencies** (in `pyproject.toml`):

- `pytest>=7.0.0`
- `black>=23.0.0`
- `ruff>=0.1.0`
- `mypy>=1.0.0`
- `pre-commit>=3.0.0`
- `isort>=5.13.2`

**Notebook/Visualization dependencies:**

- `jupyter>=1.0.0`
- `matplotlib>=3.5.0`
- `scipy>=1.7.0`
- `igraph` (for graph visualization)

### Numba Usage

Many inference functions are JIT-compiled with Numba for performance:

- Use `@nb.njit` decorator
- Functions should be Numba-compatible (no Python objects, limited NumPy features)
- MyPy ignores numba imports: `ignore_missing_imports = true` for `numba.*`

## Mathematical Notation

The implementation follows this notation:

- Observations: `x_j ∈ {0,1,...,E-1}`
- Actions: `a_t ∈ {0,1,...,A-1}`
- Hidden states (clones): `z_t ∈ {0,1,...,H-1}`
- Transition: `T[a, i, j] ≈ P(z_t=j | z_{t-1}=i, a_{t-1}=a)`
- Forward message: `α_t(j) ∝ P(z_t=j | x_{1:t}, a_{1:t-1})`
- Backward message: `β_t(i) ∝ P(x_{t+1:T} | z_t=i, a_{t:T-1})`

## Common Tasks

### Adding a new inference function

1. Add function to `cscg/inference.py`
2. Use `@nb.njit` if performance-critical
3. Follow existing patterns for message passing
4. Export in `cscg/__init__.py`
5. **Run pre-commit**: `pre-commit run --all-files`
6. Add tests if applicable

### Modifying the model

1. Update `CHMM` class in `cscg/model.py`
2. Ensure backward compatibility or document breaking changes
3. Update `README.md` if mathematical formulation changes
4. **Run pre-commit**: `pre-commit run --all-files`

### Working on the Interactive Map

1. Review `notebooks/intro.ipynb` for place field computation examples
2. Review `notebooks/ext_data_fig_8_*.ipynb` for biological data comparison patterns
3. Consider using interactive visualization libraries (plotly, bokeh, etc.)
4. Ensure the map allows:
   - Side-by-side comparison of CSCG vs biological place fields
   - Interactive selection of clones/place cells
   - Overlay of learned transition graphs
   - Time-series visualization of place field development
5. **Run pre-commit**: `pre-commit run --all-files` (especially for notebooks)

### Adding dependencies

1. Add to `pyproject.toml` under appropriate section:
   - `dependencies` for core deps
   - `[project.optional-dependencies.dev]` for dev tools
   - `[project.optional-dependencies.notebooks]` for notebook deps
2. Update `requirements.txt` if needed (for basic installs)
3. Update conda `env.yml` if using conda
4. **Run pre-commit**: `pre-commit run --all-files`

### Working with notebooks

- Notebooks are in `notebooks/` directory
- `nbstripout` hook strips outputs on commit
- Use `jupytext` for version control if needed
- **Always run pre-commit** before committing notebooks

## Important Notes

1. **Deterministic emissions**: Default mode assumes clones deterministically emit observations. No emission matrix needed.

2. **Learned emissions**: Optional mode with emission matrix `E` for transfer learning/relabeling experiments.

3. **Action indexing**: Actions are indexed such that `a[t-1]` affects transition from `t-1` to `t`.

4. **Normalization**: Forward/backward passes use per-step normalization to avoid underflow. Log-likelihood stored in base 2.

5. **Random tie-breaking**: Viterbi decoding uses `rargmax()` to avoid systematic bias on ties.

6. **Type flexibility**: MyPy is configured permissively to work with Numba. Type hints are encouraged but not strictly enforced.

7. **Place Fields**: The forward messages (`mess_fwd`) contain the probability distribution over clones at each time step. Place fields are computed by aggregating these probabilities over spatial positions.

8. **Biological Comparison**: The `ext_data_fig_8_*.ipynb` notebooks contain examples of comparing CSCG outputs to biological hippocampus data. Use these as templates for the interactive map.

## Getting Help

- See `README.md` for detailed mathematical documentation
- Check `cscg/__init__.py` for exported API
- Review `notebooks/intro.ipynb` for usage examples and place field computation
- Review `notebooks/ext_data_fig_8_*.ipynb` for biological data comparison examples
- Check `.pre-commit-config.yaml` for code quality standards
- **Remember**: Always run pre-commit before committing!
