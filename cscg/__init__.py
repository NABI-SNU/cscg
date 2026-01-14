"""Clone-Structured Cognitive Graphs (CSCG) - Action-Augmented Cloned HMM.

This package implements a clone-structured hidden Markov model (CHMM) where:
- Each observation symbol has multiple "clones" (hidden states)
- Each clone deterministically emits one observation symbol
- Transitions are conditioned on actions

Main components:
- CHMM: Main model class
- Inference algorithms: forward, backward, Viterbi decoding
- Learning: EM updates for transitions and emissions
- Data generation: Utilities for generating synthetic data
"""

from .datagen import datagen_structured_obs_room
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
from .model import CHMM
from .utils import rargmax, validate_seq

__all__ = [
    # Main model
    "CHMM",
    # Inference algorithms
    "forward",
    "backward",
    "forward_mp",
    "backtrace",
    "forwardE",
    "backwardE",
    "forwardE_mp",
    "backtraceE",
    "forward_mp_all",
    "backtrace_all",
    # Learning
    "updateC",
    "updateCE",
    # Utilities
    "validate_seq",
    "rargmax",
    # Data generation
    "datagen_structured_obs_room",
]

__version__ = "0.1.0"
