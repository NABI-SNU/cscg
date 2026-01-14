"""Data generation utilities for CSCG."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def datagen_structured_obs_room(
    room: ArrayLike,
    start_r: int | None = None,
    start_c: int | None = None,
    no_left: list[tuple[int, int]] = [],  # noqa: B006
    no_right: list[tuple[int, int]] = [],  # noqa: B006
    no_up: list[tuple[int, int]] = [],  # noqa: B006
    no_down: list[tuple[int, int]] = [],  # noqa: B006
    length: int = 10000,
    seed: int = 42,
) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
    """Generate observation-action sequences from a structured room environment.

    Parameters
    ----------
    room : ArrayLike[int, :, :]
        2D  array representing the room. Inaccessible locations are marked by -1.
    start_r : int, optional
        Starting row position. If None, randomly chosen.
    start_c : int, optional
        Starting column position. If None, randomly chosen.
    no_left : list
        List of (r, c) tuples from which left action is not allowed.
    no_right : list
        List of (r, c) tuples from which right action is not allowed.
    no_up : list
        List of (r, c) tuples from which up action is not allowed.
    no_down : list
        List of (r, c) tuples from which down action is not allowed.
    length : int, default=10000
        Length of the generated sequence.
    seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    actions : ArrayLike[int, :]
        Array of actions (0: left, 1: right, 2: up, 3: down)
    x : ArrayLike[int, :]
        Array of observations (room values at each position)
    rc : ArrayLike[int, :, 2]
        Array of (row, column) positions at each time step
    """
    np.random.seed(seed)
    room_arr = np.array(room)
    H, W = room_arr.shape

    if start_r is None:
        start_r = np.random.randint(H)
    if start_c is None:
        start_c = np.random.randint(W)

    actions: np.ndarray = np.zeros(length, dtype=np.int64)
    x: np.ndarray = np.zeros(length, dtype=np.int64)  # observations
    rc: np.ndarray = np.zeros((length, 2), dtype=np.int64)  # actual r&c

    r, c = start_r, start_c
    x[0] = room_arr[r, c]
    rc[0] = r, c

    forbidden_locs = {
        "left": no_left,
        "right": no_right,
        "up": no_up,
        "down": no_down,
    }
    act_no = {
        "left": 0,
        "right": 1,
        "up": 2,
        "down": 3,
    }

    for count in range(length - 1):
        act_list = [
            act_no[action]
            for action in forbidden_locs.keys()
            if (r, c) not in forbidden_locs[action]
        ]
        a = np.random.choice(act_list)

        # Check for actions taking out of the matrix boundary.
        prev_r, prev_c = r, c

        if a == act_no["left"] and 0 < c:
            c -= 1
        elif a == act_no["right"] and c < W - 1:
            c += 1
        elif a == act_no["up"] and 0 < r:
            r -= 1
        elif a == act_no["down"] and r < H - 1:
            r += 1

        # Check whether action is taking to inaccessible states.
        temp_x = room_arr[r, c]
        if temp_x == -1:
            r, c = prev_r, prev_c
            pass

        actions[count] = a
        x[count + 1] = room_arr[r, c]
        rc[count + 1] = r, c

    return actions, x, rc
