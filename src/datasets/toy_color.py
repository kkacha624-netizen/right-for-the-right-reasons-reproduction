from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import TensorDataset


COLORS = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
    ],
    dtype=np.float32,
)


def _top_middle_all_different(image: np.ndarray) -> bool:
    top_middle = [(0, 1), (0, 2), (0, 3)]
    return len({tuple(image[row, col]) for row, col in top_middle}) == 3


def _corners_all_same(image: np.ndarray) -> bool:
    corners = [(0, 0), (0, 4), (4, 0), (4, 4)]
    return len({tuple(image[row, col]) for row, col in corners}) == 1


def _sample(
    rng: np.random.Generator,
    n: int,
    mask_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = COLORS[rng.integers(0, len(COLORS), size=(n, 5, 5))]
    corners = [(0, 0), (0, 4), (4, 0), (4, 4)]
    top_middle = [(0, 1), (0, 2), (0, 3)]
    y = np.zeros(n, dtype=np.int64)
    for i in range(n):
        positive = rng.random() < 0.5
        if positive:
            color = COLORS[rng.integers(0, len(COLORS))]
            for row, col in corners:
                x[i, row, col] = color
            perm = rng.choice(len(COLORS), size=3, replace=False)
            for idx, (row, col) in zip(perm, top_middle):
                x[i, row, col] = COLORS[idx]
            y[i] = 1
        else:
            while _corners_all_same(x[i]) or _top_middle_all_different(x[i]):
                for row, col in corners:
                    x[i, row, col] = COLORS[rng.integers(0, len(COLORS))]
                repeated = COLORS[rng.integers(0, len(COLORS))]
                x[i, 0, 1] = repeated
                x[i, 0, 2] = repeated
                x[i, 0, 3] = COLORS[rng.integers(0, len(COLORS))]
            y[i] = 0

    mask = np.zeros_like(x, dtype=np.float32)
    if mask_mode == "corners":
        masked_positions = corners
    elif mask_mode == "irrelevant":
        mask[:] = 1.0
        masked_positions = corners + top_middle
        for row, col in masked_positions:
            mask[:, row, col, :] = 0.0
        return x.reshape(n, -1), y, mask.reshape(n, -1)
    elif mask_mode == "none":
        masked_positions = []
    else:
        raise ValueError(f"Unknown mask_mode: {mask_mode}")
    for row, col in masked_positions:
        mask[:, row, col, :] = 1.0
    return x.reshape(n, -1), y, mask.reshape(n, -1)


def make_toy_color(
    seed: int = 0,
    n_train: int = 5000,
    n_test: int = 2000,
    mask_mode: str = "corners",
) -> tuple[TensorDataset, TensorDataset, int, int]:
    rng = np.random.default_rng(seed)
    train = _sample(rng, n_train, mask_mode)
    test = _sample(rng, n_test, mask_mode)
    train_ds = TensorDataset(*(torch.tensor(arr) for arr in train))
    test_ds = TensorDataset(*(torch.tensor(arr) for arr in test))
    return train_ds, test_ds, 75, 2
