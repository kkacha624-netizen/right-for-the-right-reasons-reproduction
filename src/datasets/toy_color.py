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


def _sample(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
            while len({tuple(x[i, row, col]) for row, col in corners}) == 1:
                for row, col in corners:
                    x[i, row, col] = COLORS[rng.integers(0, len(COLORS))]
            y[i] = 0

    mask = np.ones_like(x, dtype=np.float32)
    for row, col in corners + top_middle:
        mask[:, row, col, :] = 0.0
    return x.reshape(n, -1), y, mask.reshape(n, -1)


def make_toy_color(
    seed: int = 0,
    n_train: int = 5000,
    n_test: int = 2000,
) -> tuple[TensorDataset, TensorDataset, int, int]:
    rng = np.random.default_rng(seed)
    train = _sample(rng, n_train)
    test = _sample(rng, n_test)
    train_ds = TensorDataset(*(torch.tensor(arr) for arr in train))
    test_ds = TensorDataset(*(torch.tensor(arr) for arr in test))
    return train_ds, test_ds, 75, 2
