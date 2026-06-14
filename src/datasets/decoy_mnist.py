from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import TensorDataset
from torchvision import datasets, transforms


def _add_decoy(
    images: torch.Tensor,
    labels: torch.Tensor,
    seed: int,
    random_shade: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    x = images.clone()
    n = x.shape[0]
    corners = [(0, 0), (0, 24), (24, 0), (24, 24)]
    corner_ids = torch.randint(0, len(corners), (n,), generator=generator)
    if random_shade:
        shades = torch.rand(n, generator=generator)
    else:
        shades = (255.0 - 25.0 * labels.float()) / 255.0
    mask = torch.zeros_like(x)
    for i, corner_id in enumerate(corner_ids.tolist()):
        row, col = corners[corner_id]
        x[i, :, row : row + 4, col : col + 4] = shades[i]
        mask[i, :, row : row + 4, col : col + 4] = 1.0
    return x, mask


def make_decoy_mnist(
    data_dir: str | Path = "data/raw",
    seed: int = 0,
    train_limit: int | None = None,
    test_limit: int | None = None,
    download: bool = True,
) -> tuple[TensorDataset, TensorDataset, TensorDataset, int, int]:
    transform = transforms.ToTensor()
    train = datasets.MNIST(data_dir, train=True, download=download, transform=transform)
    test = datasets.MNIST(data_dir, train=False, download=download, transform=transform)

    train_images = torch.stack([train[i][0] for i in range(len(train))])
    train_labels = torch.tensor([train[i][1] for i in range(len(train))], dtype=torch.long)
    test_images = torch.stack([test[i][0] for i in range(len(test))])
    test_labels = torch.tensor([test[i][1] for i in range(len(test))], dtype=torch.long)
    if train_limit:
        train_images, train_labels = train_images[:train_limit], train_labels[:train_limit]
    if test_limit:
        test_images, test_labels = test_images[:test_limit], test_labels[:test_limit]

    train_x, train_mask = _add_decoy(train_images, train_labels, seed=seed, random_shade=False)
    test_correlated_x, test_correlated_mask = _add_decoy(test_images, test_labels, seed=seed + 1, random_shade=False)
    test_random_x, test_random_mask = _add_decoy(test_images, test_labels, seed=seed + 2, random_shade=True)

    return (
        TensorDataset(train_x, train_labels, train_mask),
        TensorDataset(test_correlated_x, test_labels, test_correlated_mask),
        TensorDataset(test_random_x, test_labels, test_random_mask),
        28 * 28,
        10,
    )
