from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .utils import ensure_dir


def save_image_grid(path: str | Path, images: list[np.ndarray], titles: list[str]) -> None:
    ensure_dir(Path(path).parent)
    cols = len(images)
    fig, axes = plt.subplots(1, cols, figsize=(3 * cols, 3))
    if cols == 1:
        axes = [axes]
    for ax, image, title in zip(axes, images, titles):
        ax.imshow(image, cmap="viridis")
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_bar_chart(path: str | Path, values: np.ndarray, labels: list[str], title: str) -> None:
    ensure_dir(Path(path).parent)
    fig, ax = plt.subplots(figsize=(max(6, len(values) * 0.25), 3))
    ax.bar(np.arange(len(values)), values)
    ax.set_xticks(np.arange(len(values)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
