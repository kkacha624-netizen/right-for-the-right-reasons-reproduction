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


def save_dataset_sample_images(
    output_dir: str | Path,
    images: list[np.ndarray],
    titles: list[str],
    cmap: str | None = None,
) -> list[str]:
    target_dir = ensure_dir(output_dir)
    paths: list[str] = []
    for idx, (image, title) in enumerate(zip(images[:5], titles[:5])):
        path = target_dir / f"sample_{idx:02d}.png"
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(image, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(str(path))
    return paths


def save_feature_sample_images(
    output_dir: str | Path,
    rows: np.ndarray,
    labels: list[str],
    titles: list[str],
    top_k: int = 12,
) -> list[str]:
    target_dir = ensure_dir(output_dir)
    paths: list[str] = []
    for idx, row in enumerate(rows[:5]):
        order = np.argsort(np.abs(row))[-top_k:][::-1]
        path = target_dir / f"sample_{idx:02d}.png"
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.bar(np.arange(len(order)), row[order])
        ax.set_xticks(np.arange(len(order)))
        ax.set_xticklabels([labels[i] for i in order], rotation=45, ha="right", fontsize=8)
        ax.set_title(titles[idx])
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(str(path))
    return paths


def save_text_sample_images(
    output_dir: str | Path,
    texts: list[str],
    titles: list[str],
) -> list[str]:
    target_dir = ensure_dir(output_dir)
    paths: list[str] = []
    for idx, text in enumerate(texts[:5]):
        path = target_dir / f"sample_{idx:02d}.png"
        clipped = " ".join(text.split())[:900]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.text(0.01, 0.98, clipped, va="top", ha="left", wrap=True, fontsize=9)
        ax.set_title(titles[idx])
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(str(path))
    return paths


def save_explanation_bars(
    path: str | Path,
    gradient_values: np.ndarray,
    lime_values: np.ndarray,
    labels: list[str],
    title: str,
    top_k: int = 10,
) -> None:
    ensure_dir(Path(path).parent)
    combined = np.abs(gradient_values) + np.abs(lime_values)
    order = np.argsort(combined)[-top_k:][::-1]
    x = np.arange(len(order))
    width = 0.4
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.bar(x - width / 2, gradient_values[order], width, label="input gradient")
    ax.bar(x + width / 2, lime_values[order], width, label="local surrogate")
    ax.set_xticks(x)
    ax.set_xticklabels([labels[i] for i in order], rotation=45, ha="right", fontsize=8)
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_image_explanation_grid(
    path: str | Path,
    input_image: np.ndarray,
    gradient_map: np.ndarray,
    lime_map: np.ndarray,
    title: str,
    mask: np.ndarray | None = None,
    cmap: str | None = None,
) -> None:
    images = [input_image]
    titles = ["input"]
    if mask is not None:
        images.append(mask)
        titles.append("annotation mask")
    images.extend([np.abs(gradient_map), np.abs(lime_map)])
    titles.extend(["input gradient", "local surrogate"])
    ensure_dir(Path(path).parent)
    fig, axes = plt.subplots(1, len(images), figsize=(3 * len(images), 3))
    if len(images) == 1:
        axes = [axes]
    for ax, image, panel_title in zip(axes, images, titles):
        ax.imshow(image, cmap=cmap if panel_title == "input" else "viridis")
        ax.set_title(panel_title)
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
