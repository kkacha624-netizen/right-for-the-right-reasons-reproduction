from __future__ import annotations

import numpy as np
import torch
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset


def make_iris_cancer(
    seed: int = 0,
    test_size: float = 0.3,
) -> tuple[TensorDataset, TensorDataset, TensorDataset, int, int, list[str]]:
    iris = load_iris()
    cancer = load_breast_cancer()

    iris_mask = iris.target != 0
    iris_x = iris.data[iris_mask][:100]
    iris_y = (iris.target[iris_mask][:100] == 2).astype(np.int64)

    cancer_class0 = np.where(cancer.target == 0)[0][:50]
    cancer_class1 = np.where(cancer.target == 1)[0][:50]
    cancer_x = np.vstack([cancer.data[cancer_class0], cancer.data[cancer_class1]])

    x = np.hstack([iris_x, cancer_x]).astype(np.float32)
    y = iris_y.astype(np.int64)
    feature_names = [f"iris:{name}" for name in iris.feature_names] + [
        f"cancer:{name}" for name in cancer.feature_names
    ]

    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(idx, test_size=test_size, stratify=y, random_state=seed)
    scaler = StandardScaler().fit(x[train_idx])
    x_scaled = scaler.transform(x).astype(np.float32)

    mask = np.zeros_like(x_scaled, dtype=np.float32)
    mask[:, :4] = 1.0
    x_without_iris = x_scaled.copy()
    x_without_iris[:, :4] = 0.0

    def ds(arr: np.ndarray, indices: np.ndarray) -> TensorDataset:
        return TensorDataset(
            torch.tensor(arr[indices]),
            torch.tensor(y[indices]),
            torch.tensor(mask[indices]),
        )

    return ds(x_scaled, train_idx), ds(x_scaled, test_idx), ds(x_without_iris, test_idx), x.shape[1], 2, feature_names
