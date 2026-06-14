from __future__ import annotations

import numpy as np
import torch
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import TensorDataset


DEFAULT_FORBIDDEN_WORDS = [
    "god",
    "christian",
    "christians",
    "jesus",
    "bible",
    "religion",
    "atheism",
    "atheists",
]


def make_newsgroups(
    seed: int = 0,
    max_features: int = 5000,
    forbidden_words: list[str] | None = None,
) -> tuple[TensorDataset, TensorDataset, int, int, list[str]]:
    categories = ["alt.atheism", "soc.religion.christian"]
    train = fetch_20newsgroups(
        subset="train",
        categories=categories,
        remove=("headers", "footers", "quotes"),
        random_state=seed,
    )
    test = fetch_20newsgroups(
        subset="test",
        categories=categories,
        remove=("headers", "footers", "quotes"),
        random_state=seed,
    )
    vectorizer = TfidfVectorizer(stop_words="english", max_features=max_features)
    x_train = vectorizer.fit_transform(train.data).astype(np.float32).toarray()
    x_test = vectorizer.transform(test.data).astype(np.float32).toarray()
    vocab = vectorizer.get_feature_names_out().tolist()
    forbidden = set(forbidden_words or DEFAULT_FORBIDDEN_WORDS)
    mask_vector = np.array([1.0 if word in forbidden else 0.0 for word in vocab], dtype=np.float32)
    train_mask = np.tile(mask_vector, (x_train.shape[0], 1))
    test_mask = np.tile(mask_vector, (x_test.shape[0], 1))

    train_ds = TensorDataset(torch.tensor(x_train), torch.tensor(train.target, dtype=torch.long), torch.tensor(train_mask))
    test_ds = TensorDataset(torch.tensor(x_test), torch.tensor(test.target, dtype=torch.long), torch.tensor(test_mask))
    return train_ds, test_ds, x_train.shape[1], 2, vocab
