from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from datasets.decoy_mnist import make_decoy_mnist
from datasets.newsgroups import make_newsgroups


def main() -> None:
    make_decoy_mnist(ROOT / "data" / "raw", train_limit=1, test_limit=1, download=True)
    make_newsgroups(max_features=10)
    print("Downloaded MNIST and 20 Newsgroups source data.")


if __name__ == "__main__":
    main()
