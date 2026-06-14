from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rrr.models import ConvMNISTClassifier, MLPClassifier
from rrr.train import train_classifier
from rrr.utils import ensure_dir, get_device, set_seed, write_json


def common_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quick", action="store_true", help="short smoke-test run")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lambda-rrr", type=float, default=1000.0)
    return parser


def run_pair(
    *,
    experiment: str,
    train_ds,
    test_ds,
    input_dim: int,
    output_dim: int,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lambda_rrr: float,
    model_kind: str = "mlp",
    extra_eval=None,
) -> dict:
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    def build_model():
        if model_kind == "conv":
            return ConvMNISTClassifier(output_dim)
        return MLPClassifier(input_dim=input_dim, output_dim=output_dim)

    baseline = build_model()
    baseline_metrics = train_classifier(
        baseline,
        train_loader,
        test_loader,
        device,
        epochs=epochs,
        lambda_rrr=0.0,
        quiet=True,
    )

    rrr = build_model()
    rrr_metrics = train_classifier(
        rrr,
        train_loader,
        test_loader,
        device,
        epochs=epochs,
        lambda_rrr=lambda_rrr,
        quiet=True,
    )

    payload = {
        "experiment_id": experiment,
        "device": str(device),
        "epochs": epochs,
        "batch_size": batch_size,
        "lambda_rrr": lambda_rrr,
        "baseline": baseline_metrics,
        "rrr": rrr_metrics,
    }
    if extra_eval is not None:
        payload["extra_eval"] = extra_eval(baseline, rrr)

    out_dir = ensure_dir(ROOT / "experiments" / experiment)
    write_json(out_dir / "metrics.json", payload)
    (out_dir / "notes.md").write_text(
        "# Notes\n\n"
        "Implemented as a PyTorch reproduction of the paper setting. "
        "Exact numbers can differ because some hyperparameters are under-specified in the paper.\n",
        encoding="utf-8",
    )
    return payload


def write_summary(rows: list[dict]) -> None:
    path = ensure_dir(ROOT / "results" / "tables") / "reproduction_summary.csv"
    fieldnames = ["experiment_id", "dataset", "method", "metric", "reproduced_result", "status", "notes"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
