from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from pathlib import Path

from _runner import ROOT, common_parser
from datasets.toy_color import make_toy_color
from rrr.gradients import probability_input_gradients
from rrr.models import MLPClassifier
from rrr.train import train_classifier
from rrr.utils import get_device, set_seed, write_json
from rrr.visualize import save_image_grid


def main() -> dict:
    parser = common_parser()
    args = parser.parse_args()
    set_seed(args.seed)
    n_train, n_test = (800, 300) if args.quick else (5000, 2000)
    epochs = args.epochs or (2 if args.quick else 30)
    train_ds, test_ds, input_dim, output_dim = make_toy_color(args.seed, n_train, n_test, mask_mode="corners")
    device = get_device(args.device)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    baseline = MLPClassifier(input_dim, output_dim)
    rrr = MLPClassifier(input_dim, output_dim)
    baseline_metrics = train_classifier(baseline, loader, test_loader, device, epochs, lambda_rrr=0.0, quiet=True)
    rrr_metrics = train_classifier(rrr, loader, test_loader, device, epochs, lambda_rrr=args.lambda_rrr, quiet=True)

    x, _, mask = test_ds[0]
    x_batch = x.unsqueeze(0).to(device)
    baseline_grad = probability_input_gradients(baseline.to(device), x_batch).cpu().reshape(5, 5, 3).abs().mean(dim=2).numpy()
    rrr_grad = probability_input_gradients(rrr.to(device), x_batch).cpu().reshape(5, 5, 3).abs().mean(dim=2).numpy()
    image = x.reshape(5, 5, 3).numpy()
    mask_image = mask.reshape(5, 5, 3).mean(dim=2).numpy()
    save_image_grid(
        ROOT / "results" / "figures" / "explanations" / "toy_color_gradients.png",
        [image, baseline_grad, rrr_grad, mask_image],
        ["input", "baseline gradient", "rrr gradient", "A mask"],
    )
    figure = ROOT / "results" / "figures" / "explanations" / "toy_color_gradients.png"
    payload = {
        "experiment_id": "explanation_visualization",
        "baseline": baseline_metrics,
        "rrr": rrr_metrics,
        "figure": str(figure),
        "reproduction_checks": {
            "figure_created": figure.exists(),
            "baseline_high_accuracy": baseline_metrics["accuracy"] >= 0.90,
            "rrr_high_accuracy": rrr_metrics["accuracy"] >= 0.90,
        },
    }
    write_json(ROOT / "experiments" / "explanation_visualization" / "metrics.json", payload)
    return payload


if __name__ == "__main__":
    print(main())
