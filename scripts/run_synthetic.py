from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from _runner import ROOT, common_parser, run_pair
from datasets.toy_color import make_toy_color
from rrr.explain import gradient_attribution, lime_image_grid_attribution
from rrr.gradients import probability_input_gradients
from rrr.utils import get_device, set_seed, write_json
from rrr.visualize import save_dataset_sample_images, save_image_explanation_grid


def _rule_gradient_fractions(model, dataset, device, batch_size: int) -> dict[str, float]:
    corner_idx = []
    top_idx = []
    for row, col in [(0, 0), (0, 4), (4, 0), (4, 4)]:
        corner_idx.extend([(row * 5 + col) * 3 + channel for channel in range(3)])
    for row, col in [(0, 1), (0, 2), (0, 3)]:
        top_idx.extend([(row * 5 + col) * 3 + channel for channel in range(3)])
    total = 0.0
    corner = 0.0
    top = 0.0
    for batch in DataLoader(dataset, batch_size=batch_size):
        grads = probability_input_gradients(model.to(device), batch[0].to(device)).abs().cpu()
        total += float(grads.sum())
        corner += float(grads[:, corner_idx].sum())
        top += float(grads[:, top_idx].sum())
    return {
        "corner_fraction": corner / max(total, 1e-12),
        "top_middle_fraction": top / max(total, 1e-12),
    }


def _save_sample_explanations(baseline, rrr, dataset, device) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for idx in range(5):
        x, y, mask = dataset[idx]
        image = x.reshape(5, 5, 3).numpy()
        mask_image = mask.reshape(5, 5, 3).mean(dim=2).numpy()
        row: dict[str, str] = {"sample": str(idx), "label": str(int(y))}
        for model_name, model in [("baseline", baseline), ("rrr", rrr)]:
            grad, pred, prob = gradient_attribution(model, x, device)
            lime_map, lime_pred, lime_prob = lime_image_grid_attribution(
                model,
                x.reshape(5, 5, 3).permute(2, 0, 1),
                device,
                grid_size=5,
                num_samples=128,
                seed=idx,
            )
            path = ROOT / "results" / "figures" / "explanations" / "toy_color" / f"{model_name}_sample_{idx:02d}.png"
            save_image_explanation_grid(
                path,
                image,
                grad.reshape(5, 5, 3).mean(axis=2),
                lime_map,
                f"{model_name} / y={int(y)} / pred={pred} / p={prob:.3f}",
                mask=mask_image,
            )
            row[f"{model_name}_figure"] = str(path)
            row[f"{model_name}_pred"] = str(pred)
            row[f"{model_name}_prob"] = f"{prob:.6f}"
        rows.append(row)
    return rows


def main() -> dict:
    parser = common_parser()
    parser.set_defaults(lambda_rrr=100.0)
    args = parser.parse_args()
    set_seed(args.seed)
    n_train, n_test = (800, 300) if args.quick else (5000, 2000)
    epochs = args.epochs or (3 if args.quick else 30)
    train_ds, test_ds, input_dim, output_dim = make_toy_color(args.seed, n_train, n_test, mask_mode="corners")
    sample_images = save_dataset_sample_images(
        ROOT / "results" / "figures" / "samples" / "toy_color",
        [train_ds[i][0].reshape(5, 5, 3).numpy() for i in range(5)],
        [f"Toy Color sample {i} / y={int(train_ds[i][1])}" for i in range(5)],
    )
    device = get_device(args.device)

    def extra_eval(baseline, rrr):
        return {
            "baseline_rule_gradients": _rule_gradient_fractions(baseline, test_ds, device, args.batch_size),
            "rrr_rule_gradients": _rule_gradient_fractions(rrr, test_ds, device, args.batch_size),
            "sample_explanations": _save_sample_explanations(baseline, rrr, test_ds, device),
        }

    result = run_pair(
        experiment="toy_color",
        train_ds=train_ds,
        test_ds=test_ds,
        input_dim=input_dim,
        output_dim=output_dim,
        device=device,
        epochs=epochs,
        batch_size=args.batch_size,
        lambda_rrr=args.lambda_rrr,
        extra_eval=extra_eval,
    )
    result["sample_images"] = sample_images
    result["reproduction_checks"] = {
        "rrr_high_accuracy": result["rrr"]["accuracy"] >= 0.90,
        "rrr_uses_top_middle_more_than_baseline": (
            result["extra_eval"]["rrr_rule_gradients"]["top_middle_fraction"]
            > result["extra_eval"]["baseline_rule_gradients"]["top_middle_fraction"]
        ),
        "rrr_suppresses_corners_vs_baseline": (
            result["extra_eval"]["rrr_rule_gradients"]["corner_fraction"]
            < result["extra_eval"]["baseline_rule_gradients"]["corner_fraction"]
        ),
    }
    write_json(ROOT / "experiments" / "toy_color" / "metrics.json", result)
    return result


if __name__ == "__main__":
    print(main())
