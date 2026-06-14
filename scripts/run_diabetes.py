from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from _runner import ROOT, common_parser, run_pair
from datasets.iris_cancer import make_iris_cancer
from rrr.evaluate import evaluate
from rrr.explain import gradient_attribution, lime_tabular_attribution, top_attributions
from rrr.models import MLPClassifier
from rrr.train import train_classifier
from rrr.utils import ensure_dir, get_device, set_seed, write_json
from rrr.visualize import save_explanation_bars, save_feature_sample_images


def _mean_std(values: list[float]) -> dict[str, float]:
    arr = np.array(values, dtype=float)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0}


def _save_sample_explanations(baseline, rrr, dataset, feature_names: list[str], device) -> list[dict]:
    rows: list[dict] = []
    for idx in range(5):
        x, y, _ = dataset[idx]
        row = {"sample": idx, "label": int(y)}
        for model_name, model in [("baseline", baseline), ("rrr", rrr)]:
            grad, pred, prob = gradient_attribution(model, x, device)
            lime_values, _, _ = lime_tabular_attribution(model, x, device, num_samples=256, seed=idx)
            path = ROOT / "results" / "figures" / "explanations" / "iris_cancer" / f"{model_name}_sample_{idx:02d}.png"
            save_explanation_bars(
                path,
                grad.reshape(-1),
                lime_values.reshape(-1),
                feature_names,
                f"{model_name} / y={int(y)} / pred={pred} / p={prob:.3f}",
                top_k=12,
            )
            row[f"{model_name}_figure"] = str(path)
            row[f"{model_name}_pred"] = pred
            row[f"{model_name}_prob"] = prob
            row[f"{model_name}_top_gradient"] = top_attributions(grad, feature_names, k=8)
            row[f"{model_name}_top_surrogate"] = top_attributions(lime_values, feature_names, k=8)
        rows.append(row)
    return rows


def main() -> dict:
    parser = common_parser()
    parser.add_argument("--splits", type=int, default=None)
    args = parser.parse_args()
    set_seed(args.seed)
    epochs = args.epochs or (5 if args.quick else 80)
    splits = args.splits or (5 if args.quick else 350)
    train_ds, test_ds, test_without_iris_ds, input_dim, output_dim, _ = make_iris_cancer(args.seed)
    device = get_device(args.device)
    feature_names = make_iris_cancer(args.seed)[5]
    x_samples = train_ds.tensors[0][:5].numpy()
    y_samples = train_ds.tensors[1][:5].numpy().tolist()
    sample_images = save_feature_sample_images(
        ROOT / "results" / "figures" / "samples" / "iris_cancer",
        x_samples,
        feature_names,
        [f"Iris-Cancer sample {i} / y={y}" for i, y in enumerate(y_samples)],
    )

    def extra_eval(baseline, rrr):
        loader = DataLoader(test_without_iris_ds, batch_size=args.batch_size)
        return {
            "baseline_without_iris": evaluate(baseline, loader, device),
            "rrr_without_iris": evaluate(rrr, loader, device),
            "sample_explanations": _save_sample_explanations(baseline, rrr, test_ds, feature_names, device),
        }

    single = run_pair(
        experiment="iris_cancer",
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

    aggregate = {
        "baseline_accuracy": [],
        "baseline_without_iris_accuracy": [],
        "rrr_accuracy": [],
        "rrr_without_iris_accuracy": [],
    }
    for split_id in range(splits):
        split_seed = args.seed + split_id
        set_seed(split_seed)
        split_train, split_test, split_without_iris, _, _, _ = make_iris_cancer(split_seed)
        train_loader = DataLoader(split_train, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(split_test, batch_size=args.batch_size)
        without_loader = DataLoader(split_without_iris, batch_size=args.batch_size)

        baseline = MLPClassifier(input_dim, output_dim).to(device)
        baseline_metrics = train_classifier(
            baseline,
            train_loader,
            test_loader,
            device,
            epochs=epochs,
            lambda_rrr=0.0,
            quiet=True,
        )
        baseline_without = evaluate(baseline, without_loader, device)

        rrr = MLPClassifier(input_dim, output_dim).to(device)
        rrr_metrics = train_classifier(
            rrr,
            train_loader,
            test_loader,
            device,
            epochs=epochs,
            lambda_rrr=args.lambda_rrr,
            quiet=True,
        )
        rrr_without = evaluate(rrr, without_loader, device)

        aggregate["baseline_accuracy"].append(baseline_metrics["accuracy"])
        aggregate["baseline_without_iris_accuracy"].append(baseline_without["accuracy"])
        aggregate["rrr_accuracy"].append(rrr_metrics["accuracy"])
        aggregate["rrr_without_iris_accuracy"].append(rrr_without["accuracy"])

    single["sample_images"] = sample_images
    single["split_aggregate"] = {key: _mean_std(values) for key, values in aggregate.items()}
    single["paper_targets"] = {
        "baseline_accuracy_mean": 0.92,
        "baseline_without_iris_accuracy_mean": 0.81,
        "splits": 350,
    }
    single["reproduction_checks"] = {
        "used_paper_split_count": splits == 350,
        "baseline_accuracy_close_to_paper": abs(single["split_aggregate"]["baseline_accuracy"]["mean"] - 0.92) <= 0.08,
        "baseline_without_iris_degrades": (
            single["split_aggregate"]["baseline_accuracy"]["mean"]
            > single["split_aggregate"]["baseline_without_iris_accuracy"]["mean"]
        ),
        "rrr_reduces_iris_removal_gap": abs(
            single["split_aggregate"]["rrr_accuracy"]["mean"]
            - single["split_aggregate"]["rrr_without_iris_accuracy"]["mean"]
        )
        < abs(
            single["split_aggregate"]["baseline_accuracy"]["mean"]
            - single["split_aggregate"]["baseline_without_iris_accuracy"]["mean"]
        ),
    }
    write_json(ROOT / "experiments" / "iris_cancer" / "metrics.json", single)
    return single


if __name__ == "__main__":
    print(main())
