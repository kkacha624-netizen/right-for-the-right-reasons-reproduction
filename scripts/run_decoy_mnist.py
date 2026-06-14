from __future__ import annotations

from torch.utils.data import DataLoader

from _runner import ROOT, common_parser, run_pair
from datasets.decoy_mnist import make_decoy_mnist
from rrr.evaluate import evaluate
from rrr.gradients import probability_input_gradients
from rrr.utils import get_device, set_seed, write_json
from rrr.visualize import save_dataset_sample_images


def _swatch_gradient_fraction(model, dataset, device, batch_size: int) -> float:
    masked = 0.0
    total = 0.0
    for batch in DataLoader(dataset, batch_size=batch_size):
        x, _, mask = batch[0].to(device), batch[1].to(device), batch[2].to(device)
        grads = probability_input_gradients(model.to(device), x).abs()
        masked += float((grads * mask).sum().detach().cpu())
        total += float(grads.sum().detach().cpu())
    return masked / max(total, 1e-12)


def main() -> dict:
    parser = common_parser()
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()
    set_seed(args.seed)
    train_limit = 2000 if args.quick else None
    test_limit = 1000 if args.quick else None
    epochs = args.epochs or (1 if args.quick else 10)
    train_ds, test_correlated_ds, test_random_ds, input_dim, output_dim = make_decoy_mnist(
        args.data_dir,
        seed=args.seed,
        train_limit=train_limit,
        test_limit=test_limit,
        download=not args.no_download,
    )
    device = get_device(args.device)
    sample_images = save_dataset_sample_images(
        ROOT / "results" / "figures" / "samples" / "decoy_mnist",
        [train_ds[i][0].squeeze(0).numpy() for i in range(5)],
        [f"Decoy MNIST sample {i} / y={int(train_ds[i][1])}" for i in range(5)],
        cmap="gray",
    )

    def extra_eval(baseline, rrr):
        loader = DataLoader(test_random_ds, batch_size=args.batch_size)
        return {
            "baseline_random_swatch": evaluate(baseline, loader, device),
            "rrr_random_swatch": evaluate(rrr, loader, device),
            "baseline_swatch_gradient_fraction": _swatch_gradient_fraction(
                baseline, test_random_ds, device, args.batch_size
            ),
            "rrr_swatch_gradient_fraction": _swatch_gradient_fraction(rrr, test_random_ds, device, args.batch_size),
        }

    result = run_pair(
        experiment="decoy_mnist",
        train_ds=train_ds,
        test_ds=test_correlated_ds,
        input_dim=input_dim,
        output_dim=output_dim,
        device=device,
        epochs=epochs,
        batch_size=args.batch_size,
        lambda_rrr=args.lambda_rrr,
        model_kind="conv",
        extra_eval=extra_eval,
    )
    result["sample_images"] = sample_images
    result["paper_targets"] = {
        "baseline_random_swatch_accuracy": 0.55,
        "rrr_random_swatch_accuracy": "near_standard_mnist",
    }
    result["reproduction_checks"] = {
        "baseline_uses_decoy": result["extra_eval"]["baseline_random_swatch"]["accuracy"] <= 0.70,
        "rrr_generalizes_when_swatch_randomized": result["extra_eval"]["rrr_random_swatch"]["accuracy"] >= 0.90,
        "rrr_suppresses_swatch_gradients": (
            result["extra_eval"]["rrr_swatch_gradient_fraction"]
            < result["extra_eval"]["baseline_swatch_gradient_fraction"]
        ),
    }
    write_json(ROOT / "experiments" / "decoy_mnist" / "metrics.json", result)
    return result


if __name__ == "__main__":
    print(main())
