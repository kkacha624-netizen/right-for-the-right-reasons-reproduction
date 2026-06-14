from __future__ import annotations

from torch.utils.data import DataLoader

from _runner import common_parser, run_pair
from datasets.decoy_mnist import make_decoy_mnist
from rrr.evaluate import evaluate
from rrr.utils import get_device, set_seed


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

    def extra_eval(baseline, rrr):
        loader = DataLoader(test_random_ds, batch_size=args.batch_size)
        return {
            "baseline_random_swatch": evaluate(baseline, loader, device),
            "rrr_random_swatch": evaluate(rrr, loader, device),
        }

    return run_pair(
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


if __name__ == "__main__":
    print(main())
