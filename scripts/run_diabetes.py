from __future__ import annotations

from torch.utils.data import DataLoader

from _runner import common_parser, run_pair
from datasets.iris_cancer import make_iris_cancer
from rrr.evaluate import evaluate
from rrr.utils import get_device, set_seed


def main() -> dict:
    parser = common_parser()
    args = parser.parse_args()
    set_seed(args.seed)
    epochs = args.epochs or (5 if args.quick else 80)
    train_ds, test_ds, test_without_iris_ds, input_dim, output_dim, _ = make_iris_cancer(args.seed)
    device = get_device(args.device)

    def extra_eval(baseline, rrr):
        loader = DataLoader(test_without_iris_ds, batch_size=args.batch_size)
        return {
            "baseline_without_iris": evaluate(baseline, loader, device),
            "rrr_without_iris": evaluate(rrr, loader, device),
        }

    return run_pair(
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


if __name__ == "__main__":
    print(main())
