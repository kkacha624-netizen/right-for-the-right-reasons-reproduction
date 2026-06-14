from __future__ import annotations

from _runner import common_parser, run_pair
from datasets.toy_color import make_toy_color
from rrr.utils import get_device, set_seed


def main() -> dict:
    parser = common_parser()
    args = parser.parse_args()
    set_seed(args.seed)
    n_train, n_test = (800, 300) if args.quick else (5000, 2000)
    epochs = args.epochs or (3 if args.quick else 30)
    train_ds, test_ds, input_dim, output_dim = make_toy_color(args.seed, n_train, n_test)
    return run_pair(
        experiment="toy_color",
        train_ds=train_ds,
        test_ds=test_ds,
        input_dim=input_dim,
        output_dim=output_dim,
        device=get_device(args.device),
        epochs=epochs,
        batch_size=args.batch_size,
        lambda_rrr=args.lambda_rrr,
    )


if __name__ == "__main__":
    print(main())
