from __future__ import annotations

from _runner import common_parser, run_pair
from datasets.newsgroups import make_newsgroups
from rrr.utils import get_device, set_seed


def main() -> dict:
    parser = common_parser()
    parser.add_argument("--max-features", type=int, default=5000)
    args = parser.parse_args()
    set_seed(args.seed)
    max_features = 1000 if args.quick else args.max_features
    epochs = args.epochs or (3 if args.quick else 30)
    train_ds, test_ds, input_dim, output_dim, _ = make_newsgroups(args.seed, max_features)
    return run_pair(
        experiment="newsgroups",
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
