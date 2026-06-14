from __future__ import annotations

from sklearn.datasets import fetch_20newsgroups

from _runner import ROOT, common_parser, run_pair
from datasets.newsgroups import make_newsgroups
from rrr.utils import get_device, set_seed, write_json
from rrr.visualize import save_text_sample_images


def main() -> dict:
    parser = common_parser()
    parser.add_argument("--max-features", type=int, default=5000)
    parser.add_argument("--remove-metadata", action="store_true")
    args = parser.parse_args()
    set_seed(args.seed)
    max_features = 1000 if args.quick else args.max_features
    epochs = args.epochs or (3 if args.quick else 30)
    train_ds, test_ds, input_dim, output_dim, _ = make_newsgroups(
        args.seed,
        max_features,
        remove_metadata=args.remove_metadata,
    )
    categories = ["alt.atheism", "soc.religion.christian"]
    raw_train = fetch_20newsgroups(
        subset="train",
        categories=categories,
        remove=("headers", "footers", "quotes") if args.remove_metadata else (),
        random_state=args.seed,
    )
    sample_images = save_text_sample_images(
        ROOT / "results" / "figures" / "samples" / "newsgroups",
        raw_train.data[:5],
        [f"20 Newsgroups sample {i} / y={raw_train.target_names[raw_train.target[i]]}" for i in range(5)],
    )
    result = run_pair(
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
    result["sample_images"] = sample_images
    result["preprocessing"] = {
        "categories": categories,
        "max_features": max_features,
        "remove_metadata": args.remove_metadata,
    }
    result["paper_targets"] = {"baseline_accuracy": 0.94, "tfidf_components": 5000}
    result["reproduction_checks"] = {
        "uses_5000_tfidf_components": max_features == 5000,
        "baseline_close_to_paper_94_percent": abs(result["baseline"]["accuracy"] - 0.94) <= 0.06,
        "rrr_retains_reasonable_accuracy": result["rrr"]["accuracy"] >= result["baseline"]["accuracy"] - 0.08,
    }
    write_json(ROOT / "experiments" / "newsgroups" / "metrics.json", result)
    return result


if __name__ == "__main__":
    print(main())
