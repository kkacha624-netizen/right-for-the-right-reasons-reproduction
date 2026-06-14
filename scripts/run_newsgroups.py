from __future__ import annotations

import csv

from sklearn.datasets import fetch_20newsgroups
from torch.utils.data import DataLoader

from _runner import ROOT, common_parser, run_pair
from datasets.newsgroups import make_newsgroups
from rrr.gradients import probability_input_gradients
from rrr.utils import get_device, set_seed, write_json
from rrr.visualize import save_text_sample_images


def _write_top_gradient_words(baseline, rrr, dataset, vocab: list[str], device, batch_size: int) -> dict[str, str]:
    def top_words(model) -> list[tuple[str, float]]:
        total = None
        seen = 0
        for batch in DataLoader(dataset, batch_size=batch_size):
            x = batch[0].to(device)
            grads = probability_input_gradients(model.to(device), x).abs().cpu()
            total = grads.sum(dim=0) if total is None else total + grads.sum(dim=0)
            seen += grads.shape[0]
            if seen >= 512:
                break
        mean_grad = total / max(seen, 1)
        order = mean_grad.argsort(descending=True)[:50].tolist()
        return [(vocab[i], float(mean_grad[i])) for i in order]

    out_dir = ROOT / "results" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "baseline_top_gradient_words": out_dir / "newsgroups_baseline_top_gradient_words.csv",
        "rrr_top_gradient_words": out_dir / "newsgroups_rrr_top_gradient_words.csv",
    }
    for key, rows in [
        ("baseline_top_gradient_words", top_words(baseline)),
        ("rrr_top_gradient_words", top_words(rrr)),
    ]:
        with paths[key].open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["rank", "word", "mean_abs_input_gradient"])
            for rank, (word, value) in enumerate(rows, start=1):
                writer.writerow([rank, word, value])
    return {key: str(path) for key, path in paths.items()}


def main() -> dict:
    parser = common_parser()
    parser.add_argument("--max-features", type=int, default=5000)
    parser.add_argument("--remove-metadata", action="store_true")
    args = parser.parse_args()
    set_seed(args.seed)
    max_features = 1000 if args.quick else args.max_features
    epochs = args.epochs or (3 if args.quick else 30)
    train_ds, test_ds, input_dim, output_dim, vocab = make_newsgroups(
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
    device = get_device(args.device)

    def extra_eval(baseline, rrr):
        return _write_top_gradient_words(baseline, rrr, test_ds, vocab, device, args.batch_size)

    result = run_pair(
        experiment="newsgroups",
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
