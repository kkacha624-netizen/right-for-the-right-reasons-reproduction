from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from _runner import common_parser
from datasets.toy_color import make_toy_color
from rrr.gradients import probability_input_gradients
from rrr.models import MLPClassifier
from rrr.train import train_classifier
from rrr.utils import ensure_dir, get_device, set_seed, write_json


def main() -> dict:
    parser = common_parser()
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--cutoff", type=float, default=0.67)
    parser.add_argument("--late-lambda-rrr", type=float, default=1_000_000.0)
    args = parser.parse_args()
    set_seed(args.seed)
    epochs = args.epochs or (2 if args.quick else 20)
    n_train, n_test = (800, 300) if args.quick else (5000, 2000)
    train_ds, test_ds, input_dim, output_dim = make_toy_color(args.seed, n_train, n_test, mask_mode="none")
    device = get_device(args.device)
    x_train, y_train, _ = train_ds.tensors
    mask = torch.zeros_like(x_train)
    history = []

    for round_id in range(args.rounds):
        round_train = TensorDataset(x_train, y_train, mask)
        train_loader = DataLoader(round_train, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size)
        model = MLPClassifier(input_dim, output_dim)
        metrics = train_classifier(
            model,
            train_loader,
            test_loader,
            device,
            epochs=epochs,
            lambda_rrr=0.0 if round_id == 0 else (args.lambda_rrr if round_id == 1 else args.late_lambda_rrr),
            quiet=True,
        )
        model.eval()
        new_masks = []
        for batch in DataLoader(round_train, batch_size=args.batch_size):
            grads = probability_input_gradients(model.to(device), batch[0].to(device)).abs().cpu()
            max_grad = grads.max(dim=1, keepdim=True).values.clamp_min(1e-12)
            new_masks.append((grads / max_grad >= args.cutoff).float())
        round_mask = torch.cat(new_masks, dim=0)
        mask = torch.maximum(mask, round_mask)
        masked_fraction = float(mask.mean())
        history.append(
            {
                "round": round_id,
                "accuracy": metrics["accuracy"],
                "masked_fraction": masked_fraction,
                "cutoff": args.cutoff,
            }
        )

    payload = {
        "experiment_id": "find_another_explanation",
        "device": str(device),
        "history": history,
        "reproduction_checks": {
            "first_model_high_accuracy": history[0]["accuracy"] >= 0.90,
            "second_model_keeps_useful_accuracy": len(history) < 2 or history[1]["accuracy"] >= 0.80,
            "later_model_degrades_after_rules_are_masked": len(history) < 3 or history[-1]["accuracy"] <= 0.70,
            "uses_paper_cutoff": abs(args.cutoff - 0.67) < 1e-9,
        },
    }
    out_dir = ensure_dir("experiments/find_another_explanation")
    write_json(out_dir / "metrics.json", payload)
    (out_dir / "notes.md").write_text(
        "# Notes\n\nIterative find-another-explanation reproduction: after each model, high-gradient features are added to A for the next model.\n",
        encoding="utf-8",
    )
    return payload


if __name__ == "__main__":
    print(main())
