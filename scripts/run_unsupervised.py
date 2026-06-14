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
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args()
    set_seed(args.seed)
    epochs = args.epochs or (2 if args.quick else 20)
    n_train, n_test = (800, 300) if args.quick else (5000, 2000)
    train_ds, test_ds, input_dim, output_dim = make_toy_color(args.seed, n_train, n_test)
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
            lambda_rrr=args.lambda_rrr if round_id > 0 else 0.0,
            quiet=True,
        )
        model.eval()
        grads = []
        for batch in DataLoader(round_train, batch_size=args.batch_size):
            grads.append(probability_input_gradients(model.to(device), batch[0].to(device)).abs().cpu())
        mean_grad = torch.cat(grads).mean(dim=0)
        top = torch.topk(mean_grad, k=args.top_k).indices
        mask[:, top] = 1.0
        history.append({"round": round_id, "accuracy": metrics["accuracy"], "masked_features": top.tolist()})

    payload = {"experiment_id": "find_another_explanation", "device": str(device), "history": history}
    out_dir = ensure_dir("experiments/find_another_explanation")
    write_json(out_dir / "metrics.json", payload)
    (out_dir / "notes.md").write_text(
        "# Notes\n\nIterative find-another-explanation reproduction: after each model, high-gradient features are added to A for the next model.\n",
        encoding="utf-8",
    )
    return payload


if __name__ == "__main__":
    print(main())
