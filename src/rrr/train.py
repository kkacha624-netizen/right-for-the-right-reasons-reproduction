from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .evaluate import evaluate
from .losses import rrr_loss


def train_classifier(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    device: torch.device,
    epochs: int,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    lambda_rrr: float = 0.0,
    quiet: bool = False,
) -> dict[str, float]:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    last_parts = None
    iterator = range(epochs)
    if not quiet:
        iterator = tqdm(iterator, desc="train", leave=False)
    for _ in iterator:
        model.train()
        for batch in train_loader:
            x, y = batch[0].to(device), batch[1].to(device)
            mask = batch[2].to(device) if len(batch) > 2 else None
            optimizer.zero_grad(set_to_none=True)
            parts = rrr_loss(model, x, y, mask, lambda_rrr=lambda_rrr)
            parts.total.backward()
            optimizer.step()
            last_parts = parts
    metrics = evaluate(model, val_loader or train_loader, device)
    if last_parts is not None:
        metrics.update(
            {
                "classification_loss": float(last_parts.classification.detach().cpu()),
                "explanation_regularization_loss": float(last_parts.explanation.detach().cpu()),
            }
        )
    return metrics
