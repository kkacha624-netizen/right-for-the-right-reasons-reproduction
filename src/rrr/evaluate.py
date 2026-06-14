from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader, device: torch.device) -> dict[str, float]:
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    for batch in loader:
        x, y = batch[0].to(device), batch[1].to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y, reduction="sum")
        pred = logits.argmax(dim=1)
        total += y.numel()
        correct += (pred == y).sum().item()
        loss_sum += loss.item()
    return {"accuracy": correct / max(total, 1), "loss": loss_sum / max(total, 1)}
