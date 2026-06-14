from __future__ import annotations

import torch
import torch.nn.functional as F


def log_prob_input_gradients(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor | None = None,
    sum_classes: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = inputs.detach().clone().requires_grad_(True)
    logits = model(x)
    log_probs = F.log_softmax(logits, dim=1)
    if sum_classes:
        selected = log_probs.sum()
    else:
        if targets is None:
            targets = logits.argmax(dim=1)
        selected = log_probs.gather(1, targets.view(-1, 1)).sum()
    gradients = torch.autograd.grad(selected, x, create_graph=True)[0]
    return gradients, logits


def probability_input_gradients(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor | None = None,
) -> torch.Tensor:
    x = inputs.detach().clone().requires_grad_(True)
    logits = model(x)
    probs = F.softmax(logits, dim=1)
    if targets is None:
        targets = logits.argmax(dim=1)
    selected = probs.gather(1, targets.view(-1, 1)).sum()
    return torch.autograd.grad(selected, x)[0].detach()


def masked_gradient_penalty(gradients: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return ((gradients * mask) ** 2).sum(dim=tuple(range(1, gradients.ndim))).mean()
