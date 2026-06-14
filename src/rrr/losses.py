from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .gradients import log_prob_input_gradients, masked_gradient_penalty


@dataclass
class LossParts:
    total: torch.Tensor
    classification: torch.Tensor
    explanation: torch.Tensor


def rrr_loss(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    annotation_mask: torch.Tensor | None,
    lambda_rrr: float = 0.0,
) -> LossParts:
    if annotation_mask is None or lambda_rrr <= 0:
        logits = model(inputs)
        classification = F.cross_entropy(logits, targets)
        explanation = torch.zeros((), device=inputs.device)
        return LossParts(classification, classification, explanation)

    gradients, logits = log_prob_input_gradients(model, inputs, targets)
    classification = F.cross_entropy(logits, targets)
    explanation = masked_gradient_penalty(gradients, annotation_mask)
    total = classification + lambda_rrr * explanation
    return LossParts(total, classification, explanation)
