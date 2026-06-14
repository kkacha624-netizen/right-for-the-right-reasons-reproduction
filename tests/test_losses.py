import torch

from rrr.losses import rrr_loss
from rrr.models import MLPClassifier


def test_rrr_loss_has_explanation_term():
    model = MLPClassifier(4, 2)
    x = torch.randn(3, 4)
    y = torch.tensor([0, 1, 0])
    mask = torch.ones_like(x)
    parts = rrr_loss(model, x, y, mask, lambda_rrr=10.0)
    assert parts.total.item() >= parts.classification.item()
    assert parts.explanation.item() >= 0.0
