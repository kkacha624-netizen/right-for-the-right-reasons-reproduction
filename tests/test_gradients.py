import torch

from rrr.gradients import log_prob_input_gradients
from rrr.models import MLPClassifier


def test_log_prob_input_gradients_shape():
    model = MLPClassifier(5, 2)
    x = torch.randn(7, 5)
    y = torch.tensor([0, 1, 0, 1, 0, 1, 0])
    grads, logits = log_prob_input_gradients(model, x, y)
    assert grads.shape == x.shape
    assert logits.shape == (7, 2)
