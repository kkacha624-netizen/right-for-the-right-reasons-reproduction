from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import Ridge

from .gradients import probability_input_gradients


@torch.no_grad()
def predict_proba(model: torch.nn.Module, x: torch.Tensor, device: torch.device) -> np.ndarray:
    model.eval()
    logits = model(x.to(device))
    return F.softmax(logits, dim=1).detach().cpu().numpy()


def gradient_attribution(
    model: torch.nn.Module,
    x: torch.Tensor,
    device: torch.device,
) -> tuple[np.ndarray, int, float]:
    probs = predict_proba(model, x.unsqueeze(0), device)[0]
    pred = int(np.argmax(probs))
    grad = probability_input_gradients(model.to(device), x.unsqueeze(0).to(device), torch.tensor([pred], device=device))
    return grad.detach().cpu().squeeze(0).numpy(), pred, float(probs[pred])


def lime_tabular_attribution(
    model: torch.nn.Module,
    x: torch.Tensor,
    device: torch.device,
    num_samples: int = 256,
    seed: int = 0,
    noise_scale: float = 0.5,
) -> tuple[np.ndarray, int, float]:
    rng = np.random.default_rng(seed)
    x_np = x.detach().cpu().numpy().astype(np.float32)
    probs = predict_proba(model, x.unsqueeze(0), device)[0]
    pred = int(np.argmax(probs))
    perturb = x_np + rng.normal(0.0, noise_scale, size=(num_samples, x_np.size)).astype(np.float32)
    perturb[0] = x_np
    y = predict_proba(model, torch.tensor(perturb, dtype=x.dtype), device)[:, pred]
    distances = np.linalg.norm(perturb - x_np, axis=1)
    kernel_width = np.sqrt(x_np.size) * 0.75
    weights = np.exp(-(distances**2) / max(kernel_width**2, 1e-12))
    surrogate = Ridge(alpha=1.0)
    surrogate.fit(perturb, y, sample_weight=weights)
    return surrogate.coef_.astype(np.float32), pred, float(probs[pred])


def lime_image_grid_attribution(
    model: torch.nn.Module,
    x: torch.Tensor,
    device: torch.device,
    grid_size: int,
    num_samples: int = 256,
    seed: int = 0,
) -> tuple[np.ndarray, int, float]:
    rng = np.random.default_rng(seed)
    c, h, w = x.shape
    probs = predict_proba(model, x.unsqueeze(0), device)[0]
    pred = int(np.argmax(probs))
    masks = rng.integers(0, 2, size=(num_samples, grid_size, grid_size)).astype(np.float32)
    masks[0] = 1.0
    perturbed = []
    baseline = torch.zeros_like(x)
    for mask in masks:
        mask_tensor = torch.tensor(mask, dtype=x.dtype).repeat_interleave(h // grid_size, axis=0).repeat_interleave(
            w // grid_size, axis=1
        )
        mask_tensor = mask_tensor[:h, :w].unsqueeze(0).repeat(c, 1, 1)
        perturbed.append(x * mask_tensor + baseline * (1.0 - mask_tensor))
    batch = torch.stack(perturbed)
    y = predict_proba(model, batch, device)[:, pred]
    distances = np.linalg.norm((1.0 - masks).reshape(num_samples, -1), axis=1)
    kernel_width = np.sqrt(grid_size * grid_size) * 0.75
    weights = np.exp(-(distances**2) / max(kernel_width**2, 1e-12))
    surrogate = Ridge(alpha=1.0)
    surrogate.fit(masks.reshape(num_samples, -1), y, sample_weight=weights)
    heatmap = surrogate.coef_.reshape(grid_size, grid_size)
    heatmap = np.repeat(np.repeat(heatmap, h // grid_size, axis=0), w // grid_size, axis=1)[:h, :w]
    return heatmap.astype(np.float32), pred, float(probs[pred])


def top_attributions(values: np.ndarray, labels: list[str], k: int = 10) -> list[dict[str, float | str | int]]:
    flat = np.asarray(values).reshape(-1)
    order = np.argsort(np.abs(flat))[-k:][::-1]
    return [
        {"rank": rank, "feature": labels[int(idx)], "attribution": float(flat[int(idx)])}
        for rank, idx in enumerate(order, start=1)
    ]
