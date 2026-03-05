python
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from typing import Tuple

def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    data: torch.Tensor,
    target: torch.Tensor,
    loss_fn: nn.Module,
    scaler: GradScaler,
) -> float:
    """
    Single mixed-precision training step.
    Returns the detached loss value.
    """
    model.train()
    optimizer.zero_grad(set_to_none=True)  # faster than zeroing

    with autocast(device_type="cuda", dtype=torch.float16):
        output = model(data)
        loss = loss_fn(output, target)

    scaler.scale(loss).backward()

    # Optional: gradient clipping can be inserted here
    # scaler.unscale_(optimizer)
    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    scaler.step(optimizer)
    scaler.update()

    return loss.item()