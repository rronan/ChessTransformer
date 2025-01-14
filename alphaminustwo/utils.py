import torch
from torch import nn
import os


def init_log(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "log.txt")
    with open(log_file, "w") as _:
        pass
    return log_file


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    step: int,
    log_dir: str,
    val_loss_accum: float,
):
    checkpoint = {
        "model": getattr(model, "_orig_mod", model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "config": model.config,
        "step": step,
        "val_loss": val_loss_accum,
    }
    torch.save(checkpoint, os.path.join(log_dir, f"model_{step:06d}.pt"))


def update_stats_(i, running_stats, stats):
    for k, v in stats.items():
        running_stats[k] = (running_stats[k] * i + v) / (i + 1)
    res = " | ".join([f"{k}: {v:.6f}" for k, v in running_stats.items()])
    return res
