import torch
import random
from torch import nn
import os

MANUAL_SEED = 1


def set_seed(device: str):
    torch.manual_seed(MANUAL_SEED)
    if device == "cuda":
        torch.cuda.manual_seed(MANUAL_SEED)
    random.seed(MANUAL_SEED)


def set_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    torch.set_float32_matmul_precision("high")  # on RTF4090, 40% speedup
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    return device


def init_log(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "log.txt")
    with open(log_file, "w") as _:
        pass
    return log_file


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
):
    print("Loading:", checkpoint_path)
    chkp = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(chkp["model"])
    optimizer.load_state_dict(chkp["optimizer"])
    scheduler.load_state_dict(chkp["scheduler"])
    return chkp["step"]


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    step: int,
    log_dir: str,
    current_elo: float,
):
    checkpoint = {
        "model": getattr(model, "_orig_mod", model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "config": model.config,
        "step": step,
        "current_elo": current_elo,
    }
    torch.save(checkpoint, os.path.join(log_dir, f"model_{step:06d}.pt"))


def update_stats_(i, running_stats, stats):
    for k, v in stats.items():
        running_stats[k] = (running_stats[k] * i + v) / (i + 1)
    res = " | ".join([f"{k}: {v:.6f}" for k, v in running_stats.items()])
    return res
