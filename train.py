import wandb
import sys
from collections import defaultdict
from tqdm import tqdm
from tqdm import trange
import torch

from alphaminustwo.model import GPT
from alphaminustwo.utils import init_log, update_stats_, save_checkpoint
from alphaminustwo.dataset import get_train_loader, get_val_loader
from alphaminustwo.config import TrainCFG, ModelCFG
from alphaminustwo.schedulers import get_scheduler

train_cfg = TrainCFG()
model_cfg = ModelCFG()

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)
torch.set_float32_matmul_precision("high")  # on RTF4090, 40% speedup
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.manual_seed(train_cfg.manual_seed)
if device == "cuda":
    torch.cuda.manual_seed(train_cfg.manual_seed)

model = GPT(model_cfg).to(device)
print(sum([x.numel() for x in model.parameters() if x.requires_grad]), "parameters")
train_loader = get_train_loader(train_cfg.data_path, train_cfg.bsz, train_cfg.val_size)
val_loader = get_val_loader(train_cfg.data_path, 2 * train_cfg.bsz, train_cfg.val_size)
optimizer = model.configure_optimizers(
    train_cfg.weight_decay, train_cfg.lr, (train_cfg.beta1, train_cfg.beta2), device
)
scheduler = get_scheduler(optimizer, train_cfg)

if len(sys.argv) > 1:
    print("Loading:", sys.argv[1])
    chkp = torch.load(sys.argv[1], weights_only=False)
    optimizer.load_state_dict(chkp["optimizer"])
    model.load_state_dict(chkp["model"])
if train_cfg.compile:
    model = torch.compile(model)

wandb.login()
run = wandb.init(
    project="chess_transformer",
    config=vars(train_cfg) | vars(model_cfg),
    resume_from=train_cfg.wandb_resume_from,
)
if train_cfg.watch_model:
    run.watch(model)
log_file = init_log(train_cfg.log_dir)

for step in range(0, train_cfg.max_steps, train_cfg.log_interval):
    if (step % train_cfg.val_interval == 0 and step > 0) or (
        train_cfg.start_with_eval and step == 0
    ):
        model.eval()
        with torch.no_grad():
            loss_eval_accum, loss_move_accum, loss_accum = 0, 0, 0
            running_stats = defaultdict(float)
            for i, (x, y, z) in enumerate((pbar := tqdm(val_loader))):
                x, y, z = x.to(device), y.to(device), z.to(device)
                *_, loss_eval, loss_move, loss = model(x, y, z)
                stats = {
                    "val_loss_eval": loss_eval.item(),
                    "val_loss_move": loss_move.item(),
                    "val_loss": loss.item(),
                }
                running_stats_str = update_stats_(i, running_stats, stats)
                desc = f"{step:06d} | {running_stats_str}"
                pbar.set_description(desc)
        wandb.log(running_stats)
        with open(log_file, "a") as f:
            f.write(desc + "\n")
        if step > 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                step=step,
                log_dir=train_cfg.log_dir,
                val_loss_accum=loss_accum,
            )
    model.train()
    running_stats = defaultdict(float)
    for i in (pbar := trange(train_cfg.log_interval)):
        x, y, z = next(train_loader)
        x, y, z = x.to(device), y.to(device), z.to(device)
        optimizer.zero_grad()
        *_, loss_eval, loss_move, loss = model(x, y, z)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()
        scheduler.step()
        stats = {
            "train_loss_eval": loss_eval.item(),
            "train_loss_move": loss_move.item(),
            "train_loss": loss.item(),
            "grad_norm": norm.item(),
            "lr": scheduler.get_last_lr()[0],
        }
        running_stats_str = update_stats_(i, running_stats, stats)
        desc = f"{step + i + 1:06d} | {running_stats_str}"
        pbar.set_description(desc)
    wandb.log(running_stats)
    with open(log_file, "a") as f:
        f.write(desc + "\n")
