import wandb
import sys
from collections import defaultdict
import random
from tqdm import trange
import torch

from alphaminustwo.model import GPT
from alphaminustwo.puzzle import load_puzzles, evaluate_model_on_puzzles
from alphaminustwo.utils import (
    init_log,
    update_stats_,
    save_checkpoint,
    set_device,
    load_checkpoint,
    set_seed,
)
from alphaminustwo.dataset import get_train_loader_line_augmented
from alphaminustwo import config
from alphaminustwo.schedulers import get_scheduler

train_cfg = config.TrainCFG()
model_cfg = config.GPT124M()

device = set_device()
set_seed()

model = GPT(model_cfg).to(device)
print(sum([x.numel() for x in model.parameters() if x.requires_grad]), "parameters")
train_loader = get_train_loader_line_augmented(
    data_path=train_cfg.data_path,
    bsz=train_cfg.bsz,
    n_max=train_cfg.n_max,
    min_depth=train_cfg.min_depth,
    num_workers=train_cfg.num_workers,
    extra_embedding=train_cfg.extra_embedding,
    shuffle=train_cfg.shuffle,
)
optimizer = model.configure_optimizers(
    train_cfg.weight_decay, train_cfg.lr, (train_cfg.beta1, train_cfg.beta2), device
)
scheduler = get_scheduler(optimizer, train_cfg)

init_step = 0
if len(sys.argv) > 1:
    init_step = load_checkpoint(sys.argv[1], model, optimizer, scheduler)
if train_cfg.compile:
    model = torch.compile(model)
    print("Model compiled")

wandb.login()
run = wandb.init(
    project="chess_transformer",
    config={"model": vars(model_cfg), "training": vars(train_cfg)},
)
init_log(train_cfg.log_dir)

puzzles = load_puzzles(train_cfg.puzzle_path)
current_elo = train_cfg.initial_puzzle_elo

stats = {}
for step in range(init_step, train_cfg.max_steps, train_cfg.log_interval):
    model.eval()
    with torch.no_grad():
        puzzle_sample = random.sample(puzzles, train_cfg.n_puzzles)
        current_elo, result_list = evaluate_model_on_puzzles(
            model=model,
            puzzles=puzzle_sample,
            initial_elo_estimate=current_elo,
            extra_embedding=train_cfg.extra_embedding,
        )
    wandb.log({"puzzle_elo": current_elo, "step": step})
    if step > 0 and step % train_cfg.checkpoint_interval == 0:
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=step,
            log_dir=train_cfg.log_dir,
            current_elo=current_elo,
        )
    model.train()
    running_stats = defaultdict(float)
    for i in (pbar := trange(train_cfg.log_interval)):
        optimizer.zero_grad()
        loss_score, loss_move, loss = 0, 0, 0
        for j in range(train_cfg.accumulate_grad_steps):
            x, y, z = next(train_loader)  # Load new batch for each accumulation step
            x, y, z = x.to(device), y.to(device), z.to(device)
            *_, loss_score_j, loss_move_j, loss_j = model(x, y, z)
            loss_j = loss_j / train_cfg.accumulate_grad_steps
            loss_j.backward()
            loss_score += loss_score_j / train_cfg.accumulate_grad_steps
            loss_move += loss_move_j / train_cfg.accumulate_grad_steps
            loss += loss_j
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()
        scheduler.step()
        stats = {
            "loss_score": loss_score.item(),
            "loss_move": loss_move.item(),
            "loss": loss.item(),
            "grad_norm": norm.item(),
            "lr": scheduler.get_last_lr()[0],
        }
        running_stats_str = update_stats_(i, running_stats, stats)
        desc = f"{step + i + 1:06d} | {running_stats_str}"
        pbar.set_description(desc)
    wandb.log({"step": step, **running_stats})
