import wandb
import sys
import os
from collections import defaultdict
import random
import torch
from dotenv import load_dotenv

from alphaminustwo.model import GPT
from alphaminustwo.puzzle import load_puzzles, evaluate_model_on_puzzles
from alphaminustwo.utils import (
    init_log,
    update_stats_,
    save_checkpoint,
    set_device,
    load_model_only,
    set_seed,
)
from alphaminustwo.dataset import fen2tensor
from alphaminustwo.grpo import (
    masked_log_probs,
    gather_logp,
    exact_kl,
    exact_entropy,
    grpo_loss,
)
from alphaminustwo.rollout import (
    rollout_puzzles,
    rollout_selfplay,
    sample_start_fens,
    flatten_episodes,
)
from alphaminustwo import config
from alphaminustwo.schedulers import get_scheduler

load_dotenv()

grpo_cfg = config.GRPOCFG()
model_cfg = config.GPT124M()

device = set_device()
set_seed(device)

model = GPT(model_cfg).to(device)
print(sum([x.numel() for x in model.parameters() if x.requires_grad]), "parameters")

checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else grpo_cfg.init_checkpoint
init_step, current_elo = 0, grpo_cfg.initial_puzzle_elo
if os.path.exists(checkpoint_path):
    init_step, elo = load_model_only(checkpoint_path, model)
    if elo is not None:
        current_elo = elo
else:
    print(
        f"WARNING: checkpoint {checkpoint_path} not found, "
        "starting GRPO from randomly initialized weights"
    )

# Frozen reference policy for the KL penalty
ref_model = GPT(model_cfg).to(device)
ref_model.load_state_dict(model.state_dict())
ref_model.eval()
ref_model.requires_grad_(False)

if grpo_cfg.freeze_score_head:
    model.score_head.requires_grad_(False)

# configure_optimizers filters out params with requires_grad=False
optimizer = model.configure_optimizers(
    grpo_cfg.weight_decay, grpo_cfg.lr, (grpo_cfg.beta1, grpo_cfg.beta2), device
)
scheduler = get_scheduler(optimizer, grpo_cfg)

if grpo_cfg.compile:
    model = torch.compile(model)  # type: ignore
    print("Model compiled")

wandb.login(key=os.getenv("WANDB_API_KEY"))
run = wandb.init(
    entity="rronan-cole-polytechnique",
    project="chess_transformer",
    config={"model": vars(model_cfg), "grpo": vars(grpo_cfg)},
)
init_log(grpo_cfg.log_dir)

puzzles = load_puzzles(grpo_cfg.puzzle_path)
# Held-out validation split, deterministic across restarts (dedicated RNG so the
# split is independent of global random state)
random.Random(0).shuffle(puzzles)
val_puzzles, puzzles = puzzles[:500_000], puzzles[500_000:]


def collect_episodes():
    episodes = []
    if grpo_cfg.source in ("puzzle", "mixed"):
        puzzle_sample = random.sample(puzzles, grpo_cfg.puzzle_groups_per_step)
        episodes += rollout_puzzles(
            model=model,
            puzzles=puzzle_sample,
            G=grpo_cfg.group_size_G,
            extra_embedding=grpo_cfg.extra_embedding,
            temperature=grpo_cfg.temperature,
            group_id_start=0,
            reward_mode=grpo_cfg.puzzle_reward_mode,
        )
    if grpo_cfg.source in ("selfplay", "mixed"):
        start_fens = sample_start_fens(
            grpo_cfg.selfplay_starts_per_step, grpo_cfg.max_random_opening_plies
        )
        episodes += rollout_selfplay(
            model=model,
            start_fens=start_fens,
            G=grpo_cfg.group_size_G,
            max_plies=grpo_cfg.max_plies,
            extra_embedding=grpo_cfg.extra_embedding,
            temperature=grpo_cfg.temperature,
            group_id_start=grpo_cfg.puzzle_groups_per_step,
        )
    return episodes


stats = {}
running_stats: defaultdict[str, float] = defaultdict(float)
for step in range(init_step, init_step + grpo_cfg.max_steps):
    rel_step = step - init_step
    if rel_step % grpo_cfg.eval_interval == 0:
        model.eval()
        with torch.no_grad():
            puzzle_sample = random.sample(val_puzzles, grpo_cfg.n_puzzles)
            current_elo, result_list = evaluate_model_on_puzzles(
                model=model,
                puzzles=puzzle_sample,
                initial_elo_estimate=current_elo,
                extra_embedding=grpo_cfg.extra_embedding,
            )
        wandb.log({"puzzle_elo": current_elo, "step": step})
        if rel_step > 0 and rel_step % grpo_cfg.checkpoint_interval == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                step=step,
                log_dir=grpo_cfg.log_dir,
                current_elo=current_elo,
            )
        running_stats = defaultdict(float)

    # Rollout phase
    model.eval()
    episodes = collect_episodes()
    batch = flatten_episodes(episodes, grpo_cfg.adv_eps)
    N = len(batch["fens"])

    # Optimization phase
    model.train()
    perm_metrics: defaultdict[str, float] = defaultdict(float)
    n_mb = 0
    for _ in range(grpo_cfg.inner_epochs):
        perm = torch.randperm(N)
        for mb_start in range(0, N, grpo_cfg.train_mb):
            idx = perm[mb_start : mb_start + grpo_cfg.train_mb]
            x = torch.stack(
                [
                    fen2tensor(batch["fens"][i], grpo_cfg.extra_embedding)
                    for i in idx.tolist()
                ]
            ).to(device)
            masks = batch["masks"][idx].to(device)
            actions = batch["actions"][idx].to(device)
            optimizer.zero_grad()
            _, logits, *_ = model(x, None)
            logp = masked_log_probs(logits / grpo_cfg.temperature, masks)
            with torch.no_grad():
                _, ref_logits, *_ = ref_model(x, None)
                logp_ref = masked_log_probs(ref_logits / grpo_cfg.temperature, masks)
            loss, metrics = grpo_loss(
                logp_new=gather_logp(logp, actions),
                logp_old=batch["logp_old"][idx].to(device),
                advantages=batch["adv"][idx].to(device),
                kl=exact_kl(logp, logp_ref),
                entropy=exact_entropy(logp),
                clip_eps=grpo_cfg.clip_eps,
                kl_beta=grpo_cfg.kl_beta,
                entropy_coef=grpo_cfg.entropy_coef,
                move_weights=batch["weights"][idx].to(device),
            )
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), grpo_cfg.grad_clip
            )
            optimizer.step()
            scheduler.step()
            metrics["loss"] = loss.item()
            metrics["grad_norm"] = norm.item()
            for k, v in metrics.items():
                perm_metrics[k] = (perm_metrics[k] * n_mb + v) / (n_mb + 1)
            n_mb += 1

    stats = {
        **perm_metrics,
        **{f"reward_{k}": v for k, v in batch["mean_reward"].items()},
        "lr": scheduler.get_last_lr()[0],
        "n_moves": N,
    }
    i = rel_step % grpo_cfg.eval_interval
    running_stats_str = update_stats_(i, running_stats, stats)
    print(f"{step + 1:06d} | {running_stats_str}")
    wandb.log({"step": step, **stats})
