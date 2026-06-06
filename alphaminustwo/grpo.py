import torch
import torch.nn.functional as F


def masked_log_probs(logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    masked = logits.masked_fill(legal_mask == 0, float("-inf"))
    return F.log_softmax(masked, dim=-1)


def gather_logp(logp: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    return logp.gather(-1, actions.unsqueeze(-1)).squeeze(-1)


def exact_kl(logp_theta: torch.Tensor, logp_ref: torch.Tensor) -> torch.Tensor:
    legal = torch.isfinite(logp_theta)
    lt = logp_theta.masked_fill(~legal, 0.0)
    lr = logp_ref.masked_fill(~legal, 0.0)
    p_theta = lt.exp() * legal
    return (p_theta * (lt - lr)).sum(dim=-1)


def exact_entropy(logp_theta: torch.Tensor) -> torch.Tensor:
    legal = torch.isfinite(logp_theta)
    lt = logp_theta.masked_fill(~legal, 0.0)
    p_theta = lt.exp() * legal
    return -(p_theta * lt).sum(dim=-1)


def group_advantages(rewards: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    mean = rewards.mean(dim=-1, keepdim=True)
    # correction=0 so single-episode groups give std=0 -> adv=0 instead of nan
    std = rewards.std(dim=-1, keepdim=True, correction=0)
    return (rewards - mean) / (std + eps)


def grpo_loss(
    logp_new: torch.Tensor,
    logp_old: torch.Tensor,
    advantages: torch.Tensor,
    kl: torch.Tensor,
    entropy: torch.Tensor,
    clip_eps: float,
    kl_beta: float,
    entropy_coef: float,
    move_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    """
    PPO-clipped surrogate with exact-KL penalty and optional entropy bonus.

    All per-move tensors are flat [N] (one entry per (episode, move)):
    - logp_new / logp_old: log pi(a_t|s_t) under current / behavior policy
    - advantages: the episode's group-relative advantage, repeated per move
    - kl / entropy: exact per-state KL to the reference and entropy
    - move_weights: optional [N] weights (e.g. 1/T_i for per-episode length
      normalization); defaults to uniform.
    """
    if move_weights is None:
        move_weights = torch.ones_like(logp_new)
    move_weights = move_weights / move_weights.sum()
    ratio = (logp_new - logp_old).exp()
    clipped = ratio.clamp(1 - clip_eps, 1 + clip_eps)
    surrogate = torch.minimum(ratio * advantages, clipped * advantages)
    pg_loss = -(move_weights * surrogate).sum()
    kl_loss = (move_weights * kl).sum()
    entropy_mean = (move_weights * entropy).sum()
    loss = pg_loss + kl_beta * kl_loss - entropy_coef * entropy_mean
    metrics = {
        "pg_loss": pg_loss.item(),
        "mean_kl": kl_loss.item(),
        "mean_entropy": entropy_mean.item(),
        "mean_ratio": ratio.mean().item(),
        "clip_frac": ((ratio - 1).abs() > clip_eps).float().mean().item(),
        "mean_adv": advantages.mean().item(),
    }
    return loss, metrics
