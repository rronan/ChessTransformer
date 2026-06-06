import random
from collections import defaultdict
from dataclasses import dataclass

import torch
import chess

from alphaminustwo.dataset import fen2tensor, uci2index, index2uci, uci_match
from alphaminustwo.grpo import masked_log_probs, gather_logp, group_advantages


@dataclass
class Episode:
    fens: list[str]
    actions: torch.Tensor  # [T] long, flattened 64x64 move indices
    logp_old: torch.Tensor  # [T] behavior-policy log-probs at rollout time
    legal_masks: torch.Tensor  # [T, 4096] bool, cached so train masks match rollout
    reward: float
    group_id: int
    source: str  # "puzzle" | "selfplay"
    # Per-step rewards (puzzle "step" mode). When set, advantages are computed
    # per (group, step) instead of per episode: all attempts that reach step t
    # sit at the identical scripted position, so they form a valid GRPO group.
    step_rewards: torch.Tensor | None = None


def legal_mask_from_board(board: chess.Board) -> torch.Tensor:
    mask = torch.zeros(64**2, dtype=torch.bool)
    for move in board.legal_moves:
        mask[uci2index(move.uci())] = True
    return mask


def index_to_move(board: chess.Board, index: int) -> chess.Move:
    """
    Decode a flattened from-to index into a legal move on `board`.

    uci2index collapses promotion variants into one from-to index, so when the
    bare move is not legal we auto-promote to queen (matching the convention
    that the policy's action space ignores underpromotion).
    """
    uci = index2uci(index)
    move = chess.Move.from_uci(uci)
    if move in board.legal_moves:
        return move
    move = chess.Move.from_uci(uci + "q")
    if move in board.legal_moves:
        return move
    raise ValueError(f"Index {index} ({uci}) is not legal on {board.fen()}")


def batched_policy_step(
    model,
    board_list: list[chess.Board],
    extra_embedding: bool,
    temperature: float = 1.0,
    sample: bool = True,
):
    """
    One policy forward over all boards at once (the GPU-efficiency core: N
    concurrent games = N rows per forward). Same distribution as
    GPT.generate_from_board: softmax over legal moves of the masked logits.

    Returns (actions [N] long, logp_chosen [N], legal_masks [N, 4096] bool),
    all on CPU.
    """
    device = model.device()
    x = torch.stack(
        [fen2tensor(board.fen(), extra_embedding) for board in board_list]
    ).to(device)
    with torch.no_grad():
        _, logits, *_ = model(x, None)
    masks = torch.stack([legal_mask_from_board(board) for board in board_list]).to(
        device
    )
    logp = masked_log_probs(logits / temperature, masks)
    if sample:
        actions = torch.multinomial(logp.exp(), 1).squeeze(-1)
    else:
        actions = logp.argmax(dim=-1)
    logp_chosen = gather_logp(logp, actions)
    return actions.cpu(), logp_chosen.cpu(), masks.cpu()


def rollout_puzzles(
    model,
    puzzles: list[dict],
    G: int,
    extra_embedding: bool,
    temperature: float = 1.0,
    group_id_start: int = 0,
    reward_mode: str = "step",
) -> list[Episode]:
    """
    Full-line puzzle episodes, mirroring evaluate_puzzle's convention: moves
    alternate [opponent, solver, opponent, solver, ...]; the opponent move is
    pushed first and the model plays at the odd indices. After each correct
    solver move the scripted opponent reply is pushed and the line continues;
    the first wrong move ends the episode. Group = G attempts per puzzle.

    reward_mode "step": per-move rewards, 1.0 for each correct move and 0.0
    for the first error (advantages then normalize per (group, step)).
    "binary": one episode reward, 1.0 iff every solver move was found (the Elo
    eval's definition of solved). "partial": fraction of solver moves found.
    """
    attempts = []
    n_groups = 0
    for puzzle in puzzles:
        if len(puzzle["moves"]) < 2:
            continue
        board = chess.Board(puzzle["fen"])
        try:
            board.push_uci(puzzle["moves"][0])
        except (chess.InvalidMoveError, chess.IllegalMoveError):
            continue
        for _ in range(G):
            attempts.append(
                {
                    "board": board.copy(),
                    "moves": puzzle["moves"],
                    "ply": 1,  # index of the expected solver move
                    "n_correct": 0,
                    "n_total": len(puzzle["moves"]) // 2,
                    "steps": [],
                    "group_id": group_id_start + n_groups,
                }
            )
        n_groups += 1
    live = attempts
    while live:
        actions, logp_old, masks = batched_policy_step(
            model, [a["board"] for a in live], extra_embedding, temperature, sample=True
        )
        next_live = []
        for k, a in enumerate(live):
            a["steps"].append((a["board"].fen(), actions[k], logp_old[k], masks[k]))
            expected = a["moves"][a["ply"]]
            if not uci_match(index2uci(actions[k].item()), expected):
                continue  # wrong move ends the episode
            a["n_correct"] += 1
            a["board"].push_uci(expected)
            if a["ply"] + 1 < len(a["moves"]):
                a["board"].push_uci(a["moves"][a["ply"] + 1])  # opponent reply
                a["ply"] += 2
                if a["ply"] < len(a["moves"]):
                    next_live.append(a)
        live = next_live
    episodes = []
    for a in attempts:
        T = len(a["steps"])
        step_rewards = None
        if reward_mode == "step":
            # all recorded moves are correct except (possibly) the last one
            step_rewards = torch.ones(T)
            if a["n_correct"] < T:
                step_rewards[-1] = 0.0
            reward = a["n_correct"] / a["n_total"]  # for logging
        elif reward_mode == "binary":
            reward = float(a["n_correct"] == a["n_total"])
        else:
            reward = a["n_correct"] / a["n_total"]
        episodes.append(
            Episode(
                fens=[s[0] for s in a["steps"]],
                actions=torch.stack([s[1] for s in a["steps"]]),
                logp_old=torch.stack([s[2] for s in a["steps"]]),
                legal_masks=torch.stack([s[3] for s in a["steps"]]),
                reward=reward,
                group_id=a["group_id"],
                source="puzzle",
                step_rewards=step_rewards,
            )
        )
    return episodes


def flatten_episodes(episodes: list[Episode], adv_eps: float = 1e-4) -> dict:
    """
    Compute group-relative advantages, then flatten to per-move tensors.

    Episode-level rewards (self-play, binary/partial puzzles): one advantage
    per episode, normalized within (source, group_id), repeated on every move,
    with 1/T per-episode length normalization.

    Per-step rewards (puzzle "step" mode): one advantage per move, normalized
    within (source, group_id, step) across the attempts that reached that step
    (they all sit at the same scripted position). Each move weighs 1.0 — every
    decision is its own single-step group.
    """
    groups = defaultdict(list)
    for ep in episodes:
        groups[(ep.source, ep.group_id)].append(ep)
    episode_adv = {}  # id(ep) -> float
    step_adv = {}  # id(ep) -> {t: float}
    for key, eps in groups.items():
        if eps[0].step_rewards is not None:
            for t in range(max(len(ep.fens) for ep in eps)):
                reached = [ep for ep in eps if len(ep.fens) > t]
                rewards = torch.tensor([[ep.step_rewards[t] for ep in reached]])
                adv = group_advantages(rewards, adv_eps).squeeze(0)
                for ep, a in zip(reached, adv):
                    step_adv.setdefault(id(ep), {})[t] = a.item()
        else:
            rewards = torch.tensor([[ep.reward for ep in eps]])
            adv = group_advantages(rewards, adv_eps).squeeze(0)
            for ep, a in zip(eps, adv):
                episode_adv[id(ep)] = a.item()
    fens, actions, logp_old, masks, adv, weights = [], [], [], [], [], []
    for ep in episodes:
        T = len(ep.fens)
        fens += ep.fens
        actions.append(ep.actions)
        logp_old.append(ep.logp_old)
        masks.append(ep.legal_masks)
        if ep.step_rewards is not None:
            adv += [step_adv[id(ep)][t] for t in range(T)]
            weights += [1.0] * T  # each step is its own decision
        else:
            adv += [episode_adv[id(ep)]] * T
            weights += [1.0 / T] * T  # per-episode length normalization
    return {
        "fens": fens,
        "actions": torch.cat(actions),
        "logp_old": torch.cat(logp_old),
        "masks": torch.cat(masks),
        "adv": torch.tensor(adv),
        "weights": torch.tensor(weights),
        "mean_reward": {
            source: sum(ep.reward for ep in episodes if ep.source == source)
            / max(1, sum(1 for ep in episodes if ep.source == source))
            for source in {ep.source for ep in episodes}
        },
    }


def sample_start_fens(n: int, max_random_plies: int = 8) -> list[str]:
    """Vary self-play starts with 0..max_random_plies uniform random plies."""
    fens = []
    for _ in range(n):
        board = chess.Board()
        for _ in range(random.randint(0, max_random_plies)):
            moves = list(board.legal_moves)
            if not moves or board.is_game_over():
                break
            board.push(random.choice(moves))
        fens.append(board.fen())
    return fens


def rollout_selfplay(
    model,
    start_fens: list[str],
    G: int,
    max_plies: int,
    extra_embedding: bool,
    temperature: float = 1.0,
    group_id_start: int = 0,
) -> list[Episode]:
    """
    The model plays both sides of G games per start position. Every ply, all
    live boards go through one batched_policy_step. Games hitting max_plies are
    adjudicated as draws. One Episode per (game, color); reward +1/-1/0 from
    that color's perspective; group = (start, color).
    """
    games = []
    for start_idx, fen in enumerate(start_fens):
        for _ in range(G):
            games.append(
                {
                    "board": chess.Board(fen),
                    "start_idx": start_idx,
                    "steps": {chess.WHITE: [], chess.BLACK: []},
                }
            )
    for _ in range(max_plies):
        live = [g for g in games if not g["board"].is_game_over()]
        if not live:
            break
        actions, logp_old, masks = batched_policy_step(
            model, [g["board"] for g in live], extra_embedding, temperature, sample=True
        )
        for k, g in enumerate(live):
            board = g["board"]
            move = index_to_move(board, actions[k].item())
            g["steps"][board.turn].append(
                (board.fen(), actions[k], logp_old[k], masks[k])
            )
            board.push(move)
    episodes = []
    for g in games:
        outcome = g["board"].outcome()
        if outcome is None or outcome.winner is None:
            rewards = {chess.WHITE: 0.0, chess.BLACK: 0.0}
        else:
            rewards = {
                chess.WHITE: 1.0 if outcome.winner == chess.WHITE else -1.0,
                chess.BLACK: 1.0 if outcome.winner == chess.BLACK else -1.0,
            }
        for color in [chess.WHITE, chess.BLACK]:
            steps = g["steps"][color]
            if not steps:
                continue
            episodes.append(
                Episode(
                    fens=[s[0] for s in steps],
                    actions=torch.stack([s[1] for s in steps]),
                    logp_old=torch.stack([s[2] for s in steps]),
                    legal_masks=torch.stack([s[3] for s in steps]),
                    reward=rewards[color],
                    group_id=group_id_start + 2 * g["start_idx"] + int(color),
                    source="selfplay",
                )
            )
    return episodes
