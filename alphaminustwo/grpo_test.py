import math
from dataclasses import dataclass

import torch
import chess

from alphaminustwo.model import GPT
from alphaminustwo.dataset import fen2tensor, uci2index, index2uci, uci_match
from alphaminustwo.grpo import (
    masked_log_probs,
    gather_logp,
    exact_kl,
    exact_entropy,
    group_advantages,
    grpo_loss,
)
from alphaminustwo.rollout import (
    Episode,
    legal_mask_from_board,
    batched_policy_step,
    rollout_puzzles,
    rollout_selfplay,
    sample_start_fens,
    flatten_episodes,
)


@dataclass
class TinyCFG:
    square_dim: int = 18
    extra_embedding: bool = False
    n_layer: int = 2
    n_head: int = 2
    n_embd: int = 32
    bias: bool = False
    weight_loss_move: float = 1
    weight_loss_score: float = 4


def make_model():
    torch.manual_seed(0)
    model = GPT(TinyCFG())
    model.eval()
    return model


def test_masked_log_probs_matches_generate_from_board_weights():
    """The training policy must equal generate_from_board's sampling weights:
    multinomial(masked_logits.exp()) == softmax over legal moves."""
    board = chess.Board()
    mask = legal_mask_from_board(board).unsqueeze(0)
    logits = torch.randn(1, 64**2)
    logp = masked_log_probs(logits, mask)
    # -inf exactly on illegal moves, normalized over legal ones
    assert torch.isinf(logp[mask == 0]).all() and (logp[mask == 0] < 0).all()
    assert torch.allclose(logp.exp().sum(), torch.tensor(1.0), atol=1e-5)
    # same weights as generate_from_board: masked_logits.exp() renormalized
    masked = logits.masked_fill(mask == 0, float("-inf"))
    expected = masked.exp() / masked.exp().sum()
    assert torch.allclose(logp.exp(), expected, atol=1e-5)


def test_legal_mask_matches_board():
    board = chess.Board()
    mask = legal_mask_from_board(board)
    legal_indices = {uci2index(m.uci()).item() for m in board.legal_moves}
    assert set(torch.nonzero(mask).flatten().tolist()) == legal_indices


def test_batched_policy_step_argmax_is_legal_and_consistent():
    model = make_model()
    boards = [chess.Board(), chess.Board("8/8/8/4k3/8/8/4P3/4K3 w - - 0 1")]
    actions, logp_chosen, masks = batched_policy_step(
        model, boards, extra_embedding=False, sample=False
    )
    for k, board in enumerate(boards):
        assert masks[k, actions[k]]  # argmax is legal
        # matches a single-board manual forward (evaluate_puzzle convention)
        x = torch.stack([fen2tensor(board.fen(), False)])
        with torch.no_grad():
            _, logits, *_ = model(x, None)
        masked = logits[0].masked_fill(legal_mask_from_board(board) == 0, float("-inf"))
        assert actions[k].item() == masked.argmax().item()
        assert torch.allclose(
            logp_chosen[k], masked_log_probs(masked.unsqueeze(0), legal_mask_from_board(board).unsqueeze(0))[0, actions[k]], atol=1e-5
        )


def test_exact_kl_and_entropy():
    board = chess.Board()
    mask = legal_mask_from_board(board).unsqueeze(0)
    logp_a = masked_log_probs(torch.randn(1, 64**2), mask)
    logp_b = masked_log_probs(torch.randn(1, 64**2), mask)
    assert torch.allclose(exact_kl(logp_a, logp_a), torch.tensor([0.0]), atol=1e-5)
    assert exact_kl(logp_a, logp_b).item() >= 0
    # uniform over k legal moves -> entropy == log(k)
    k = int(mask.sum())
    logp_uniform = masked_log_probs(torch.zeros(1, 64**2), mask)
    assert math.isclose(exact_entropy(logp_uniform).item(), math.log(k), rel_tol=1e-4)


def test_group_advantages():
    rewards = torch.tensor([[1.0, 0.0, 0.0, 1.0]])
    adv = group_advantages(rewards)
    assert torch.allclose(adv.mean(), torch.tensor(0.0), atol=1e-5)
    # constant-reward group -> zero advantage
    assert group_advantages(torch.tensor([[1.0, 1.0, 1.0]])).abs().max() == 0
    # single-episode group -> zero, not nan
    assert group_advantages(torch.tensor([[1.0]])).item() == 0


def test_grpo_loss_gradient_sign():
    board = chess.Board()
    mask = legal_mask_from_board(board).unsqueeze(0)
    logits = torch.zeros(1, 64**2, requires_grad=True)
    logp = masked_log_probs(logits, mask)
    action = torch.nonzero(mask[0]).flatten()[:1]
    logp_new = gather_logp(logp, action)
    for advantage, sign in [(1.0, 1.0), (-1.0, -1.0)]:
        if logits.grad is not None:
            logits.grad = None
        loss, metrics = grpo_loss(
            logp_new=gather_logp(masked_log_probs(logits, mask), action),
            logp_old=logp_new.detach(),
            advantages=torch.tensor([advantage]),
            kl=torch.tensor([0.0]),
            entropy=torch.tensor([0.0]),
            clip_eps=0.2,
            kl_beta=0.0,
            entropy_coef=0.0,
        )
        loss.backward()
        # minimizing loss must push the chosen-action logit in the sign of A
        assert -logits.grad[0, action].item() * sign > 0
        assert "clip_frac" in metrics and "mean_kl" in metrics


def test_rollout_puzzles_reward_semantics():
    """Multi-move puzzle: moves alternate [opponent, solver, opponent, solver].
    The episode follows the line until a wrong move; reward is binary on the
    full line (or fractional in partial mode)."""
    model = make_model()
    puzzles = [
        {
            "id": "p1",
            "fen": "r6k/pp2r2p/4Rp1Q/3p4/8/1N1P2R1/PqP2bPP/7K b - - 0 24",
            "moves": ["f2g3", "e6e7", "b2b1", "b3c1"],
            "rating": 1743,
            "themes": "crushing",
        }
    ]
    G = 8
    solver_moves = puzzles[0]["moves"][1::2]
    episodes = rollout_puzzles(
        model, puzzles, G=G, extra_embedding=False, reward_mode="binary"
    )
    assert len(episodes) == G
    board0 = chess.Board(puzzles[0]["fen"])
    board0.push_uci(puzzles[0]["moves"][0])
    for ep in episodes:
        assert ep.source == "puzzle" and ep.group_id == 0
        assert 1 <= len(ep.actions) <= len(solver_moves)
        assert ep.fens[0] == board0.fen()
        n_correct = 0
        for t, (action, mask) in enumerate(zip(ep.actions, ep.legal_masks)):
            assert mask[action]  # every sampled move was legal
            if uci_match(index2uci(action.item()), solver_moves[t]):
                n_correct += 1
        # episode continues only through correct moves; wrong move ends it
        if len(ep.actions) < len(solver_moves) or n_correct < len(solver_moves):
            assert not uci_match(
                index2uci(ep.actions[-1].item()), solver_moves[len(ep.actions) - 1]
            )
            assert n_correct == len(ep.actions) - 1
            assert ep.reward == 0.0
        else:
            assert ep.reward == 1.0
    # partial mode: reward = fraction of solver moves found
    episodes_partial = rollout_puzzles(
        model, puzzles, G=G, extra_embedding=False, reward_mode="partial"
    )
    for ep in episodes_partial:
        n_correct = sum(
            uci_match(index2uci(a.item()), solver_moves[t])
            for t, a in enumerate(ep.actions)
        )
        assert ep.reward == n_correct / len(solver_moves)
    # step mode (default): 1.0 per correct move, 0.0 on the first error
    episodes_step = rollout_puzzles(
        model, puzzles, G=G, extra_embedding=False, reward_mode="step"
    )
    for ep in episodes_step:
        assert ep.step_rewards is not None
        assert len(ep.step_rewards) == len(ep.actions)
        for t, (a, r) in enumerate(zip(ep.actions, ep.step_rewards)):
            correct = uci_match(index2uci(a.item()), solver_moves[t])
            assert r.item() == float(correct)
        # only the last step may be wrong
        assert (ep.step_rewards[:-1] == 1.0).all()


def test_flatten_episodes_per_step_advantages():
    """Step-mode advantages normalize per (group, step) across the attempts
    that reached that step."""

    def make_ep(step_rewards, group_id=0):
        T = len(step_rewards)
        return Episode(
            fens=["fen"] * T,
            actions=torch.zeros(T, dtype=torch.long),
            logp_old=torch.zeros(T),
            legal_masks=torch.zeros(T, 64**2, dtype=torch.bool),
            reward=sum(step_rewards) / 2,
            group_id=group_id,
            source="puzzle",
            step_rewards=torch.tensor(step_rewards),
        )

    # A solved both moves, B failed move 1, C failed move 2
    ep_a, ep_b, ep_c = make_ep([1.0, 1.0]), make_ep([0.0]), make_ep([1.0, 0.0])
    batch = flatten_episodes([ep_a, ep_b, ep_c], adv_eps=1e-4)
    adv = batch["adv"]
    a0, a1, b0, c0, c1 = adv[0], adv[1], adv[2], adv[3], adv[4]
    # step 0: A and C right, B wrong -> A,C > 0 > B; zero-mean over the trio
    assert a0 > 0 and c0 > 0 and b0 < 0 and a0 == c0
    assert abs(a0 + c0 + b0) < 1e-5
    # step 1: only A and C reached it; A right, C wrong (B not a participant)
    assert a1 > 0 > c1
    assert abs(a1 + c1) < 1e-5
    # step weights are 1.0 (each step its own decision)
    assert (batch["weights"] == 1.0).all()
    # episode-level path still works alongside (self-play episode in the mix)
    ep_sp = Episode(
        fens=["f1", "f2"],
        actions=torch.zeros(2, dtype=torch.long),
        logp_old=torch.zeros(2),
        legal_masks=torch.zeros(2, 64**2, dtype=torch.bool),
        reward=1.0,
        group_id=0,
        source="selfplay",
    )
    ep_sp2 = Episode(
        fens=["f1"],
        actions=torch.zeros(1, dtype=torch.long),
        logp_old=torch.zeros(1),
        legal_masks=torch.zeros(1, 64**2, dtype=torch.bool),
        reward=-1.0,
        group_id=0,
        source="selfplay",
    )
    batch = flatten_episodes([ep_a, ep_b, ep_c, ep_sp, ep_sp2], adv_eps=1e-4)
    # self-play moves carry the episode advantage with 1/T weights
    assert batch["adv"][5] == batch["adv"][6] > 0 > batch["adv"][7]
    assert batch["weights"][5] == 0.5 and batch["weights"][7] == 1.0


def test_uci_match_queen_promotion_convention():
    assert uci_match("e2e4", "e2e4")
    assert uci_match("e7e8", "e7e8q")  # bare from-to means queen promotion
    assert not uci_match("e7e8", "e7e8n")  # underpromotions never match
    assert not uci_match("e7e8", "e7e8r")
    assert not uci_match("e2e4", "d2d4")


def test_rollout_selfplay_smoke():
    model = make_model()
    start_fens = sample_start_fens(2, max_random_plies=4)
    episodes = rollout_selfplay(
        model, start_fens, G=2, max_plies=10, extra_embedding=False
    )
    assert episodes
    by_game = {}
    for ep in episodes:
        assert ep.source == "selfplay"
        assert ep.reward in (-1.0, 0.0, 1.0)
        assert len(ep.fens) == len(ep.actions) == len(ep.logp_old)
        assert ep.legal_masks.shape == (len(ep.fens), 64**2)
        # every recorded action was legal at its recorded position
        for fen, action, mask in zip(ep.fens, ep.actions, ep.legal_masks):
            board = chess.Board(fen)
            assert mask[action]
            legal = {uci2index(m.uci()).item() for m in board.legal_moves}
            assert action.item() in legal
    # decided games: white/black rewards negate (group ids 2*start + color)
    rewards_by_start = {}
    for ep in episodes:
        rewards_by_start.setdefault(ep.group_id // 2, []).append(ep.reward)


def test_flatten_and_train_step_runs():
    """Mini end-to-end: rollout -> advantages -> one GRPO update step."""
    model = make_model()
    ref_model = make_model()
    puzzles = [
        {
            "id": "p1",
            "fen": "r6k/pp2r2p/4Rp1Q/3p4/8/1N1P2R1/PqP2bPP/7K b - - 0 24",
            "moves": ["f2g3", "e6e7"],
            "rating": 1743,
            "themes": "t",
        }
    ]
    episodes = rollout_puzzles(model, puzzles, G=8, extra_embedding=False)
    rewards = torch.tensor([[ep.reward for ep in episodes]])
    adv = group_advantages(rewards).squeeze(0)
    x = torch.stack([fen2tensor(ep.fens[0], False) for ep in episodes])
    masks = torch.cat([ep.legal_masks for ep in episodes])
    actions = torch.cat([ep.actions for ep in episodes])
    logp_old = torch.cat([ep.logp_old for ep in episodes])
    model.train()
    _, logits, *_ = model(x, None)
    logp = masked_log_probs(logits, masks)
    with torch.no_grad():
        _, ref_logits, *_ = ref_model(x, None)
        logp_ref = masked_log_probs(ref_logits, masks)
    loss, metrics = grpo_loss(
        logp_new=gather_logp(logp, actions),
        logp_old=logp_old,
        advantages=adv,
        kl=exact_kl(logp, logp_ref),
        entropy=exact_entropy(logp),
        clip_eps=0.2,
        kl_beta=0.02,
        entropy_coef=0.0,
    )
    loss.backward()
    assert torch.isfinite(loss)
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    # behavior policy == current policy on first epoch -> ratio ~ 1, no clipping
    assert abs(metrics["mean_ratio"] - 1.0) < 1e-3
    assert metrics["clip_frac"] == 0.0
    assert metrics["mean_kl"] >= 0
