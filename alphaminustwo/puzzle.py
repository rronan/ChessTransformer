import csv
import random
from pydantic import BaseModel

import torch
import chess
from tqdm import tqdm

from alphaminustwo.model import GPT
from alphaminustwo.dataset import fen2tensor, uci2index, index2uci

K = 10


class PuzzleResult(BaseModel):
    puzzle_id: str
    puzzle_rating: int
    solved: bool
    predicted_moves: list[str]
    expected_moves: list[str]
    fen: str


def load_puzzles(csv_path: str, max_puzzles: int | None = None) -> list[dict]:
    puzzles = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            puzzles.append(
                {
                    "id": row["PuzzleId"],
                    "fen": row["FEN"],
                    "moves": row["Moves"].split(),
                    "rating": int(row["Rating"]),
                    "themes": row["Themes"],
                }
            )
    if max_puzzles is not None and len(puzzles) > max_puzzles:
        puzzles = random.sample(puzzles, max_puzzles)
    return puzzles


def evaluate_puzzle(
    model: GPT, puzzle: dict, extra_embedding: bool
) -> tuple[bool, list[str]]:
    board = chess.Board(puzzle["fen"])
    moves = puzzle["moves"]
    predicted_moves = []
    for i, expected_move in enumerate(moves):
        if i % 2:
            x = fen2tensor(board.fen(), extra_embedding).unsqueeze(0).to(model.device())
            with torch.no_grad():
                _, logits, *_ = model.forward(x, None)
            mask = torch.zeros(64**2).to(model.device())
            for move in board.legal_moves:
                mask[uci2index(move.uci())] = 1
            logits = logits.masked_fill(mask == 0, float("-inf"))
            best_idx = torch.argmax(logits, dim=-1)
            predicted_move = index2uci(best_idx.item())
            predicted_moves.append(predicted_move)
            if predicted_move != expected_move:
                return False, predicted_moves
        try:
            board.push_uci(expected_move)
        except (chess.InvalidMoveError, chess.IllegalMoveError):
            return None, predicted_moves
    return True, predicted_moves


def evaluate_model_on_puzzles(
    model: GPT,
    puzzles: list[dict],
    initial_elo_estimate: float,
    extra_embedding: bool,
) -> tuple[float, list[PuzzleResult | None]]:
    model.eval()
    result_list: list[PuzzleResult] = []
    current_elo = initial_elo_estimate
    iterator = tqdm(puzzles, desc=f"Evaluating puzzles (Elo: {current_elo:.0f})")
    for puzzle in iterator:
        solved, predicted_moves = evaluate_puzzle(model, puzzle, extra_embedding)
        if solved is None:
            continue
        result_list.append(
            PuzzleResult(
                puzzle_id=puzzle["id"],
                puzzle_rating=puzzle["rating"],
                solved=solved,
                predicted_moves=predicted_moves,
                expected_moves=puzzle["moves"],
                fen=puzzle["fen"],
            )
        )
        score = float(solved) if solved else 0.0
        expected_score = 1 / (1 + 10 ** ((puzzle["rating"] - current_elo) / 400))
        current_elo += K * (score - expected_score)
        iterator.set_description(f"Evaluating puzzles (Elo: {current_elo:.0f})")
    return current_elo, result_list


def print_report(estimated_elo: float, result_list: list[PuzzleResult]):
    print("\n" + "=" * 60)
    print("PUZZLE EVALUATION REPORT")
    print("=" * 60)
    print(f"Estimated Elo Rating: {estimated_elo:.0f}")
    print(f"Total Puzzles: {len(result_list)}")
    print(f"Solved: {sum(1 for r in result_list if r.solved)}")
    print(f"Accuracy: {sum(1 for r in result_list if r.solved) / len(result_list):.1%}")
    print()
    print("Performance by Rating Range:")
    print("-" * 60)
    rating_bins = [
        (0, 1000),
        (1000, 1200),
        (1200, 1400),
        (1400, 1600),
        (1600, 1800),
        (1800, 2000),
        (2000, 2200),
        (2200, 2400),
        (2400, 9999),
    ]
    for min_r, max_r in rating_bins:
        bin_name = f"{min_r}-{max_r}"
        puzzles_in_bin = [r for r in result_list if min_r <= r.puzzle_rating < max_r]
        solved = sum(1 for r in puzzles_in_bin if r.solved)
        total = len(puzzles_in_bin)
        acc = solved / total if total > 0 else 0
        bar_length = int(acc * 30)
        bar = "█" * bar_length + "░" * (30 - bar_length)
        print(f"{bin_name:>12}: {bar} {solved:>4}/{total:<4} ({acc:.1%})")
    print("=" * 60)


def export_results_csv(result_list: list[PuzzleResult], output_path: str):
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["elo", "success", "moves", "ground_truth", "url"])
        for result in result_list:
            elo = result.puzzle_rating
            success = "1" if result.solved else "0"
            moves = " ".join(result.predicted_moves)
            ground_truth = " ".join(result.expected_moves)
            url = f"https://lichess.org/training/{result.puzzle_id} "
            writer.writerow([elo, success, moves, ground_truth, url])
    print(f"Results exported to {output_path}")
