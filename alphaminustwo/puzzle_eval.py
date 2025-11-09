"""
Evaluation function to compute model's Elo rating on chess puzzles.

This module evaluates a chess model's puzzle-solving ability by testing it against
the Lichess puzzle database and computing an Elo rating based on performance.
"""

import csv
import random
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass

import torch
import chess
from tqdm import tqdm

from alphaminustwo.model import GPT
from alphaminustwo.dataset import fen2tensor, move2tensor


def tensor2move(index: int) -> str:
    """Convert tensor index back to UCI move string (matching model's generate_from_board)."""
    uci = ""
    for square in [index // 64, index % 64]:
        i, j = square // 8, square % 8
        uci += list("hgfedcba")[i]
        uci += str(8 - j)
    return uci


@dataclass
class PuzzleResult:
    puzzle_id: str
    puzzle_rating: int
    solved: bool
    predicted_moves: List[str]
    expected_moves: List[str]
    fen: str


@dataclass
class EloReport:
    estimated_elo: float
    num_puzzles: int
    num_solved: int
    accuracy: float
    puzzles_by_rating: Dict[str, Tuple[int, int]]
    results: List[PuzzleResult]


def load_puzzles(csv_path: str, max_puzzles: Optional[int] = None) -> List[Dict]:
    """
    Load puzzles from CSV file. If max_puzzles is specified, randomly samples puzzles.

    Args:
        csv_path: Path to the puzzle CSV file
        max_puzzles: Maximum number of puzzles to sample (random sampling if specified)

    Returns:
        List of puzzle dictionaries
    """
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


def evaluate_puzzle(model: GPT, puzzle: Dict) -> Tuple[bool, List[str]]:
    board = chess.Board(puzzle["fen"])
    moves = puzzle["moves"]
    predicted_moves = []
    for i, expected_move in enumerate(moves):
        if i % 2:
            x = fen2tensor(board.fen()).unsqueeze(0).to(model.device())
            with torch.no_grad():
                _, logits, *_ = model.forward(x, None)
            mask = torch.zeros(64**2).to(model.device())
            for move in board.legal_moves:
                mask[move2tensor(move.uci())] = 1
            logits = logits.masked_fill(mask == 0, float("-inf"))
            best_idx = torch.argmax(logits, dim=-1)
            predicted_move = tensor2move(best_idx.item())
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
    csv_path: str,
    max_puzzles: int,
    initial_elo_estimate: float,
) -> EloReport:
    """
    Evaluate a chess model on puzzles and compute its Elo rating.

    Args:
        model: The chess model to evaluate
        csv_path: Path to the puzzle CSV file
        max_puzzles: Maximum number of puzzles to evaluate
        initial_elo_estimate: Initial guess for Elo rating

    Returns:
        EloReport containing Elo rating and detailed results
    """
    model.eval()

    puzzles = load_puzzles(csv_path, max_puzzles=max_puzzles)
    results = []
    current_elo = initial_elo_estimate
    iterator = tqdm(puzzles, desc=f"Evaluating puzzles (Elo: {current_elo:.0f})")
    K = 10
    for puzzle in iterator:
        solved, predicted_moves = evaluate_puzzle(model, puzzle)
        if solved is None:
            continue
        results.append(
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
    puzzles_by_rating = {}
    for min_r, max_r in rating_bins:
        bin_name = f"{min_r}-{max_r}"
        puzzles_in_bin = [r for r in results if min_r <= r.puzzle_rating < max_r]
        solved = sum(1 for r in puzzles_in_bin if r.solved)
        total = len(puzzles_in_bin)
        puzzles_by_rating[bin_name] = (solved, total)
    report = EloReport(
        estimated_elo=current_elo,
        num_puzzles=len(results),
        num_solved=sum(1 for r in results if r.solved),
        accuracy=sum(1 for r in results if r.solved) / len(results),
        puzzles_by_rating=puzzles_by_rating,
        results=results,
    )
    print_report(report)
    return report


def print_report(report: EloReport):
    """Print a formatted evaluation report."""
    print("\n" + "=" * 60)
    print("PUZZLE EVALUATION REPORT")
    print("=" * 60)
    print(f"Estimated Elo Rating: {report.estimated_elo:.0f}")
    print(f"Total Puzzles: {report.num_puzzles}")
    print(f"Solved: {report.num_solved}")
    print(f"Accuracy: {report.accuracy:.1%}")
    print()
    print("Performance by Rating Range:")
    print("-" * 60)
    for rating_bin, (solved, total) in sorted(report.puzzles_by_rating.items()):
        acc = solved / total if total > 0 else 0
        bar_length = int(acc * 30)
        bar = "█" * bar_length + "░" * (30 - bar_length)
        print(f"{rating_bin:>12}: {bar} {solved:>4}/{total:<4} ({acc:.1%})")
    print("=" * 60)


def export_results_csv(report: EloReport, output_path: str):
    """Export detailed results to CSV file in format: elo,success,moves,ground_truth,url"""
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["elo", "success", "moves", "ground_truth", "url"])
        for result in report.results:
            elo = result.puzzle_rating
            success = "1" if result.solved else "0"
            moves = " ".join(result.predicted_moves)
            ground_truth = " ".join(result.expected_moves)
            url = f"https://lichess.org/training/{result.puzzle_id} "
            writer.writerow([elo, success, moves, ground_truth, url])
    print(f"Results exported to {output_path}")
