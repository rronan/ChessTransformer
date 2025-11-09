"""
Script to evaluate a chess model on Lichess puzzles and compute Elo rating.

Usage:
    python scripts/run_puzzle_eval.py --chkp path/to/checkpoint.pt --max_puzzles 1000
"""

import argparse
import torch
from datetime import datetime

from alphaminustwo.model import GPT
from alphaminustwo.config import GPT124M, ModelCFG
from alphaminustwo.puzzle_eval import evaluate_model_on_puzzles, export_results_csv


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate chess model on puzzles and compute Elo rating"
    )
    parser.add_argument(
        "--chkp", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="data/lichess_db_puzzle.csv",
        help="Path to puzzle CSV file",
    )
    parser.add_argument(
        "--max_puzzles",
        type=int,
        default=1000,
        help="Maximum number of puzzles to evaluate",
    )
    parser.add_argument(
        "--initial_elo",
        type=float,
        default=1500,
        help="Initial Elo estimate for optimization",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu, defaults to auto-detect)",
    )
    parser.add_argument(
        "--use_old_config",
        action="store_true",
        help="Use old ModelCFG for backward compatibility with old checkpoints",
    )
    parser.add_argument(
        "--save_results",
        action="store_true",
        help="Save puzzle results to CSV file with timestamp (format: puzzle_url,success,moves)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Setup device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from {args.chkp}...")
    torch.set_float32_matmul_precision("high")
    model_cfg = ModelCFG() if args.use_old_config else GPT124M()
    print(f"Using config: {model_cfg.__class__.__name__}")
    model = GPT(model_cfg).to(device)

    chkp = torch.load(args.chkp, weights_only=False, map_location=torch.device(device))
    model.load_state_dict(chkp["model"])
    model.eval()

    print("Model loaded successfully!")

    # Run evaluation
    report = evaluate_model_on_puzzles(
        model=model,
        csv_path=args.csv_path,
        max_puzzles=args.max_puzzles,
        initial_elo_estimate=args.initial_elo,
    )

    # Save results if requested
    if args.save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"puzzle_results_{timestamp}.csv"
        export_results_csv(report, output_path)

    return report


if __name__ == "__main__":
    main()
