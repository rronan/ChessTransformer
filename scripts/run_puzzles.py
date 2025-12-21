import argparse
import torch
from datetime import datetime

from alphaminustwo.model import GPT
from alphaminustwo import config
from alphaminustwo.puzzle import (
    evaluate_model_on_puzzles,
    export_results_csv,
    print_report,
    load_puzzles,
)


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
        "--model_cfg",
        action="store_true",
        help="Use old ModelCFG for backward compatibility with old checkpoints",
    )
    parser.add_argument(
        "--save_results",
        action="store_true",
        help="Save puzzle results to CSV file with timestamp (format: puzzle_url,success,moves)",
    )
    args = parser.parse_args()
    print(args)
    return args


def main():
    args = parse_args()
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Loading model from {args.chkp} on device {device}...")
    torch.set_float32_matmul_precision("high")
    model_cfg = config.getattr(args.model_cfg)()
    model = GPT(model_cfg).to(device)
    chkp = torch.load(args.chkp, weights_only=False, map_location=torch.device(device))
    model.load_state_dict(chkp["model"])
    model.eval()
    print("Model loaded successfully!")

    puzzles = load_puzzles(args.csv_path, max_puzzles=args.max_puzzles)
    current_elo, result_list = evaluate_model_on_puzzles(
        model=model,
        puzzles=puzzles,
        initial_elo_estimate=args.initial_elo,
    )
    print_report(current_elo, result_list)
    if args.save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"puzzle_results_{timestamp}.csv"
        export_results_csv(result_list, output_path)


if __name__ == "__main__":
    main()
