import tempfile
import csv

from alphaminustwo.puzzle_eval import (
    load_puzzles,
    calculate_expected_score,
    update_elo,
    evaluate_puzzle,
)
from alphaminustwo.model import GPT
from alphaminustwo.config import GPT124M


def test_calculate_expected_score():
    """Test Elo expected score calculation."""
    # Equal ratings should give 50% expected score
    assert abs(calculate_expected_score(1500, 1500) - 0.5) < 0.01

    # 400 points difference should give ~90% expected score
    assert abs(calculate_expected_score(1900, 1500) - 0.909) < 0.01

    # Lower rated should have lower expected score
    assert calculate_expected_score(1200, 1600) < 0.5


def test_estimate_elo_rating():
    """Test Elo rating estimation."""
    # If we solve all 1500-rated puzzles, our rating should be higher
    puzzle_ratings = [1500, 1500, 1500, 1500, 1500]
    actual_scores = [1.0, 1.0, 1.0, 1.0, 1.0]
    estimated = update_elo(puzzle_ratings, actual_scores, initial_elo=1500)
    assert estimated > 1700  # Should be significantly higher

    # If we fail all puzzles, rating should be lower
    actual_scores = [0.0, 0.0, 0.0, 0.0, 0.0]
    estimated = update_elo(puzzle_ratings, actual_scores, initial_elo=1500)
    assert estimated < 1300  # Should be significantly lower

    # Mixed results
    actual_scores = [1.0, 0.0, 1.0, 0.0, 1.0]
    estimated = update_elo(puzzle_ratings, actual_scores, initial_elo=1500)
    assert 1400 < estimated < 1600  # Should be around 1500


def test_load_puzzles():
    """Test puzzle loading from CSV."""
    # Create temporary CSV file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "PuzzleId",
                "FEN",
                "Moves",
                "Rating",
                "RatingDeviation",
                "Popularity",
                "NbPlays",
                "Themes",
                "GameUrl",
                "OpeningTags",
            ]
        )
        writer.writerow(
            [
                "test1",
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -",
                "e2e4 e7e5",
                "1500",
                "75",
                "90",
                "100",
                "opening",
                "https://lichess.org/test",
                "",
            ]
        )
        writer.writerow(
            [
                "test2",
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -",
                "d2d4 d7d5",
                "1600",
                "75",
                "90",
                "100",
                "opening",
                "https://lichess.org/test2",
                "",
            ]
        )
        writer.writerow(
            [
                "test3",
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -",
                "c2c4 c7c5",
                "800",
                "75",
                "90",
                "100",
                "opening",
                "https://lichess.org/test3",
                "",
            ]
        )
        csv_path = f.name

    # Load all puzzles
    puzzles = load_puzzles(csv_path)
    assert len(puzzles) == 3

    # Load with max_puzzles (randomly sampled)
    puzzles = load_puzzles(csv_path, max_puzzles=2)
    assert len(puzzles) == 2
    # Verify all IDs are valid (could be any 2 of the 3)
    valid_ids = {"test1", "test2", "test3"}
    for puzzle in puzzles:
        assert puzzle["id"] in valid_ids


def test_evaluate_puzzle():
    """Test puzzle evaluation returns correct structure."""
    # Create a simple model (won't have trained weights)
    model_cfg = GPT124M()
    model = GPT(model_cfg)
    model.eval()

    # Create a simple puzzle
    puzzle = {
        "id": "test_puzzle",
        "fen": "r6k/pp2r2p/4Rp1Q/3p4/8/1N1P2R1/PqP2bPP/7K b - - 0 24",
        "moves": ["f2g3", "e6e7", "b2b1", "b3c1"],
        "rating": 1743,
        "themes": "crushing",
    }
    evaluate_puzzle(model, puzzle)
